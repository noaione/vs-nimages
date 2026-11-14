#!/usr/bin/env python3
"""Compare nmanga's book-enabled Pillow pipeline with native PeakStats + Levels.

With no --directory, generate a deterministic lossless PNG book. Each timed
pipeline/repeat runs in a fresh process. The book prepass is included, output
encoding is excluded, and a separate untimed pass hashes every output plane.

    .venv/bin/python tools/book_peak_bench.py --gpu --threads 1 --threads 4
    .venv/bin/python tools/book_peak_bench.py --directory sandbox/level-check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from PIL import Image

from nmanga_reference import BookReference

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REFERENCE = ROOT.parent / "nao-manga-rls"
DEFAULT_OUTPUT = ROOT / "target" / "bench" / "book-peak.json"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
KEYS = {
    "black": "NImagesBlackLevel",
    "white": "NImagesWhiteLevel",
    "black_found": "NImagesBlackPeakFound",
    "white_found": "NImagesWhitePeakFound",
    "eligible": "NImagesLevelEligible",
    "used_book_peak": "NImagesBookPeakUsed",
}


def parallel(function, items, threads: int) -> list:
    if threads == 1:
        return [function(item) for item in items]
    with ThreadPoolExecutor(max_workers=threads) as pool:
        return list(pool.map(function, items))


def peak_rss() -> int:
    if sys.platform == "win32":
        from bench import peak_rss_bytes
        return peak_rss_bytes()
    import resource
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def digest(planes) -> str:
    result = hashlib.sha256()
    for plane in planes:
        result.update(plane)
    return result.hexdigest()


def run_reference(paths: list[Path], args) -> dict:
    reference = BookReference(args.nmanga_path)
    # Exclude lazy scipy/numpy imports from the measurement.
    reference.analyze_page(paths[0], 60, 0.25, None, args.no_white)

    def analyze(path: Path):
        return reference.analyze_page(path, 60, 0.25, None, args.no_white)

    start = time.perf_counter()
    pages = dict(parallel(analyze, paths, args.worker_threads))
    shared, resolved = reference.resolve_pages(pages, args.peak_offset, 8, 55)
    scan = time.perf_counter() - start

    def process(path: Path, *, hash_pixels: bool = False) -> dict:
        page = pages[path]
        levels = resolved[path]
        eligible = levels.black_level is not None
        black = page.analysis.black_level if not eligible else levels.black_level
        record = {
            "name": path.name,
            "black": black,
            "white": levels.white_level,
            "black_found": int(page.analysis.black_peak_found),
            "white_found": int(page.analysis.white_peak_found),
            "eligible": int(eligible),
            "used_book_peak": int(levels.used_book_peak),
        }
        # Match the Pillow application: gamma is calculated before the offset.
        with Image.open(path) as image:
            image.load()
            adjusted = None
            if eligible:
                adjusted = reference.autolevel.apply_levels(
                    image,
                    black_point=black + args.peak_offset,
                    white_point=255 if args.no_white else levels.white_level,
                    gamma=reference.autolevel.gamma_correction(black),
                )
            output = image if adjusted is None else adjusted
            try:
                if hash_pixels:
                    values = np.asarray(output)
                    planes = [values.tobytes()] if values.ndim == 2 else [
                        values[:, :, plane].tobytes() for plane in range(values.shape[2])
                    ]
                    record["pixels"] = digest(planes)
            finally:
                if adjusted is not None:
                    adjusted.close()
        return record

    start = time.perf_counter()
    parallel(process, paths, args.worker_threads)
    process_seconds = time.perf_counter() - start
    records = [process(path, hash_pixels=True) for path in paths]
    return {"scan": scan, "process": process_seconds, "consensus": shared, "records": records}


def run_native(paths: list[Path], args) -> dict:
    import vapoursynth as vs

    core = vs.core
    core.num_threads = args.worker_threads
    core.max_cache_size = args.cache
    gpu = int(args.pipeline == "vapoursynth-gpu")
    # Warm the binding and device before timing a new book graph.
    warm = core.std.BlankClip(width=8, height=8, format=vs.GRAY8, length=1)
    warm = core.nimages.Levels(warm, black=12, white=255, gpu=gpu)
    if gpu:
        warm = core.std.GPUDownload(warm)
    with warm.get_frame(0):
        pass

    start = time.perf_counter()
    source = core.imgseqs.Read(files=[str(path) for path in paths], mismatch=True, prefetch=0)
    stats = core.nimages.PeakStats(
        source, upper_limit=60, peak_percentage=0.25, skip_white=int(args.no_white),
        book_peak_fallback=1, levelling_lower_limit=8, levelling_upper_limit=55,
        peak_offset=args.peak_offset, gpu=gpu,
    )
    scan = time.perf_counter() - start

    start = time.perf_counter()
    leveled = core.nimages.Levels(
        stats, use_props=True, peak_offset=args.peak_offset, auto_gamma=True, gpu=gpu,
    )
    plain = core.std.GPUDownload(stats) if gpu else stats
    output = core.std.GPUDownload(leveled) if gpu else leveled

    def process(n: int, *, hash_pixels: bool = False) -> dict:
        with stats.get_frame(n) as frame:
            record = {key: int(frame.props[prop]) for key, prop in KEYS.items()}
        record["name"] = paths[n].name
        # Never evaluate Levels on an excluded page, including all-black pages.
        with (output if record["eligible"] else plain).get_frame(n) as frame:
            if hash_pixels:
                record["pixels"] = digest(
                    np.asarray(frame[plane]).tobytes() for plane in range(frame.format.num_planes)
                )
        return record

    parallel(process, range(len(paths)), args.worker_threads)
    process_seconds = time.perf_counter() - start
    records = [process(n, hash_pixels=True) for n in range(len(paths))]
    return {"scan": scan, "process": process_seconds, "records": records}


def synthetic_book(directory: Path, width: int, height: int, pages: int) -> list[Path]:
    """Eight page types exercise majority voting, fallback and exclusion."""
    paths = []
    for n in range(pages):
        values = np.full((height, width), 255, dtype=np.uint8)
        kind = n % 8
        if kind < 4 or kind == 7:
            values[height // 4:height // 2, width // 4:3 * width // 4] = 128
            level = 12 if kind < 3 else (20 if kind == 3 else 0)
            values[height // 8:height // 8 + max(3, height // 20), width // 8:7 * width // 8] = level
        elif kind == 4:
            values[height // 2:height // 2 + 3, width // 2:width // 2 + 3] = 20
        elif kind == 5:
            values[height // 2, width // 2] = 20
        path = directory / f"{n:04d}.png"
        with Image.fromarray(values) as image:
            image.save(path)
        paths.append(path)
    return paths


def spawn(paths: list[Path], args, pipeline: str, threads: int, directory: Path, repeat: int) -> dict:
    request = directory / "request.json"
    response = directory / "response.json"
    request.write_text(json.dumps([str(path) for path in paths]), encoding="utf-8")
    command = [
        sys.executable, str(Path(__file__).resolve()), "--worker", "--pipeline", pipeline,
        "--request", str(request), "--response", str(response), "--worker-threads", str(threads),
        "--cache", str(args.cache), "--nmanga-path", str(args.nmanga_path),
        "--peak-offset", str(args.peak_offset),
    ]
    if args.no_white:
        command.append("--no-white")
    print(f"  {pipeline}, threads={threads}, repeat={repeat + 1}", flush=True)
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise SystemExit(f"{pipeline} failed:\n{completed.stdout}\n{completed.stderr}")
    return json.loads(response.read_text(encoding="utf-8"))


def agreement(reference: dict, candidate: dict) -> dict:
    expected = {page["name"]: page for page in reference["records"]}
    decisions = 0
    pixels = 0
    mismatches = []
    for page in candidate["records"]:
        want = expected[page["name"]]
        different = [key for key in KEYS if page[key] != want[key]]
        decisions += not different
        same_pixels = page["pixels"] == want["pixels"]
        pixels += same_pixels
        if different or not same_pixels:
            mismatches.append({"name": page["name"], "fields": different, "pixels": same_pixels})
    return {"decisions": decisions, "pixels": pixels, "pages": len(expected), "mismatches": mismatches}


def report(results: list[dict]) -> str:
    lines = [
        "| book | threads | pipeline | scan | output frames | total | peak rss | decisions | pixels |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    references = {(row["book"], row["threads"]): row for row in results if row["pipeline"] == "nmanga"}
    for row in results:
        same = row["agreement"]
        lines.append(
            f"| {row['book']} | {row['threads']} | {row['pipeline']} | {row['scan']:.3f} s | "
            f"{row['process']:.3f} s | {row['total']:.3f} s | {row['peak_rss'] / 2**20:.0f} MiB | "
            f"{same['decisions']}/{same['pages']} | {same['pixels']}/{same['pages']} |"
        )
    lines += ["", "| book | threads | pipeline | total speedup vs Pillow |", "| --- | ---: | --- | ---: |"]
    for row in results:
        if row["pipeline"] != "nmanga":
            reference = references[row["book"], row["threads"]]
            lines.append(f"| {row['book']} | {row['threads']} | {row['pipeline']} | {reference['total'] / row['total']:.2f}x |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--directory", type=Path, help="read a real book in place; never copy its pages")
    parser.add_argument("--size", action="append", help="synthetic WIDTHxHEIGHT; may be repeated")
    parser.add_argument("--pages", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, action="append", help="worker counts, default 1")
    parser.add_argument("--cache", type=int, default=512, help="VapourSynth frame cache, MiB")
    parser.add_argument("--gpu", action="store_true", help="also measure native Vulkan mapping")
    parser.add_argument("--no-white", action="store_true")
    parser.add_argument("--peak-offset", type=int, default=0)
    parser.add_argument("--nmanga-path", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write", type=Path, help="replace this document's book-bench result block")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--pipeline", choices=("nmanga", "vapoursynth", "vapoursynth-gpu"), help=argparse.SUPPRESS)
    parser.add_argument("--request", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--response", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-threads", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        paths = [Path(path) for path in json.loads(args.request.read_text(encoding="utf-8"))]
        result = run_reference(paths, args) if args.pipeline == "nmanga" else run_native(paths, args)
        result["peak_rss"] = peak_rss()
        result["total"] = result["scan"] + result["process"]
        args.response.write_text(json.dumps(result), encoding="utf-8")
        return 0
    threads = args.threads or [1]
    if args.repeats < 1 or args.pages < 8 or args.cache < 1 or min(threads) < 1:
        parser.error("use positive repeats/cache/threads and at least eight synthetic pages")
    pipelines = ["nmanga", "vapoursynth"] + (["vapoursynth-gpu"] if args.gpu else [])
    results = []
    with TemporaryDirectory(prefix="nimages-book-bench-") as directory:
        root = Path(directory)
        books = []
        if args.directory:
            paths = sorted(path.resolve() for path in args.directory.iterdir()
                           if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
            if not paths:
                parser.error("the book directory contains no supported images")
            books.append((args.directory.name, paths))
        else:
            for size in args.size or ["768x512", "2902x4128"]:
                try:
                    width, height = map(int, size.lower().split("x"))
                    if min(width, height) < 64:
                        raise ValueError
                except ValueError:
                    parser.error("sizes must be WIDTHxHEIGHT, with both dimensions at least 64")
                book_dir = root / size
                book_dir.mkdir()
                books.append((size, synthetic_book(book_dir, width, height, args.pages)))
        for book, paths in books:
            for path in paths:
                with Image.open(path) as image:
                    if image.mode not in ("L", "RGB"):
                        parser.error("the benchmark compares 8-bit L/RGB images without alpha or palette conversion")
            print(f"book {book}: {len(paths)} pages", flush=True)
            for count in threads:
                reference = None
                for pipeline in pipelines:
                    runs = [spawn(paths, args, pipeline, count, root, repeat) for repeat in range(args.repeats)]
                    first = runs[0]
                    if pipeline == "nmanga":
                        reference = first
                    agreements = [agreement(reference, run) for run in runs]
                    same = min(agreements, key=lambda value: (value["decisions"], value["pixels"]))
                    row = {
                        "book": book, "pages": len(paths), "threads": count, "pipeline": pipeline,
                        "scan": statistics.median(run["scan"] for run in runs),
                        "process": statistics.median(run["process"] for run in runs),
                        "total": statistics.median(run["total"] for run in runs),
                        "peak_rss": max(run["peak_rss"] for run in runs),
                        "eligible": sum(page["eligible"] for page in first["records"]),
                        "used_book_peak": sum(page["used_book_peak"] for page in first["records"]),
                        "agreement": same, "runs": runs,
                    }
                    results.append(row)
    payload = {
        "platform": platform.platform(), "python": platform.python_version(),
        "numpy": np.__version__, "pillow": Image.__version__,
        "repeats": args.repeats, "cache_mb": args.cache, "peak_offset": args.peak_offset,
        "no_white": args.no_white, "synthetic": args.directory is None,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    block = report(results)
    print("\n" + block)
    print(f"\nraw results: {args.output}")
    if args.write:
        text = args.write.read_text(encoding="utf-8")
        before, rest = text.split("<!-- book-bench:start -->", 1)
        _, after = rest.split("<!-- book-bench:end -->", 1)
        args.write.write_text(before + "<!-- book-bench:start -->\n" + block + "\n<!-- book-bench:end -->" + after,
                              encoding="utf-8")
    return int(any(row["agreement"]["decisions"] != row["pages"] or row["agreement"]["pixels"] != row["pages"]
                   for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
