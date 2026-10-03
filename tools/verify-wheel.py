"""Checks that a built wheel carries the plugin layout VapourSynth expects.

GitHub Actions runs this on every native wheel before it is uploaded, so a wheel
that lost a CPU variant, its manifest or the manifest's plugin name fails the
build instead of shipping.

    python tools/verify-wheel.py dist/vapoursynth_nimages-0.1.0-....whl
"""

from __future__ import annotations

import argparse
import glob
import pathlib
import sys
import zipfile
from typing import NamedTuple

#: Where VapourSynth looks for a plugin directory.
PLUGIN_DIRECTORY = "vapoursynth/plugins/nimages"

#: The manifest header VapourSynth reads.
MANIFEST_HEADER = "[VapourSynth Manifest V1]"


class Shape(NamedTuple):
    """The names one platform's plugin tree uses."""

    #: The stem cargo's `cdylib` target writes. This is also the name the
    #: manifest lists, so it carries the platform's prefix on macOS and Linux.
    stem: str
    extension: str
    #: Suffixes for the CPU variants that platform ships, shortest first.
    suffixes: tuple[str, ...]

    def filenames(self) -> list[str]:
        return [f"{self.stem}{suffix}{self.extension}" for suffix in self.suffixes]


def shape(wheel_name: str) -> Shape:
    """Resolves the layout from the wheel's platform tag."""
    if "win" in wheel_name or "mingw" in wheel_name:
        return Shape("vs_nimages", ".dll", ("", ".avx2"))
    if "macos" in wheel_name or "darwin" in wheel_name:
        # A macOS wheel is arm64 only, so it has one optimization level.
        return Shape("libvs_nimages", ".dylib", ("",))
    return Shape("libvs_nimages", ".so", ("", ".avx2"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", nargs="+", help="a wheel path or a glob")
    arguments = parser.parse_args()

    # CI passes a shell glob, which bash and PowerShell expand differently, so
    # the pattern is resolved here instead.
    wheels: list[pathlib.Path] = []
    for argument in arguments.wheel:
        matches = [pathlib.Path(match) for match in glob.glob(str(argument))]  # ruff: ignore[glob]
        wheels.extend(matches or [argument])

    failed = False
    for wheel in wheels:
        layout = shape(wheel.name)
        manifest_entry = f"{PLUGIN_DIRECTORY}/manifest.vs"
        with zipfile.ZipFile(wheel) as archive:
            names = set(archive.namelist())
            if manifest_entry not in names:
                print(f"{wheel.name}: missing {manifest_entry}", file=sys.stderr)
                failed = True
                continue
            manifest = archive.read(manifest_entry).decode()

        for filename in layout.filenames():
            wanted = f"{PLUGIN_DIRECTORY}/{filename}"
            if wanted not in names:
                print(f"{wheel.name}: missing {wanted}", file=sys.stderr)
                failed = True

        # The manifest names the plugin once, with the platform's library
        # prefix and without a variant suffix; VapourSynth adds the suffix
        # itself when the host CPU supports one.
        lines = manifest.splitlines()
        if lines != [MANIFEST_HEADER, layout.stem]:
            print(
                f"{wheel.name}: the manifest is {lines!r}, not "
                f"{[MANIFEST_HEADER, layout.stem]!r}",
                file=sys.stderr,
            )
            failed = True

        print(f"{wheel.name}: {layout.stem} with {', '.join(layout.filenames())}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
