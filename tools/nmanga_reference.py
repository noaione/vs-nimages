"""Use nmanga's public page analysis and levelling policy as the reference."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from PIL import Image


class BookReference:
    """The original file analyzer and resolver, with no copied policy code."""

    def __init__(self, path: Path) -> None:
        path = path.resolve()
        sys.path.insert(0, str(path))
        self.autolevel = importlib.import_module("nmanga.autolevel")
        if Path(self.autolevel.__file__).resolve() != path / "nmanga" / "autolevel.py":
            raise RuntimeError("nmanga was already imported from a different checkout")
        for name in ("analyze_book_peak_page", "resolve_book_peaks", "LevellingPolicy"):
            if not hasattr(self.autolevel, name):
                raise RuntimeError(f"nmanga is missing its public {name} reference")

    def analyze_page(self, path, upper_limit, peak_min_pct, peak_prom_pct, no_white, median_filter=True):
        if median_filter:
            return self.autolevel.analyze_book_peak_page(
                path, upper_limit=upper_limit, peak_min_pct=peak_min_pct,
                peak_prom_pct=peak_prom_pct, no_white=no_white,
            )
        with Image.open(path) as image:
            analysis = self.autolevel._find_local_peak_analysis(
                image, upper_limit=upper_limit, peak_percentage=peak_min_pct,
                peak_prominence=peak_prom_pct, skip_white_check=no_white, median_filter=False,
            )
        return path, analysis

    def resolve_pages(self, pages, offset, lower, upper, tolerance=6):
        policy = self.autolevel.LevellingPolicy(
            lower_limit=lower, upper_limit=upper, peak_offset=offset, book_peak_tolerance=tolerance,
        )
        return self.autolevel.resolve_book_peaks(pages, policy)
