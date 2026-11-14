"""Load the unchanged nmanga book functions without importing its CLI stack."""

from __future__ import annotations

import ast
import importlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

from PIL import Image, ImageFilter


class BookReference:
    """The reference's file analyzer, consensus and page resolver.

    Only the selected definitions are compiled from the CLI source. Their bodies
    are unchanged; bindings come from the real autolevel module and Pillow.
    Missing or newly required definitions fail rather than selecting a local copy.
    """

    def __init__(self, path: Path) -> None:
        path = path.resolve()
        sys.path.insert(0, str(path))
        self.autolevel = importlib.import_module("nmanga.autolevel")
        if Path(self.autolevel.__file__).resolve() != path / "nmanga" / "autolevel.py":
            raise RuntimeError("nmanga was already imported from a different checkout")
        source_path = path / "nmanga" / "cli" / "autolevel.py"
        names = {
            "_BookPeakPage",
            "_ResolvedPageLevels",
            "_analyze_book_peak_page",
            "_book_peak_consensus",
            "_resolve_book_peak_page",
            "_resolve_book_peak_pages",
        }
        source = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        definitions = [node for node in source.body if getattr(node, "name", None) in names]
        if len(definitions) != len(names):
            raise RuntimeError("nmanga book policy moved; update the reference loader")
        namespace: dict[str, Any] = {
            "__name__": "nmanga_book_reference",
            "NamedTuple": NamedTuple,
            "Sequence": Sequence,
            "Path": Path,
            "Image": Image,
            "ImageFilter": ImageFilter,
            "_LocalPeakAnalysis": self.autolevel._LocalPeakAnalysis,
            "_find_local_peak_analysis": self.autolevel._find_local_peak_analysis,
            "book_peak": self.autolevel.book_peak,
        }
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source_path), "exec"), namespace)
        self.analyze_page = namespace["_analyze_book_peak_page"]
        self.resolve_pages = namespace["_resolve_book_peak_pages"]
