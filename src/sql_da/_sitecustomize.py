"""Venv-local import bootstrap for uv editable installs.

Some Python distributions used with uv do not process editable-install `.pth`
files. This module is installed as top-level `sitecustomize.py` and only adds
this repo's `src` directory when the local package metadata points at it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname


def _add_local_src_path() -> None:
    site_packages = Path(__file__).resolve().parent
    for direct_url in site_packages.glob("sql_da-*.dist-info/direct_url.json"):
        try:
            payload = json.loads(direct_url.read_text(encoding="utf-8"))
        except OSError:
            continue
        url = str(payload.get("url", ""))
        parsed = urlparse(url)
        if parsed.scheme != "file":
            continue
        src_path = Path(url2pathname(unquote(parsed.path))) / "src"
        if (src_path / "sql_da").is_dir() and str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))


_add_local_src_path()
