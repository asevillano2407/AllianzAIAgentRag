"""Console launcher for the optional Streamlit demonstration."""

import importlib.util
import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Launch Streamlit with the packaged demo application."""
    if importlib.util.find_spec("streamlit") is None:
        print(
            'Streamlit is not installed. Run: python -m pip install -e ".[demo]"',
            file=sys.stderr,
        )
        return 1
    app_path = Path(__file__).with_name("app.py")
    completed = subprocess.run(  # noqa: S603 - fixed interpreter and local app path
        [sys.executable, "-m", "streamlit", "run", str(app_path)],
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
