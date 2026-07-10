import sys
from pathlib import Path

# allow running the suite from a raw checkout, before `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from importlib.metadata import distribution

    distribution("lcb-gate")
    _installed = True   # the pytest11 entry point already registers the plugin
except Exception:
    _installed = False

if not _installed:
    # Raw checkout only: no entry point exists, so load the plugin explicitly.
    # Doing this unconditionally double-registers the module once the package
    # is installed, and pluggy crashes before collecting a single test.
    pytest_plugins = ["lcb_gate.plugin"]
