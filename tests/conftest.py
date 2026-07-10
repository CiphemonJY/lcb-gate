import sys
from pathlib import Path

# allow running the suite from a raw checkout, before `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest_plugins = ["lcb_gate.plugin"]
