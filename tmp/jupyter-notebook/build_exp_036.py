import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "references/private-notebooks/pantanal-distill-birdclef2026-onnx-0.93.ipynb"
DST = ROOT / "notebooks/kaggle_submission_exp_036_pantanal_tf_exact_replay.ipynb"


nb = json.loads(SRC.read_text())

# Keep the source code intentionally unchanged. The goal of exp_036 is to test
# whether the reference Pantanal TF recipe itself reproduces the reported 0.930.
for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

DST.parent.mkdir(parents=True, exist_ok=True)
DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(DST)
