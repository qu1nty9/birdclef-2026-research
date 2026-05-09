import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "notebooks/kaggle_submission_exp_038_pantanal_onnx_fast_noalign.ipynb"
DST = ROOT / "notebooks/kaggle_submission_exp_039_pantanal_onnx_fast_no_file_scale.ipynb"


def src_of(cell):
    return "".join(cell.get("source", []))


def to_source(text):
    if not text.endswith("\n"):
        text += "\n"
    return text.splitlines(True)


nb = json.loads(SRC.read_text())
cells = nb["cells"]
code_cells = [c for c in cells if c.get("cell_type") == "code"]

# Cell 3: keep the V18 recipe and fast ONNX path, but disable only file-level
# confidence scaling. This re-tests the exp_019 proxy winner on the fast base.
cell3 = src_of(code_cells[3])
needle = 'CFG["delta_shift_alpha"] = 0.20\n'
patch = '''CFG["delta_shift_alpha"] = 0.20

# exp_039: public check of the exp_019 proxy winner on the timeout-safe exp_038 base.
FORCE_DISABLE_FILE_SCALE = True
if FORCE_DISABLE_FILE_SCALE:
    CFG["file_level_top_k"] = 0
    print("exp_039 override: file-level confidence scaling disabled")
'''
if needle not in cell3:
    raise RuntimeError("Could not find delta_shift_alpha anchor in CFG upgrade cell")
cell3 = cell3.replace(needle, patch, 1)
code_cells[3]["source"] = to_source(cell3)

# Cell 31: update diagnostics and log identity.
cell31 = src_of(code_cells[31])
cell31 = cell31.replace(
    'LOGS["experiment_id"] = "exp_038"\nLOGS["experiment_name"] = "pantanal_onnx_fast_noalign"\n',
    'LOGS["experiment_id"] = "exp_039"\nLOGS["experiment_name"] = "pantanal_onnx_fast_no_file_scale"\n',
)
cell31 = cell31.replace(
    'LOGS["onnx_tf_alignment"] = "disabled_for_submit_runtime"\nLOGS["temperature"] = CFG["temperature"]\n',
    'LOGS["onnx_tf_alignment"] = "disabled_for_submit_runtime"\nLOGS["force_disable_file_scale"] = bool(FORCE_DISABLE_FILE_SCALE)\nLOGS["file_level_top_k"] = int(CFG.get("file_level_top_k", 0))\nLOGS["temperature"] = CFG["temperature"]\n',
)
cell31 = cell31.replace(
    'with open("/kaggle/working/exp_038_pantanal_onnx_fast_noalign_logs.json", "w") as f:\n        json.dump(LOGS, f, indent=2, default=str)\n    print("Saved /kaggle/working/exp_038_pantanal_onnx_fast_noalign_logs.json")\n',
    'with open("/kaggle/working/exp_039_pantanal_onnx_fast_no_file_scale_logs.json", "w") as f:\n        json.dump(LOGS, f, indent=2, default=str)\n    print("Saved /kaggle/working/exp_039_pantanal_onnx_fast_no_file_scale_logs.json")\n',
)
cell31 = cell31.replace(
    '    print(f"ONNX->TF alignment: {LOGS[\'onnx_tf_alignment\']}")\n',
    '    print(f"ONNX->TF alignment: {LOGS[\'onnx_tf_alignment\']}")\n    print(f"File-level top_k: {LOGS[\'file_level_top_k\']}")\n',
)
code_cells[31]["source"] = to_source(cell31)

for cell in cells:
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

DST.parent.mkdir(parents=True, exist_ok=True)
DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(DST)
