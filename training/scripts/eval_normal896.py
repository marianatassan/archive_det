"""Avaliação normal (sem TTA) do v8 com imgsz=896 (resolução de treino)."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
from ultralytics import YOLO

TRAINING_DIR = Path(__file__).resolve().parent.parent
DATASET_YAML = TRAINING_DIR / "dataset.yaml"
RESULTS_DIR  = TRAINING_DIR / "results"

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

MODEL_PATH = RESULTS_DIR / "neu_det_20260317_011757" / "weights" / "best.pt"

print(f"{'='*60}")
print(f"  Normal@896 — v8  |  {MODEL_PATH.parent.parent.name}")
print(f"{'='*60}")

model = YOLO(str(MODEL_PATH))
m = model.val(
    data    = str(DATASET_YAML),
    split   = "test",
    imgsz   = 896,
    batch   = 4,
    workers = 0,
    augment = False,
    project = str(RESULTS_DIR),
    name    = "normal896_v8",
    verbose = False,
    plots   = False,
)

box = m.box
f1  = 2 * box.mp * box.mr / (box.mp + box.mr + 1e-9)

print(f"\n{'='*65}")
print(f"  RESULTADO Normal@896 — v8 — CONJUNTO DE TESTE")
print(f"{'='*65}")
print(f"\n  mAP@0.5      : {box.map50:.4f}")
print(f"  mAP@0.5:0.95 : {box.map:.4f}")
print(f"  Precision    : {box.mp:.4f}")
print(f"  Recall       : {box.mr:.4f}")
print(f"  F1           : {f1:.4f}")
print(f"\n  {'Classe':<22} {'mAP@0.5':>9} {'P':>8} {'R':>8}")
print(f"  {'-'*50}")
for i, cls in enumerate(CLASSES):
    print(f"  {cls:<22} {float(box.ap50[i]):>9.4f} {float(box.p[i]):>8.4f} {float(box.r[i]):>8.4f}")
