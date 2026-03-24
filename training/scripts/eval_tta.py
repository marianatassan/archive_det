"""Avaliação com TTA (Test Time Augmentation) para modelos selecionados."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
from ultralytics import YOLO

TRAINING_DIR = Path(__file__).resolve().parent.parent
DATASET_YAML = TRAINING_DIR / "dataset.yaml"
RESULTS_DIR  = TRAINING_DIR / "results"

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

RUNS = [
    ("v8",  RESULTS_DIR / "neu_det_20260317_011757" / "weights" / "best.pt"),
]

all_results = {}

for label, model_path in RUNS:
    print(f"\n{'='*60}")
    print(f"  TTA — {label}  |  {model_path.parent.parent.name}")
    print(f"{'='*60}")
    model = YOLO(str(model_path))
    m = model.val(
        data    = str(DATASET_YAML),
        split   = "test",
        imgsz   = 896,      # resolução de treino do v8
        batch   = 4,        # reduzido para caber na GTX 1050 com imgsz=896
        workers = 0,        # necessário no Windows
        augment = True,     # TTA
        project = str(RESULTS_DIR),
        name    = f"tta_{label}",
        verbose = False,
        plots   = False,
    )
    all_results[label] = m

print("\n\n" + "=" * 65)
print("  RESULTADO TTA — CONJUNTO DE TESTE (NEU-DET)")
print("=" * 65)

for label, m in all_results.items():
    box = m.box
    f1  = 2 * box.mp * box.mr / (box.mp + box.mr + 1e-9)
    print(f"\n--- {label} (com TTA) ---")
    print(f"  mAP@0.5      : {box.map50:.4f}")
    print(f"  mAP@0.5:0.95 : {box.map:.4f}")
    print(f"  Precision    : {box.mp:.4f}")
    print(f"  Recall       : {box.mr:.4f}")
    print(f"  F1           : {f1:.4f}")
    print(f"\n  {'Classe':<22} {'mAP@0.5':>9} {'P':>8} {'R':>8}")
    print(f"  {'-'*50}")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:<22} {float(box.ap50[i]):>9.4f} {float(box.p[i]):>8.4f} {float(box.r[i]):>8.4f}")
