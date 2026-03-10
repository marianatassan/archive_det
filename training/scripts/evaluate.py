"""
Script de avaliação do modelo YOLOv8 no conjunto de TESTE.

O conjunto de teste nunca participou de nenhuma decisão de treinamento
(escolha de hiperparâmetros, early stopping, seleção de checkpoint),
garantindo uma estimativa de desempenho honesta e não enviesada.

Métricas reportadas:
  Global  : mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1-Score
  Por classe: mAP@0.5, mAP@0.5:0.95, Precision, Recall

Saídas geradas:
  - Relatório no console
  - metrics_test.json  → dados numéricos para documentação do TCC
  - confusion_matrix.png, PR_curve.png, F1_curve.png (gerados pelo Ultralytics)

Uso:
  python evaluate.py

Pré-requisito:
  Executar train.py antes deste script.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

# Força UTF-8 no stdout para compatibilidade com terminais Windows (cp1252)
sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Configuração de caminhos
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).resolve().parent   # training/scripts/
TRAINING_DIR = SCRIPT_DIR.parent                 # training/

DATASET_YAML = TRAINING_DIR / "dataset.yaml"
MODELS_DIR   = TRAINING_DIR / "models"
RESULTS_DIR  = TRAINING_DIR / "results"
BEST_MODEL   = MODELS_DIR / "best.pt"

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

# ---------------------------------------------------------------------------
# Funções principais
# ---------------------------------------------------------------------------

def validate_paths():
    """Verifica a existência dos arquivos necessários para avaliação."""
    errors = []

    if not BEST_MODEL.exists():
        errors.append(f"Modelo não encontrado: {BEST_MODEL}")
    if not DATASET_YAML.exists():
        errors.append(f"dataset.yaml não encontrado: {DATASET_YAML}")

    if errors:
        for err in errors:
            print(f"[ERRO] {err}")
        print("Execute train.py antes de rodar evaluate.py.")
        sys.exit(1)


def run_evaluation() -> tuple:
    """
    Executa a avaliação do modelo best.pt no split de teste.

    O parâmetro split="test" instrui o Ultralytics a usar o caminho
    definido na chave 'test' do dataset.yaml, garantindo que o conjunto
    de teste seja avaliado de forma isolada.

    Returns:
        (metrics, run_name)
    """
    print(f"\n[INFO] Modelo    : {BEST_MODEL}")
    print(f"[INFO] Dataset   : {DATASET_YAML}")
    print(f"[INFO] Split     : test\n")

    model    = YOLO(str(BEST_MODEL))
    run_name = f"eval_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    metrics = model.val(
        data    = str(DATASET_YAML),
        split   = "test",       # avalia exclusivamente no conjunto de teste
        imgsz   = 640,
        batch   = 16,
        project = str(RESULTS_DIR),
        name    = run_name,
        verbose = True,
        plots   = True,         # gera confusion_matrix, PR_curve, F1_curve
    )

    return metrics, run_name


def print_report(metrics, run_name: str):
    """
    Exibe o relatório completo de métricas no console.

    Métricas globais e por classe são apresentadas em formato tabular
    para facilitar a transcrição ao TCC.
    """
    box = metrics.box

    # Cálculo do F1 a partir de Precision e Recall médios
    f1_mean = 2 * box.mp * box.mr / (box.mp + box.mr + 1e-9)

    print("\n" + "=" * 65)
    print("  RELATÓRIO DE AVALIAÇÃO — CONJUNTO DE TESTE (NEU-DET)")
    print("=" * 65)

    print(f"\n{'Métrica Global':<35} {'Valor':>10}")
    print("-" * 47)
    print(f"{'mAP@0.5':<35} {box.map50:>10.4f}")
    print(f"{'mAP@0.5:0.95':<35} {box.map:>10.4f}")
    print(f"{'Precision (média)':<35} {box.mp:>10.4f}")
    print(f"{'Recall (média)':<35} {box.mr:>10.4f}")
    print(f"{'F1-Score (média)':<35} {f1_mean:>10.4f}")

    print(f"\n{'Classe':<22} {'Precision':>10} {'Recall':>8} {'mAP@0.5':>9} {'mAP@0.5:0.95':>14}")
    print("-" * 65)

    for i, cls in enumerate(CLASSES):
        try:
            p    = float(box.p[i])
            r    = float(box.r[i])
            ap50 = float(box.ap50[i])
            ap   = float(box.ap[i])
            print(f"{cls:<22} {p:>10.4f} {r:>8.4f} {ap50:>9.4f} {ap:>14.4f}")
        except (IndexError, TypeError):
            print(f"{cls:<22} {'N/D':>10} {'N/D':>8} {'N/D':>9} {'N/D':>14}")

    print("\n" + "=" * 65)
    print(f"  Resultados salvos em : {RESULTS_DIR / run_name}")
    print("=" * 65)


def save_metrics_json(metrics, run_name: str):
    """
    Serializa todas as métricas em metrics_test.json.

    O arquivo JSON é adequado para importação em tabelas do TCC e
    para reprodução dos resultados por outros pesquisadores.
    """
    output_dir = RESULTS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    box     = metrics.box
    f1_mean = 2 * box.mp * box.mr / (box.mp + box.mr + 1e-9)

    report = {
        "meta": {
            "split":     "test",
            "model":     str(BEST_MODEL),
            "timestamp": datetime.now().isoformat(),
            "dataset":   str(DATASET_YAML),
        },
        "global_metrics": {
            "mAP_50":    round(float(box.map50), 6),
            "mAP_50_95": round(float(box.map),   6),
            "precision": round(float(box.mp),     6),
            "recall":    round(float(box.mr),     6),
            "f1":        round(float(f1_mean),    6),
        },
        "per_class_metrics": {},
    }

    for i, cls in enumerate(CLASSES):
        try:
            report["per_class_metrics"][cls] = {
                "precision": round(float(box.p[i]),    6),
                "recall":    round(float(box.r[i]),    6),
                "mAP_50":    round(float(box.ap50[i]), 6),
                "mAP_50_95": round(float(box.ap[i]),   6),
            }
        except (IndexError, TypeError):
            report["per_class_metrics"][cls] = "indisponível"

    json_path = output_dir / "metrics_test.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Métricas salvas em: {json_path}")


# ---------------------------------------------------------------------------
# Execução principal
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  Avaliação no Conjunto de Teste — NEU-DET / YOLOv8")
    print("=" * 65)

    validate_paths()

    metrics, run_name = run_evaluation()

    print_report(metrics, run_name)

    save_metrics_json(metrics, run_name)

    print("\n  Próximo passo: python inference.py")


if __name__ == "__main__":
    main()
