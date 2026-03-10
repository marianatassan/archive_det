"""
Script de treinamento YOLOv8 para detecção de defeitos em aço laminado a quente.

Fluxo de execução:
  1. Valida a existência dos arquivos de configuração
  2. Lê hiperparâmetros de configs/hyperparameters.yaml
  3. Inicializa YOLOv8s com pesos pré-treinados no COCO (transfer learning)
  4. Executa fine-tuning com early stopping
  5. Copia best.pt e last.pt para training/models/
  6. Registra log de treinamento em training/logs/

Uso:
  python train.py            # treinamento do zero
  python train.py --resume   # retoma a partir do último checkpoint interrompido

Pré-requisitos:
  pip install ultralytics pyyaml
  Executar convert_dataset.py antes deste script.
"""

import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml
from ultralytics import YOLO

# Força UTF-8 no stdout para compatibilidade com terminais Windows (cp1252)
sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Configuração de caminhos
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).resolve().parent   # training/scripts/
TRAINING_DIR = SCRIPT_DIR.parent                 # training/

DATASET_YAML = TRAINING_DIR / "dataset.yaml"
HYPER_YAML   = TRAINING_DIR / "configs" / "hyperparameters.yaml"
RESULTS_DIR  = TRAINING_DIR / "results"
MODELS_DIR   = TRAINING_DIR / "models"
LOGS_DIR     = TRAINING_DIR / "logs"

# ---------------------------------------------------------------------------
# Argumentos de linha de comando
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treinamento YOLOv8 para detecção de defeitos em aço laminado"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retoma o treino a partir do último checkpoint salvo em results/",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Configuração de logging
# ---------------------------------------------------------------------------

def setup_logger() -> logging.Logger:
    """
    Configura um logger com saída simultânea para console e arquivo.

    O arquivo de log é criado em training/logs/ com timestamp no nome,
    permitindo rastrear múltiplas execuções de treinamento.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger("neu_det_train")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler para arquivo
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Handler para console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# Funções principais
# ---------------------------------------------------------------------------

def load_hyperparameters(hyper_path: Path) -> dict:
    """Carrega e retorna o dicionário de hiperparâmetros do arquivo YAML."""
    with open(hyper_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    return params


def validate_paths(logger: logging.Logger):
    """
    Verifica se os arquivos necessários existem antes de iniciar o treinamento.
    Encerra o processo com mensagem de erro se algo estiver ausente.
    """
    errors = []

    if not DATASET_YAML.exists():
        errors.append(f"dataset.yaml não encontrado: {DATASET_YAML}")
    if not HYPER_YAML.exists():
        errors.append(f"hyperparameters.yaml não encontrado: {HYPER_YAML}")

    if errors:
        for err in errors:
            logger.error(err)
        logger.error("Certifique-se de executar convert_dataset.py antes de train.py.")
        sys.exit(1)


def train(logger: logging.Logger) -> tuple:
    """
    Executa o fine-tuning do YOLOv8s no dataset NEU-DET.

    Decisões de design:
    - Modelo base: yolov8s.pt (pré-treinado no COCO) — transfer learning
    - imgsz=640: resolução padrão do YOLOv8, preserva a calibração do backbone
    - patience=20: early stopping evita overfitting em dataset pequeno
    - mosaic=0.0: desabilitado para manter realismo industrial (ver plano)

    Returns:
        (results, run_name) onde run_name identifica a pasta de saída.
    """
    params   = load_hyperparameters(HYPER_YAML)
    model_id = params.pop("model", "yolov8s.pt")

    logger.info("=" * 60)
    logger.info("  Pipeline de Treinamento — NEU-DET / YOLOv8")
    logger.info("=" * 60)
    logger.info(f"Modelo base      : {model_id}")
    logger.info(f"Dataset YAML     : {DATASET_YAML}")
    logger.info(f"Hiperparâmetros  : {HYPER_YAML}")
    logger.info(f"Épocas máximas   : {params.get('epochs', 100)}")
    logger.info(f"Early stopping   : patience={params.get('patience', 20)}")
    logger.info(f"Batch size       : {params.get('batch', 16)}")
    logger.info(f"Resolução entrada: {params.get('imgsz', 640)}px")
    logger.info(f"Mosaic           : {params.get('mosaic', 0.0)} (0.0 = desabilitado)")

    # Inicializa o modelo — download automático se yolov8s.pt não existir localmente
    model    = YOLO(model_id)
    run_name = f"neu_det_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger.info(f"Iniciando treinamento → pasta de saída: {run_name}")

    results = model.train(
        data    = str(DATASET_YAML),
        project = str(RESULTS_DIR),
        name    = run_name,

        # Duração e regularização
        epochs       = params.get("epochs",       100),
        patience     = params.get("patience",      20),
        batch        = params.get("batch",         16),
        imgsz        = params.get("imgsz",        640),

        # Otimizador
        optimizer    = params.get("optimizer",    "AdamW"),
        lr0          = params.get("lr0",          0.001),
        lrf          = params.get("lrf",          0.01),
        momentum     = params.get("momentum",     0.937),
        weight_decay = params.get("weight_decay", 0.0005),

        # Warm-up
        warmup_epochs   = params.get("warmup_epochs",   3.0),
        warmup_momentum = params.get("warmup_momentum", 0.8),

        # Augmentation ajustada para inspeção industrial
        mosaic      = params.get("mosaic",      0.0),   # desabilitado
        mixup       = params.get("mixup",       0.0),   # desabilitado
        copy_paste  = params.get("copy_paste",  0.0),
        hsv_h       = params.get("hsv_h",       0.015),
        hsv_s       = params.get("hsv_s",       0.7),
        hsv_v       = params.get("hsv_v",       0.4),
        degrees     = params.get("degrees",     10.0),
        translate   = params.get("translate",   0.1),
        scale       = params.get("scale",       0.3),
        shear       = params.get("shear",       0.0),
        perspective = params.get("perspective", 0.0),
        flipud      = params.get("flipud",      0.5),
        fliplr      = params.get("fliplr",      0.5),

        # Reprodutibilidade e saída
        seed          = params.get("seed",          42),
        deterministic = params.get("deterministic", True),
        save          = params.get("save",          True),
        save_period   = params.get("save_period",   -1),
        verbose       = params.get("verbose",       True),
    )

    logger.info("Treinamento finalizado.")
    return results, run_name


def find_last_checkpoint() -> Path:
    """
    Localiza o arquivo last.pt mais recente dentro de results/.

    Percorre todas as subpastas de runs e retorna o checkpoint
    com a data de modificação mais recente.
    """
    candidates = list(RESULTS_DIR.rglob("weights/last.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resume_train(logger: logging.Logger) -> tuple:
    """
    Retoma um treinamento interrompido a partir do último checkpoint.

    O Ultralytics lê todos os hiperparâmetros diretamente do arquivo
    last.pt, garantindo continuidade exata de onde parou.

    Returns:
        (results, run_name)
    """
    last_pt = find_last_checkpoint()
    if last_pt is None:
        logger.error("Nenhum checkpoint encontrado em results/. Use python train.py sem --resume.")
        sys.exit(1)

    run_name = last_pt.parent.parent.name   # weights/ -> run_dir -> name
    logger.info("=" * 60)
    logger.info("  Retomando Treinamento Interrompido — NEU-DET / YOLOv8")
    logger.info("=" * 60)
    logger.info(f"Checkpoint : {last_pt}")
    logger.info(f"Run        : {run_name}")

    model   = YOLO(str(last_pt))
    results = model.train(resume=True)

    logger.info("Treinamento retomado e finalizado.")
    return results, run_name


def copy_best_models(run_name: str, logger: logging.Logger):
    """
    Copia best.pt e last.pt da pasta de resultados para training/models/.

    - best.pt: checkpoint com maior mAP@0.5 na validação → usar para publicação
    - last.pt: último checkpoint → permite retomar treinamento interrompido
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    weights_dir = RESULTS_DIR / run_name / "weights"

    for weight_file in ("best.pt", "last.pt"):
        src = weights_dir / weight_file
        dst = MODELS_DIR / weight_file

        if src.exists():
            shutil.copy2(str(src), str(dst))
            logger.info(f"Modelo copiado: {dst}")
        else:
            logger.warning(f"Arquivo não encontrado para cópia: {src}")


# ---------------------------------------------------------------------------
# Execução principal
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    logger = setup_logger()

    if args.resume:
        results, run_name = resume_train(logger)
    else:
        validate_paths(logger)
        results, run_name = train(logger)

    copy_best_models(run_name, logger)

    logger.info("=" * 60)
    logger.info("  Pipeline concluído com sucesso!")
    logger.info(f"  Resultados : {RESULTS_DIR / run_name}")
    logger.info(f"  Modelos    : {MODELS_DIR}")
    logger.info("  Próximo passo: python evaluate.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
