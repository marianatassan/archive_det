"""
Script de inferência para geração de imagens com bounding boxes preditas.

Executa o modelo treinado (best.pt) em um conjunto de imagens e salva
as visualizações anotadas em training/predictions/.

Finalidades no contexto do TCC:
  - Análise qualitativa das detecções (acertos, erros e falsos positivos)
  - Identificação de padrões de confusão entre classes similares
  - Geração de figuras ilustrativas para o trabalho acadêmico

Uso padrão (conjunto de teste):
  python inference.py

Uso com diretório/arquivo customizado:
  python inference.py --source caminho/para/imagens
  python inference.py --source imagem.jpg --conf 0.4

Argumentos:
  --source  : Diretório ou arquivo de imagem (padrão: split de teste)
  --conf    : Limiar de confiança mínima (padrão: 0.25)
  --iou     : Limiar de IoU para NMS (padrão: 0.45)

Pré-requisito:
  Executar train.py antes deste script.
"""

import argparse
import sys
import yaml
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Força UTF-8 no stdout para compatibilidade com terminais Windows (cp1252)
sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers para leitura/escrita com caminhos Unicode no Windows
# ---------------------------------------------------------------------------

def imread_unicode(path: Path):
    """Lê imagem de caminho com caracteres Unicode (workaround para Windows)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img) -> bool:
    """Salva imagem em caminho com caracteres Unicode (workaround para Windows)."""
    success, buf = cv2.imencode(path.suffix.lower(), img)
    if success:
        buf.tofile(str(path))
    return success

# ---------------------------------------------------------------------------
# Configuração de caminhos
# ---------------------------------------------------------------------------

SCRIPT_DIR      = Path(__file__).resolve().parent   # training/scripts/
TRAINING_DIR    = SCRIPT_DIR.parent                 # training/

DATASET_YAML    = TRAINING_DIR / "dataset.yaml"
MODELS_DIR      = TRAINING_DIR / "models"
PREDICTIONS_DIR = TRAINING_DIR / "predictions"
BEST_MODEL      = MODELS_DIR / "best.pt"

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

# Paleta de cores por classe (BGR) — escolhidas para contraste visual em cinza
CLASS_COLORS = {
    "crazing":         (0,   255,   0),    # verde
    "inclusion":       (0,   165, 255),    # laranja
    "patches":         (255,   0,   0),    # azul
    "pitted_surface":  (0,     0, 255),    # vermelho
    "rolled-in_scale": (255,   0, 255),    # magenta
    "scratches":       (0,   255, 255),    # ciano
}

# Extensões de imagem aceitas
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inferência YOLOv8 — Defeitos em Aço Laminado a Quente"
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Diretório ou arquivo de imagem (padrão: split de teste do dataset)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Limiar de confiança mínima para aceitar detecção (padrão: 0.25)",
    )
    parser.add_argument(
        "--iou", type=float, default=0.45,
        help="Limiar de IoU para Non-Maximum Suppression (padrão: 0.45)",
    )
    return parser.parse_args()


def validate_model():
    """Verifica se o modelo treinado existe."""
    if not BEST_MODEL.exists():
        print(f"[ERRO] Modelo não encontrado: {BEST_MODEL}")
        print("Execute train.py antes de rodar inference.py.")
        sys.exit(1)


def resolve_source(source_arg) -> Path:
    """
    Determina o diretório/arquivo de entrada para inferência.

    Se não informado via argumento, lê o caminho do split de teste
    diretamente do dataset.yaml para garantir consistência com a avaliação.
    """
    if source_arg is not None:
        source = Path(source_arg)
        if not source.exists():
            print(f"[ERRO] Caminho não encontrado: {source}")
            sys.exit(1)
        return source

    # Lê o split de teste a partir do dataset.yaml
    if not DATASET_YAML.exists():
        print(f"[ERRO] dataset.yaml não encontrado: {DATASET_YAML}")
        sys.exit(1)

    with open(DATASET_YAML, "r", encoding="utf-8") as f:
        ds_cfg = yaml.safe_load(f)

    dataset_root = Path(ds_cfg["path"])
    test_images  = dataset_root / ds_cfg.get("test", "images/test")

    if not test_images.exists():
        print(f"[ERRO] Diretório de teste não encontrado: {test_images}")
        sys.exit(1)

    return test_images


def collect_images(source: Path) -> list:
    """
    Coleta todos os arquivos de imagem em um diretório (recursivo)
    ou retorna uma lista unitária se a fonte for um arquivo único.
    """
    if source.is_file():
        return [source]

    images = [p for p in source.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS]

    if not images:
        print(f"[ERRO] Nenhuma imagem encontrada em: {source}")
        sys.exit(1)

    return sorted(images)


def draw_detections(img_bgr, boxes) -> tuple:
    """
    Desenha bounding boxes e labels sobre a imagem.

    Design:
    - Caixa colorida por classe para diferenciação visual rápida
    - Label com nome da classe e score de confiança
    - Fundo sólido no label para legibilidade sobre qualquer fundo

    Args:
        img_bgr : Imagem OpenCV (BGR)
        boxes   : Objeto boxes retornado pelo Ultralytics

    Returns:
        (img_anotada, n_deteccoes)
    """
    if boxes is None or len(boxes) == 0:
        return img_bgr, 0

    n_det = 0
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id     = int(box.cls[0])
        cls_name   = CLASSES[cls_id] if cls_id < len(CLASSES) else str(cls_id)
        confidence = float(box.conf[0])
        color      = CLASS_COLORS.get(cls_name, (255, 255, 255))
        label      = f"{cls_name} {confidence:.2f}"

        # Bounding box
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness=2)

        # Dimensiona o label ao tamanho da caixa de texto
        (lw, lh), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1
        )

        # Fundo do label (retângulo sólido acima da bbox)
        label_y = max(y1, lh + baseline + 4)
        cv2.rectangle(
            img_bgr,
            (x1, label_y - lh - baseline - 4),
            (x1 + lw, label_y),
            color,
            thickness=-1,
        )

        # Texto do label
        cv2.putText(
            img_bgr,
            label,
            (x1, label_y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        n_det += 1

    return img_bgr, n_det


# ---------------------------------------------------------------------------
# Função principal de inferência
# ---------------------------------------------------------------------------

def run_inference(source: Path, conf: float, iou: float) -> tuple:
    """
    Executa a inferência em todas as imagens da fonte e salva os resultados.

    Returns:
        (out_dir, total, com_deteccao, sem_deteccao)
    """
    model    = YOLO(str(BEST_MODEL))
    run_name = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir  = PREDICTIONS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    images        = collect_images(source)
    total         = len(images)
    com_deteccao  = 0
    sem_deteccao  = 0

    print(f"\n[INFO] Total de imagens : {total}")
    print(f"[INFO] Confiança mínima : {conf}")
    print(f"[INFO] IoU (NMS)        : {iou}")
    print(f"[INFO] Saída            : {out_dir}\n")

    for idx, img_path in enumerate(images, start=1):
        # Lê a imagem via numpy para suportar caminhos Unicode no Windows,
        # depois passa o array diretamente ao modelo (evita o mesmo problema
        # que ocorre quando model.predict recebe um caminho de arquivo)
        img_bgr = imread_unicode(img_path)

        results = model.predict(
            source  = img_bgr,
            conf    = conf,
            iou     = iou,
            verbose = False,
        )

        result      = results[0]
        img_bgr_out = result.orig_img.copy()

        # Anota bounding boxes manualmente para cores por classe
        img_anotada, n_det = draw_detections(img_bgr_out, result.boxes)

        if n_det > 0:
            com_deteccao += 1
        else:
            sem_deteccao += 1

        # Salva imagem anotada com suporte a caminho Unicode
        out_path = out_dir / img_path.name
        imwrite_unicode(out_path, img_anotada)

        # Progresso a cada 50 imagens
        if idx % 50 == 0 or idx == total:
            print(f"  Processadas: {idx}/{total}")

    return out_dir, total, com_deteccao, sem_deteccao


# ---------------------------------------------------------------------------
# Execução principal
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Inferência YOLOv8 — Defeitos em Aço Laminado a Quente")
    print("=" * 60)

    args = parse_args()

    validate_model()

    source = resolve_source(args.source)

    out_dir, total, com_det, sem_det = run_inference(
        source = source,
        conf   = args.conf,
        iou    = args.iou,
    )

    print("\n" + "=" * 60)
    print("  Inferência concluída!")
    print(f"  Total de imagens   : {total}")
    print(f"  Com detecções      : {com_det}")
    print(f"  Sem detecções      : {sem_det}")
    print(f"  Imagens salvas em  : {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
