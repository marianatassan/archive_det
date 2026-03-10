"""
Script de conversão e preparação do dataset NEU-DET para YOLOv8.

Operações realizadas:
  1. Varredura de todas as imagens e anotações (train + validation originais)
  2. Verificação automática do número de canais das imagens
  3. Conversão de anotações Pascal VOC (XML) → formato YOLO (TXT normalizado)
  4. Re-divisão estratificada por classe: 70% treino / 15% validação / 15% teste
  5. Criação da estrutura de diretórios exigida pelo YOLOv8
  6. Geração automática do arquivo dataset.yaml com caminhos absolutos

Formato YOLO (por linha, por objeto):
  class_id  x_center  y_center  width  height   (todos normalizados em [0, 1])

Uso:
  python convert_dataset.py

Pré-requisito:
  pip install opencv-python
"""

import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

# Força UTF-8 no stdout para compatibilidade com terminais Windows (cp1252)
sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers para leitura/escrita de imagens com caminhos Unicode no Windows
# ---------------------------------------------------------------------------
# cv2.imread / cv2.imwrite usam a API C de arquivo, que não suporta caracteres
# não-ASCII em caminhos no Windows. A solução é ler bytes com numpy (que usa a
# API Unicode do Windows) e decodificar/codificar na memória com cv2.imdecode
# / cv2.imencode, os quais operam sobre buffers e não sobre caminhos de arquivo.

def imread_unicode(path: Path):
    """Lê uma imagem de um caminho que pode conter caracteres Unicode."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img) -> bool:
    """Salva uma imagem em um caminho que pode conter caracteres Unicode."""
    success, buf = cv2.imencode(path.suffix.lower(), img)
    if success:
        buf.tofile(str(path))
    return success

# ---------------------------------------------------------------------------
# Configuração de caminhos
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).resolve().parent   # training/scripts/
TRAINING_DIR = SCRIPT_DIR.parent                 # training/
PROJECT_DIR  = TRAINING_DIR.parent               # archive_det/
NEU_DET_DIR  = PROJECT_DIR / "NEU-DET"           # dataset original

DATASET_OUT  = TRAINING_DIR / "dataset"          # destino convertido
YAML_OUT     = TRAINING_DIR / "dataset.yaml"     # arquivo de config do dataset

# ---------------------------------------------------------------------------
# Configuração do split e classes
# ---------------------------------------------------------------------------

SEED        = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 0.15  (implícito: o restante após train e val)

# Ordem alfabética → define os IDs de classe no formato YOLO
CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}

# Splits de origem no NEU-DET (train + validation são tratados como pool único)
SOURCE_SPLITS = [
    NEU_DET_DIR / "train",
    NEU_DET_DIR / "validation",
]

# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def verify_image_channels(sample_path: Path) -> int:
    """
    Lê uma imagem de amostra e retorna o número de canais detectado.

    Returns:
        1 — escala de cinza pura
        3 — BGR (já em 3 canais)
    """
    img = imread_unicode(sample_path)
    if img is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem: {sample_path}")
    return 1 if img.ndim == 2 else img.shape[2]


def xml_to_yolo(xml_path: Path, img_width: int, img_height: int) -> list:
    """
    Converte um arquivo de anotação Pascal VOC (XML) para linhas no formato YOLO.

    Fórmulas de normalização:
      x_center = (xmin + xmax) / (2 * W)
      y_center = (ymin + ymax) / (2 * H)
      width    = (xmax - xmin) / W
      height   = (ymax - ymin) / H

    Args:
        xml_path   : Caminho para o arquivo .xml
        img_width  : Largura real da imagem em pixels
        img_height : Altura real da imagem em pixels

    Returns:
        Lista de strings, uma por objeto anotado no formato YOLO.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []

    for obj in root.findall("object"):
        class_name = obj.find("name").text.strip()

        if class_name not in CLASS_TO_ID:
            print(f"  [AVISO] Classe desconhecida ignorada: '{class_name}' em {xml_path.name}")
            continue

        class_id = CLASS_TO_ID[class_name]
        bndbox   = obj.find("bndbox")

        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        # Conversão para coordenadas normalizadas
        cx = (xmin + xmax) / (2.0 * img_width)
        cy = (ymin + ymax) / (2.0 * img_height)
        w  = (xmax - xmin) / img_width
        h  = (ymax - ymin) / img_height

        # Clamp para garantir que nenhum valor saia de [0, 1]
        cx, cy, w, h = (
            max(0.0, min(1.0, cx)),
            max(0.0, min(1.0, cy)),
            max(0.0, min(1.0, w)),
            max(0.0, min(1.0, h)),
        )

        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines


def collect_samples_by_class() -> dict:
    """
    Varre as pastas de origem e agrupa amostras por classe.

    Returns:
        { class_name: [(img_path, xml_path), ...] }
    """
    samples = {cls: [] for cls in CLASSES}

    for split_dir in SOURCE_SPLITS:
        images_dir      = split_dir / "images"
        annotations_dir = split_dir / "annotations"

        for class_name in CLASSES:
            class_img_dir = images_dir / class_name
            if not class_img_dir.exists():
                print(f"  [AVISO] Diretório não encontrado: {class_img_dir}")
                continue

            for img_file in sorted(class_img_dir.glob("*.jpg")):
                xml_file = annotations_dir / (img_file.stem + ".xml")
                if not xml_file.exists():
                    print(f"  [AVISO] Anotação ausente: {img_file.name}")
                    continue
                samples[class_name].append((img_file, xml_file))

    return samples


def stratified_split(samples: dict) -> tuple:
    """
    Divide o dataset de forma estratificada por classe (70 / 15 / 15).

    A estratificação garante que cada split mantenha a proporção de
    amostras por classe, preservando o balanceamento do NEU-DET.

    Returns:
        (train_samples, val_samples, test_samples)
        Cada amostra é uma tupla (img_path, xml_path, class_name).
    """
    random.seed(SEED)
    train_all, val_all, test_all = [], [], []

    print(f"\n  {'Classe':<22} {'Treino':>8} {'Val':>6} {'Teste':>7}")
    print("  " + "-" * 46)

    for class_name, class_samples in samples.items():
        shuffled = class_samples.copy()
        random.shuffle(shuffled)

        n       = len(shuffled)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)
        # n_test  = restante

        train_cls = shuffled[:n_train]
        val_cls   = shuffled[n_train : n_train + n_val]
        test_cls  = shuffled[n_train + n_val :]

        train_all.extend([(p, x, class_name) for p, x in train_cls])
        val_all.extend(  [(p, x, class_name) for p, x in val_cls])
        test_all.extend( [(p, x, class_name) for p, x in test_cls])

        print(
            f"  {class_name:<22} {len(train_cls):>8} {len(val_cls):>6} {len(test_cls):>7}"
        )

    return train_all, val_all, test_all


def create_directory_structure():
    """Cria os subdiretórios de imagens e labels para cada split."""
    for split in ("train", "val", "test"):
        (DATASET_OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_OUT / "labels" / split).mkdir(parents=True, exist_ok=True)


def process_samples(samples: list, split_name: str, needs_channel_conversion: bool):
    """
    Processa um conjunto de amostras:
      - Copia a imagem (convertendo para BGR se necessário)
      - Converte a anotação XML → TXT YOLO

    Args:
        samples                   : Lista de (img_path, xml_path, class_name)
        split_name                : "train", "val" ou "test"
        needs_channel_conversion  : True se as imagens precisam ser convertidas
                                    de 1 canal para 3 canais
    """
    img_out_dir = DATASET_OUT / "images" / split_name
    lbl_out_dir = DATASET_OUT / "labels" / split_name

    for img_path, xml_path, _ in samples:
        dest_img = img_out_dir / img_path.name

        # --- Copia / converte a imagem ---
        if needs_channel_conversion:
            img = imread_unicode(img_path)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            imwrite_unicode(dest_img, img)
        else:
            shutil.copy2(str(img_path), str(dest_img))

        # --- Obtém dimensões reais da imagem salva ---
        img_cv = imread_unicode(dest_img)
        h, w   = img_cv.shape[:2]

        # --- Converte e salva a anotação YOLO ---
        yolo_lines = xml_to_yolo(xml_path, img_width=w, img_height=h)
        dest_lbl   = lbl_out_dir / (img_path.stem + ".txt")
        dest_lbl.write_text("\n".join(yolo_lines), encoding="utf-8")


def generate_dataset_yaml():
    """
    Gera o arquivo dataset.yaml com caminhos absolutos.

    O Ultralytics YOLOv8 usa esse arquivo como ponto central de configuração
    do dataset durante treino, validação e avaliação.
    """
    dataset_path = DATASET_OUT.resolve()

    # Formata a lista de classes no padrão YAML de sequência de fluxo
    names_str = "[" + ", ".join(CLASSES) + "]"

    yaml_content = (
        "# Dataset: NEU-DET — Detecção de Defeitos em Aço Laminado a Quente\n"
        "# Re-split estratificado por classe: 70% treino / 15% val / 15% teste\n"
        "# Gerado automaticamente por convert_dataset.py\n"
        "#\n"
        "# Classes (ID → nome):\n"
    )
    for idx, cls in enumerate(CLASSES):
        yaml_content += f"#   {idx}: {cls}\n"

    yaml_content += (
        "\n"
        f"path: {dataset_path.as_posix()}\n"
        "train: images/train\n"
        "val:   images/val\n"
        "test:  images/test\n"
        "\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {names_str}\n"
    )

    YAML_OUT.write_text(yaml_content, encoding="utf-8")
    print(f"\n  [OK] dataset.yaml → {YAML_OUT}")


# ---------------------------------------------------------------------------
# Execução principal
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Conversão NEU-DET → YOLOv8")
    print("  Split: 70% treino / 15% val / 15% teste (estratificado)")
    print("=" * 60)

    # Verifica existência do dataset original
    if not NEU_DET_DIR.exists():
        print(f"[ERRO] Dataset não encontrado: {NEU_DET_DIR}")
        return

    # --- Passo 1: Verificação de canais ---
    print("\n[PASSO 1] Verificando canais das imagens...")
    sample_img = next((NEU_DET_DIR / "train" / "images" / CLASSES[0]).glob("*.jpg"))
    n_channels = verify_image_channels(sample_img)
    needs_conversion = n_channels < 3

    print(f"  Canais detectados  : {n_channels}")
    if needs_conversion:
        print("  Ação               : conversão GRAY → BGR (3 canais)")
    else:
        print("  Ação               : sem conversão (já em 3 canais)")

    # --- Passo 2: Coleta de amostras ---
    print("\n[PASSO 2] Coletando amostras por classe...")
    samples_by_class = collect_samples_by_class()
    total = sum(len(v) for v in samples_by_class.values())
    print(f"  Total encontrado   : {total} imagens")

    # --- Passo 3: Split estratificado ---
    print("\n[PASSO 3] Dividindo dataset (70 / 15 / 15)...")
    train_samples, val_samples, test_samples = stratified_split(samples_by_class)

    print(
        f"\n  Total — Treino: {len(train_samples)} | "
        f"Val: {len(val_samples)} | Teste: {len(test_samples)}"
    )

    # --- Passo 4: Criação de diretórios ---
    print("\n[PASSO 4] Criando estrutura de diretórios...")
    create_directory_structure()
    print(f"  Diretório de saída : {DATASET_OUT}")

    # --- Passo 5: Processamento das amostras ---
    print("\n[PASSO 5] Copiando imagens e convertendo anotações...")
    for split_name, split_samples in [
        ("train", train_samples),
        ("val",   val_samples),
        ("test",  test_samples),
    ]:
        print(f"  Processando '{split_name}' ({len(split_samples)} amostras)...")
        process_samples(split_samples, split_name, needs_conversion)

    # --- Passo 6: Geração do YAML ---
    print("\n[PASSO 6] Gerando dataset.yaml...")
    generate_dataset_yaml()

    print("\n" + "=" * 60)
    print("  Conversão concluída com sucesso!")
    print("  Próximo passo: python train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
