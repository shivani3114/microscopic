import json
import random
import numpy as np
import joblib
from pathlib import Path
from PIL import Image, ImageDraw

random.seed(42)

DATASET_DIR = Path("dataset/test")
OUTPUT_DIR = Path("image_based_output")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_DIR = Path("outputs")
SK_MODEL_PATH = MODEL_DIR / "microscopic_classifier.joblib"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
IMG_SIZE = (64, 64)
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

NAMING_CRITERIA = {
    "Amoeba":            "Irregular, blob-like shape with pseudopods; no fixed form",
    "Euglena":           "Elongated, spindle-shaped with a visible flagellum",
    "Hydra":             "Tubular body with tentacles extending from one end",
    "Paramecium":        "Oval/slipper-shaped, covered with cilia all around",
    "Rod_bacteria":      "Cylindrical/rod-shaped cells (Bacilli); straight edges",
    "Spherical_bacteria":"Round/spherical cells (Cocci); circular uniform shape",
    "Spiral_bacteria":   "Twisted/helical/spiral-shaped cells (Spirilla)",
    "Yeast":             "Oval budding cells; single or clustered round shapes",
}

def load_and_resize(path, size=(160, 160)):
    return Image.open(path).convert("RGB").resize(size)

def get_sample_images(class_name, n=3):
    class_dir = DATASET_DIR / class_name
    if not class_dir.exists():
        return []
    images = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
    random.shuffle(images)
    return images[:n]

def predict(image_path, model, encoder):
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    features = (np.asarray(img, dtype=np.float32) / 255.0).flatten().reshape(1, -1)
    pred_idx = model.predict(features)[0]
    probs = model.predict_proba(features)[0]
    predicted = encoder.inverse_transform([pred_idx])[0]
    return predicted, float(np.max(probs)), probs

def draw_text_wrapped(draw, text, x, y, max_width, line_height=16, fill=(60, 60, 60)):
    words = text.split()
    line = ""
    for word in words:
        test = line + word + " "
        if len(test) * 7 > max_width and line:
            draw.text((x, y), line.strip(), fill=fill)
            y += line_height
            line = word + " "
        else:
            line = test
    if line:
        draw.text((x, y), line.strip(), fill=fill)
    return y + line_height

payload = joblib.load(SK_MODEL_PATH)
model = payload["model"]
encoder = payload["label_encoder"]
with open(CLASS_NAMES_PATH) as f:
    class_names = json.load(f)

# Individual class cards
print("[INFO] Generating individual class output images...")
THUMB, PADDING, HEADER, CRITERIA_H, SAMPLE_LABEL_H, N_SAMPLES, PRED_H = 160, 20, 50, 60, 24, 3, 50
card_w = PADDING * 2 + THUMB * N_SAMPLES + PADDING * (N_SAMPLES - 1)
card_h = HEADER + CRITERIA_H + SAMPLE_LABEL_H + THUMB + PRED_H + PADDING * 2

for class_name in class_names:
    samples = get_sample_images(class_name, N_SAMPLES)
    if not samples:
        print(f"  [SKIP] {class_name}")
        continue

    card = Image.new("RGB", (card_w, card_h), (245, 248, 252))
    draw = ImageDraw.Draw(card)

    draw.rectangle((0, 0, card_w, HEADER), fill=(40, 80, 140))
    draw.text((PADDING, 14), f"Class: {class_name}", fill="white")

    draw.rectangle((0, HEADER, card_w, HEADER + CRITERIA_H), fill=(220, 230, 245))
    draw.text((PADDING, HEADER + 6), "Naming Criteria:", fill=(30, 60, 120))
    draw_text_wrapped(draw, NAMING_CRITERIA.get(class_name, ""), PADDING, HEADER + 22, card_w - PADDING * 2, fill=(50, 50, 80))

    y_label = HEADER + CRITERIA_H
    draw.rectangle((0, y_label, card_w, y_label + SAMPLE_LABEL_H), fill=(200, 215, 240))
    draw.text((PADDING, y_label + 5), f"Sample Images from Test Set ({len(samples)} shown)", fill=(30, 60, 120))

    y_img = y_label + SAMPLE_LABEL_H
    for i, img_path in enumerate(samples):
        x = PADDING + i * (THUMB + PADDING)
        card.paste(load_and_resize(img_path, (THUMB, THUMB)), (x, y_img))
        predicted, confidence, _ = predict(img_path, model, encoder)
        correct = predicted == class_name
        draw.rectangle((x, y_img + THUMB, x + THUMB, y_img + THUMB + PRED_H - 4), fill=(50, 160, 80) if correct else (200, 60, 60))
        draw.text((x + 4, y_img + THUMB + 6), img_path.name[:18], fill="white")
        draw.text((x + 4, y_img + THUMB + 22), f"{'OK' if correct else 'X'} {predicted} ({confidence*100:.0f}%)"[:22], fill="white")

    draw.rectangle((0, 0, card_w - 1, card_h - 1), outline=(40, 80, 140), width=2)
    out_path = OUTPUT_DIR / f"class_{class_name}.png"
    card.save(out_path)
    print(f"  Saved: {out_path}")

# Comparative grid
print("\n[INFO] Generating comparative grid...")
GRID_THUMB, GRID_ROWS, GRID_HEADER, GRID_LEFT = 120, 3, 60, 160
CELL_W, CELL_H = GRID_THUMB + 10, GRID_THUMB + 40
GRID_COLS = len(class_names)
grid_w = GRID_LEFT + CELL_W * GRID_COLS + 20
grid_h = GRID_HEADER + CELL_H * GRID_ROWS + 20

grid = Image.new("RGB", (grid_w, grid_h), (250, 250, 255))
draw = ImageDraw.Draw(grid)

draw.rectangle((0, 0, grid_w, GRID_HEADER), fill=(30, 60, 130))
draw.text((20, 10), "Microscopic Organism - Comparative Image Samples", fill="white")
draw.text((20, 32), f"Classes: {', '.join(class_names)}", fill=(180, 200, 240))

for col, cls in enumerate(class_names):
    x = GRID_LEFT + col * CELL_W + 5
    draw.rectangle((x, GRID_HEADER, x + CELL_W - 4, GRID_HEADER + 28), fill=(60, 100, 180))
    draw.text((x + 4, GRID_HEADER + 7), cls.replace("_bacteria", "_bact").replace("Spherical", "Spher.")[:14], fill="white")

for row in range(GRID_ROWS):
    draw.text((8, GRID_HEADER + 28 + row * CELL_H + CELL_H // 2 - 8), f"Sample {row+1}", fill=(40, 80, 140))

for col, cls in enumerate(class_names):
    for row, img_path in enumerate(get_sample_images(cls, GRID_ROWS)):
        x = GRID_LEFT + col * CELL_W + 5
        y = GRID_HEADER + 28 + row * CELL_H
        grid.paste(load_and_resize(img_path, (GRID_THUMB, GRID_THUMB)), (x, y))
        predicted, conf, _ = predict(img_path, model, encoder)
        draw.rectangle((x, y + GRID_THUMB, x + GRID_THUMB, y + GRID_THUMB + 22), fill=(50, 160, 80) if predicted == cls else (200, 60, 60))
        draw.text((x + 2, y + GRID_THUMB + 4), f"{'OK' if predicted == cls else 'X'} {conf*100:.0f}%", fill="white")

for col in range(GRID_COLS + 1):
    draw.line((GRID_LEFT + col * CELL_W + 2, GRID_HEADER, GRID_LEFT + col * CELL_W + 2, grid_h - 10), fill=(180, 190, 220))
for row in range(GRID_ROWS + 1):
    draw.line((GRID_LEFT, GRID_HEADER + 28 + row * CELL_H, grid_w - 10, GRID_HEADER + 28 + row * CELL_H), fill=(180, 190, 220))

draw.rectangle((0, 0, grid_w - 1, grid_h - 1), outline=(30, 60, 130), width=3)
grid.save(OUTPUT_DIR / "comparative_grid.png")
print(f"  Saved: {OUTPUT_DIR / 'comparative_grid.png'}")

# Naming criteria card
print("\n[INFO] Generating naming criteria card...")
CRIT_W, ROW_H = 900, 70
CRIT_H = 60 + ROW_H * len(class_names) + 20
crit_img = Image.new("RGB", (CRIT_W, CRIT_H), (250, 252, 255))
draw = ImageDraw.Draw(crit_img)

draw.rectangle((0, 0, CRIT_W, 55), fill=(30, 60, 130))
draw.text((20, 16), "Microscopic Organism - Naming Criteria Reference", fill="white")

for i, (cls, criteria) in enumerate(NAMING_CRITERIA.items()):
    y = 60 + i * ROW_H
    draw.rectangle((0, y, CRIT_W, y + ROW_H - 2), fill=[(235, 242, 255), (245, 250, 255)][i % 2])
    draw.rectangle((0, y, 8, y + ROW_H - 2), fill=(40, 80, 160))
    draw.text((20, y + 8), cls, fill=(20, 50, 120))
    draw_text_wrapped(draw, criteria, 20, y + 30, CRIT_W - 40, fill=(60, 60, 80))
    draw.line((0, y + ROW_H - 2, CRIT_W, y + ROW_H - 2), fill=(200, 210, 230))

draw.rectangle((0, 0, CRIT_W - 1, CRIT_H - 1), outline=(30, 60, 130), width=2)
crit_img.save(OUTPUT_DIR / "naming_criteria.png")
print(f"  Saved: {OUTPUT_DIR / 'naming_criteria.png'}")

print("\nDone. All outputs saved in 'image_based_output/' folder.")
