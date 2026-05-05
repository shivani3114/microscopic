import random
from pathlib import Path
from PIL import Image, ImageDraw

random.seed(42)

CLASSES = ["Amoeba", "Euglena", "Hydra", "Paramecium", "Rod_bacteria", "Spherical_bacteria", "Spiral_bacteria", "Yeast"]
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
THUMB = 150

def get_sample(cls):
    images = [p for p in Path(f"dataset/test/{cls}").iterdir() if p.suffix.lower() in VALID_EXTS]
    return random.choice(images) if images else None

cols, rows = len(CLASSES), 1
W, H = cols * (THUMB + 10) + 10, THUMB + 50
out = Image.new("RGB", (W, H), (240, 240, 240))
draw = ImageDraw.Draw(out)

for i, cls in enumerate(CLASSES):
    img_path = get_sample(cls)
    if img_path:
        img = Image.open(img_path).convert("RGB").resize((THUMB, THUMB))
        x = 10 + i * (THUMB + 10)
        out.paste(img, (x, 10))
        draw.rectangle((x, THUMB + 10, x + THUMB, THUMB + 35), fill=(40, 80, 140))
        draw.text((x + 5, THUMB + 15), cls[:14], fill="white")

out.save("outputs/sample_output.png")
print("Saved: outputs/sample_output.png")
