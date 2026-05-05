import json
import random
import numpy as np
import joblib
from pathlib import Path
from PIL import Image, ImageDraw
from datetime import datetime

random.seed(42)

# Paths
UPLOAD_DIR = Path("../static/uploads")
OUTPUT_DIR = Path("outputs")
DATASET_DIR = Path("dataset/test")
SK_MODEL_PATH = OUTPUT_DIR / "microscopic_classifier.joblib"
CLASS_NAMES_PATH = OUTPUT_DIR / "class_names.json"
IMG_SIZE = (64, 64)

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
    try:
        return Image.open(path).convert("RGB").resize(size)
    except:
        return Image.new("RGB", size, (200, 200, 200))

def get_sample_images(class_name, n=3):
    class_dir = DATASET_DIR / class_name
    if not class_dir.exists():
        return []
    images = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    random.shuffle(images)
    return images[:n]

def predict(image_path, model, encoder):
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    features = (np.asarray(img, dtype=np.float32) / 255.0).flatten().reshape(1, -1)
    pred_idx = model.predict(features)[0]
    probs = model.predict_proba(features)[0]
    predicted = encoder.inverse_transform([pred_idx])[0]
    confidence = float(np.max(probs))
    return predicted, confidence, probs

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

# Load model
payload = joblib.load(SK_MODEL_PATH)
model = payload["model"]
encoder = payload["label_encoder"]
with open(CLASS_NAMES_PATH) as f:
    class_names = json.load(f)

# Process uploaded image
uploaded_images = list(UPLOAD_DIR.glob("*.jpg")) + list(UPLOAD_DIR.glob("*.jpeg")) + list(UPLOAD_DIR.glob("*.png"))

if uploaded_images:
    print(f"[INFO] Processing {len(uploaded_images)} uploaded image(s)...")
    
    for uploaded_path in uploaded_images:
        print(f"\n[PROCESSING] {uploaded_path.name}")
        
        # Predict
        predicted_class, confidence, all_probs = predict(uploaded_path, model, encoder)
        
        # Create prediction result card
        CARD_W, CARD_H = 700, 350
        card = Image.new("RGB", (CARD_W, CARD_H), (245, 250, 255))
        draw = ImageDraw.Draw(card)
        
        # Header
        draw.rectangle((0, 0, CARD_W, 60), fill=(30, 80, 150))
        draw.text((20, 15), f"Uploaded Image Analysis - {uploaded_path.stem}", fill="white")
        
        # Left side: uploaded image
        IMG_DISPLAY_SIZE = 160
        uploaded_display = load_and_resize(uploaded_path, (IMG_DISPLAY_SIZE, IMG_DISPLAY_SIZE))
        card.paste(uploaded_display, (20, 80))
        
        # Right side: prediction info
        px, py = 200, 80
        draw.text((px, py), "PREDICTION RESULT:", fill=(30, 80, 150))
        draw.text((px, py + 28), f"Class: {predicted_class}", fill=(20, 60, 140), )
        draw.text((px, py + 50), f"Confidence: {confidence*100:.1f}%", fill=(50, 100, 180))
        
        # Criteria
        draw.rectangle((20, 260, CARD_W - 20, 340), fill=(220, 230, 250))
        draw.text((30, 270), "Class Characteristics:", fill=(30, 80, 150))
        draw_text_wrapped(draw, NAMING_CRITERIA.get(predicted_class, ""), 30, 290, CARD_W - 60, line_height=14, fill=(50, 80, 120))
        
        draw.rectangle((0, 0, CARD_W - 1, CARD_H - 1), outline=(30, 80, 150), width=2)
        result_path = OUTPUT_DIR / f"prediction_result_{uploaded_path.stem}.png"
        card.save(result_path)
        print(f"  Saved: {result_path}")
        
        # Create similar organism comparison
        print(f"  Creating {predicted_class} comparison...")
        similar_samples = get_sample_images(predicted_class, 3)
        
        if similar_samples:
            COMP_W, COMP_H = 750, 280
            comp = Image.new("RGB", (COMP_W, COMP_H), (245, 250, 255))
            draw = ImageDraw.Draw(comp)
            
            # Header
            draw.rectangle((0, 0, COMP_W, 50), fill=(30, 80, 150))
            draw.text((20, 12), f"Similar {predicted_class} Organisms from Dataset", fill="white")
            
            # Uploaded image on left
            comp.paste(uploaded_display, (15, 70))
            draw.text((15, 250), "Uploaded", fill=(50, 80, 120))
            
            # Similar samples
            thumb_size = 140
            for i, sample in enumerate(similar_samples):
                x = 180 + i * (thumb_size + 15)
                sample_thumb = load_and_resize(sample, (thumb_size, thumb_size))
                comp.paste(sample_thumb, (x, 70))
                pred, conf, _ = predict(sample, model, encoder)
                draw.rectangle((x, 220, x + thumb_size, 245), fill=(50, 160, 80) if pred == predicted_class else (200, 80, 80))
                draw.text((x + 4, 225), f"{conf*100:.0f}%", fill="white")
            
            draw.rectangle((0, 0, COMP_W - 1, COMP_H - 1), outline=(30, 80, 150), width=2)
            comp_path = OUTPUT_DIR / f"similar_{predicted_class}_{uploaded_path.stem}.png"
            comp.save(comp_path)
            print(f"  Saved: {comp_path}")

print("\n[DONE] Uploaded image analysis complete.")
