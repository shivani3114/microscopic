import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import os
import random

random.seed(42)

MODEL_DIR = Path("outputs")
DATASET_DIR = Path("dataset/test")
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

def load_model():
    try:
        import joblib
        if not SK_MODEL_PATH.exists():
            return None, None
        payload = joblib.load(SK_MODEL_PATH)
        return payload["model"], payload["label_encoder"]
    except:
        return None, None

def predict(image_path, model, encoder):
    try:
        img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
        features = (np.asarray(img, dtype=np.float32) / 255.0).flatten().reshape(1, -1)
        pred_idx = model.predict(features)[0]
        probs = model.predict_proba(features)[0]
        predicted = encoder.inverse_transform([pred_idx])[0]
        return predicted, float(np.max(probs)), probs
    except:
        return None, None, None

def get_sample_images(class_name, n=3):
    class_dir = DATASET_DIR / class_name
    if not class_dir.exists():
        return []
    images = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
    return images[:n]

def load_and_resize(path, size=(160, 160)):
    try:
        return Image.open(path).convert("RGB").resize(size)
    except:
        return Image.new("RGB", size, (200, 200, 200))

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

def generate_prediction_result(image_path, output_dir="outputs"):
    """Generate prediction result visualization for uploaded image"""
    Path(output_dir).mkdir(exist_ok=True)
    
    model, encoder = load_model()
    if model is None:
        return None
    
    try:
        # Predict
        predicted, confidence, probs = predict(image_path, model, encoder)
        if predicted is None:
            return None
        
        # Load uploaded image
        uploaded_img = load_and_resize(image_path, (200, 200))
        
        # Create result card
        card_w, card_h = 600, 400
        card = Image.new("RGB", (card_w, card_h), (245, 248, 252))
        draw = ImageDraw.Draw(card)
        
        # Header
        draw.rectangle((0, 0, card_w, 60), fill=(40, 80, 140))
        draw.text((20, 15), "Prediction Result", fill="white")
        
        # Uploaded image
        card.paste(uploaded_img, (30, 80))
        
        # Prediction info
        x_info = 250
        y_info = 90
        
        draw.text((x_info, y_info), "Predicted Species:", fill=(40, 80, 140))
        draw.text((x_info, y_info + 25), predicted, fill=(50, 150, 50))
        
        draw.text((x_info, y_info + 55), "Confidence:", fill=(40, 80, 140))
        draw.text((x_info, y_info + 80), f"{confidence * 100:.1f}%", fill=(50, 150, 50))
        
        # Confidence bar
        bar_w = 150
        bar_h = 20
        bar_filled = int(bar_w * confidence)
        draw.rectangle((x_info, y_info + 110, x_info + bar_w, y_info + 110 + bar_h), fill=(200, 200, 200))
        draw.rectangle((x_info, y_info + 110, x_info + bar_filled, y_info + 110 + bar_h), fill=(50, 150, 50))
        
        # Criteria
        y_crit = 280
        draw.text((20, y_crit), "Morphological Description:", fill=(40, 80, 140))
        criteria = NAMING_CRITERIA.get(predicted, "")
        draw_text_wrapped(draw, criteria, 20, y_crit + 25, card_w - 40, fill=(60, 60, 80))
        
        # Border
        draw.rectangle((0, 0, card_w - 1, card_h - 1), outline=(40, 80, 140), width=2)
        
        output_path = Path(output_dir) / "prediction_result.png"
        card.save(output_path)
        
        return {
            "predicted": predicted,
            "confidence": confidence,
            "image_path": str(output_path)
        }
    except Exception as e:
        print(f"Error generating prediction result: {e}")
        return None

def generate_similar_samples(predicted_class, output_dir="outputs"):
    """Generate grid of similar organism samples"""
    Path(output_dir).mkdir(exist_ok=True)
    
    try:
        # Get samples of predicted class and similar morphologies
        classes_to_show = [predicted_class]
        
        samples_dict = {}
        for cls in classes_to_show:
            samples = get_sample_images(cls, 4)
            if samples:
                samples_dict[cls] = samples
        
        if not samples_dict:
            return None
        
        # Create grid
        THUMB = 120
        ROWS = 4
        cols = len(samples_dict)
        CELL_W = THUMB + 10
        CELL_H = THUMB + 50
        
        grid_w = 50 + CELL_W * cols + 20
        grid_h = 80 + CELL_H * ROWS + 20
        
        grid = Image.new("RGB", (grid_w, grid_h), (250, 250, 255))
        draw = ImageDraw.Draw(grid)
        
        # Header
        draw.rectangle((0, 0, grid_w, 70), fill=(30, 60, 130))
        draw.text((20, 10), f"Similar Samples - {predicted_class}", fill="white")
        draw.text((20, 35), "Images from dataset showing similar morphology", fill=(180, 200, 240))
        
        # Column headers
        for col, cls in enumerate(samples_dict.keys()):
            x = 50 + col * CELL_W
            draw.rectangle((x, 70, x + CELL_W - 4, 95), fill=(60, 100, 180))
            draw.text((x + 4, 75), cls[:16], fill="white")
        
        # Images
        model, encoder = load_model()
        for col, (cls, images) in enumerate(samples_dict.items()):
            for row, img_path in enumerate(images[:ROWS]):
                x = 50 + col * CELL_W
                y = 95 + row * CELL_H
                
                grid.paste(load_and_resize(img_path, (THUMB, THUMB)), (x, y))
                
                if model and encoder:
                    predicted, conf, _ = predict(img_path, model, encoder)
                    color = (50, 160, 80) if predicted == cls else (200, 60, 60)
                    draw.rectangle((x, y + THUMB, x + THUMB, y + THUMB + 22), fill=color)
                    draw.text((x + 2, y + THUMB + 4), f"{conf*100:.0f}%" if conf else "N/A", fill="white")
        
        # Border
        draw.rectangle((0, 0, grid_w - 1, grid_h - 1), outline=(30, 60, 130), width=2)
        
        output_path = Path(output_dir) / "similar_samples.png"
        grid.save(output_path)
        return str(output_path)
    except Exception as e:
        print(f"Error generating similar samples: {e}")
        return None
