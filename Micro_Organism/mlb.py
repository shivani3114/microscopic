import os
import json
import shutil
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

SEED = 42
IMG_SIZE = (64, 64)
DATASET_DIR = "dataset"
OUTPUT_DIR = "outputs"
SK_MODEL_PATH = os.path.join(OUTPUT_DIR, "microscopic_classifier.joblib")
CLASS_NAMES_PATH = os.path.join(OUTPUT_DIR, "class_names.json")
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SOURCE_CLASSES = ["Amoeba", "Euglena", "Hydra", "Paramecium", "Rod_bacteria", "Spherical_bacteria", "Spiral_bacteria", "Yeast"]

random.seed(SEED)
np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def is_image_file(path):
    return Path(path).suffix.lower() in VALID_EXTS


def setup_dataset():
    base = Path(DATASET_DIR)
    if all((base / s).exists() for s in ["train", "val", "test"]):
        print(f"[INFO] Dataset already exists in '{DATASET_DIR}'.")
        return
    if base.exists():
        shutil.rmtree(base)
    splits = {"train": 0.7, "val": 0.15, "test": 0.15}
    for split in splits:
        for cls in SOURCE_CLASSES:
            (base / split / cls).mkdir(parents=True, exist_ok=True)
    for cls in SOURCE_CLASSES:
        images = [p for p in Path(cls).iterdir() if p.is_file() and is_image_file(p)]
        random.shuffle(images)
        n = len(images)
        t = int(n * splits["train"])
        v = int(n * splits["val"])
        for i, img in enumerate(images):
            split = "train" if i < t else "val" if i < t + v else "test"
            shutil.copy2(img, base / split / cls / img.name)
        print(f"  {cls}: {n} images -> train={t}, val={v}, test={n-t-v}")
    print("[INFO] Dataset setup complete.")


def extract_features(image_path):
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    return (np.asarray(img, dtype=np.float32) / 255.0).flatten()


def load_dataset():
    datasets, classes = {}, set()
    for split in ["train", "val", "test"]:
        X, y = [], []
        for cls_dir in sorted((Path(DATASET_DIR) / split).iterdir()):
            if not cls_dir.is_dir():
                continue
            classes.add(cls_dir.name)
            for img in sorted(cls_dir.iterdir()):
                if img.is_file() and is_image_file(img):
                    X.append(extract_features(str(img)))
                    y.append(cls_dir.name)
        datasets[split] = (np.array(X, dtype=np.float32), np.array(y))
    return datasets["train"], datasets["val"], datasets["test"], sorted(classes)


def train(train_data, val_data):
    X_train, y_train = train_data
    X_val, y_val = val_data
    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_val_enc = encoder.transform(y_val)
    model = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
                          alpha=1e-4, batch_size=16, learning_rate_init=1e-3,
                          max_iter=40, early_stopping=True, random_state=SEED, verbose=False)
    model.fit(X_train, y_train_enc)
    val_acc = float(np.mean(model.predict(X_val) == y_val_enc))
    print(f"[INFO] Validation Accuracy: {val_acc:.4f}")
    joblib.dump({"model": model, "label_encoder": encoder}, SK_MODEL_PATH)
    return {"model": model, "label_encoder": encoder}


def evaluate(payload, test_data, class_names):
    model, encoder = payload["model"], payload["label_encoder"]
    X_test, y_test = test_data
    y_pred_enc = model.predict(X_test)
    y_pred = encoder.inverse_transform(y_pred_enc)
    y_test_enc = encoder.transform(y_test)
    metrics = {
        "accuracy": float(np.mean(y_pred_enc == y_test_enc)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }
    report = classification_report(y_test, y_pred, labels=class_names, output_dict=True, zero_division=0)
    with open(os.path.join(OUTPUT_DIR, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=4)
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    save_confusion_matrix(cm, class_names)
    return metrics


def save_confusion_matrix(cm, class_names):
    n, cell, lm, tm = len(class_names), 70, 140, 100
    w, h = lm + cell * n + 40, tm + cell * n + 60
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    draw.text((w // 2 - 70, 20), "Confusion Matrix", fill="black")
    vmax = int(np.max(cm)) if cm.size else 1
    for i, cls in enumerate(class_names):
        draw.text((10, tm + i * cell + 25), cls[:12], fill="black")
        draw.text((lm + i * cell + 5, 70), cls[:10], fill="black")
    for r in range(n):
        for c in range(n):
            x0, y0 = lm + c * cell, tm + r * cell
            shade = int(255 - 170 * max(0.0, min(1.0, cm[r, c] / max(vmax, 1))))
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=(shade, shade, 255), outline="black")
            draw.text((x0 + 22, y0 + 24), str(int(cm[r, c])), fill="black")
    img.save(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))


def predict_and_save(image_path, payload, class_names):
    model, encoder = payload["model"], payload["label_encoder"]
    features = extract_features(image_path).reshape(1, -1)
    pred_idx = int(model.predict(features)[0])
    predicted = encoder.inverse_transform([pred_idx])[0]
    probs = model.predict_proba(features)[0]
    aligned = np.zeros(len(class_names), dtype=np.float32)
    for i, cls in enumerate(encoder.classes_):
        aligned[class_names.index(cls)] = probs[i]
    confidence = float(np.max(aligned))

    img = Image.open(image_path).convert("RGB").resize((300, 300))
    canvas = Image.new("RGB", (950, 460), "white")
    canvas.paste(img, (30, 80))
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 30), "Microscopic Organism Classification", fill="black")
    draw.text((30, 50), "Input Image", fill="black")
    draw.text((360, 60), f"Predicted: {predicted}", fill="black")
    draw.text((360, 84), f"Confidence: {confidence*100:.2f}%", fill="black")
    top3 = np.argsort(aligned)[::-1][:3]
    draw.text((360, 130), "Top 3 Predictions:", fill="black")
    for rank, idx in enumerate(top3):
        y = 155 + rank * 50
        draw.text((360, y), f"{rank+1}. {class_names[idx]}", fill="black")
        draw.rectangle((540, y, 900, y + 30), outline="black")
        draw.rectangle((540, y, 540 + int(360 * aligned[idx]), y + 30),
                       fill=(60, 160, 90) if class_names[idx] == predicted else (80, 140, 220))
        draw.text((910, y + 5), f"{aligned[idx]*100:.1f}%", fill="black")
    out = os.path.join(OUTPUT_DIR, "prediction_result.png")
    canvas.save(out)
    print(f"[INFO] Prediction saved to: {out}")


if __name__ == "__main__":
    setup_dataset()
    print("[INFO] Loading dataset...")
    train_data, val_data, test_data, class_names = load_dataset()
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f)

    print("[INFO] Training model...")
    payload = train(train_data, val_data)

    print("[INFO] Evaluating model...")
    metrics = evaluate(payload, test_data, class_names)
    print(f"[INFO] Test Accuracy: {metrics['accuracy']*100:.2f}%")
    print(metrics)

    for cls_dir in sorted((Path(DATASET_DIR) / "test").iterdir()):
        for img in sorted(cls_dir.iterdir()):
            if img.is_file() and is_image_file(img):
                predict_and_save(str(img), payload, class_names)
                break
        break
