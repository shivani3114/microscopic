import os
import json
import random
import numpy as np
import joblib
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageTk

import customtkinter as ctk

# ── Config ──────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

random.seed(42)
DATASET_DIR  = Path("dataset/test")
OUTPUT_DIR   = Path("outputs")
IMG_SIZE     = (64, 64)
VALID_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

NAMING_CRITERIA = {
    "Amoeba":            "Irregular blob-like shape with pseudopods; no fixed form",
    "Euglena":           "Elongated spindle-shaped with a visible flagellum",
    "Hydra":             "Tubular body with tentacles extending from one end",
    "Paramecium":        "Oval/slipper-shaped, covered with cilia all around",
    "Rod_bacteria":      "Cylindrical rod-shaped cells (Bacilli); straight edges",
    "Spherical_bacteria":"Round/spherical cells (Cocci); circular uniform shape",
    "Spiral_bacteria":   "Twisted/helical/spiral-shaped cells (Spirilla)",
    "Yeast":             "Oval budding cells; single or clustered round shapes",
}

CLASS_COLORS = {
    "Amoeba":            "#e74c3c",
    "Euglena":           "#2ecc71",
    "Hydra":             "#3498db",
    "Paramecium":        "#9b59b6",
    "Rod_bacteria":      "#e67e22",
    "Spherical_bacteria":"#1abc9c",
    "Spiral_bacteria":   "#f39c12",
    "Yeast":             "#e91e63",
}

# ── Load model ───────────────────────────────────────────────
payload      = joblib.load(OUTPUT_DIR / "microscopic_classifier.joblib")
MODEL        = payload["model"]
ENCODER      = payload["label_encoder"]
with open(OUTPUT_DIR / "class_names.json") as f:
    CLASS_NAMES = json.load(f)

# ── Helpers ──────────────────────────────────────────────────
def get_samples(cls, n=4):
    d = DATASET_DIR / cls
    if not d.exists():
        return []
    imgs = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
    random.shuffle(imgs)
    return imgs[:n]

def predict_image(path):
    img  = Image.open(path).convert("RGB").resize(IMG_SIZE)
    arr  = np.asarray(img, dtype=np.float32) / 255.0
    feat = arr.flatten().reshape(1, -1)
    idx  = MODEL.predict(feat)[0]
    prob = MODEL.predict_proba(feat)[0]
    return ENCODER.inverse_transform([idx])[0], float(np.max(prob)), prob

def pil_to_ctk(path, size=(150, 150)):
    img = Image.open(path).convert("RGB").resize(size)
    return ImageTk.PhotoImage(img)

def load_report():
    with open(OUTPUT_DIR / "classification_report.json") as f:
        return json.load(f)

# ════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════
class MicroscopeApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Microscopic Organism Classifier")
        self.geometry("1200x750")
        self.resizable(True, True)
        self._photo_refs = []   # keep PhotoImage refs alive

        self._build_sidebar()
        self._build_main()
        self._show_tab("classify")

    # ── Sidebar ─────────────────────────────────────────────
    def _build_sidebar(self):
        sb = ctk.CTkFrame(self, width=200, corner_radius=0, fg_color="#1a1a2e")
        sb.pack(side="left", fill="y")
        sb.pack_propagate(False)

        ctk.CTkLabel(sb, text="Microscopic\nClassifier",
                     font=ctk.CTkFont(size=18, weight="bold"),
                     text_color="#4fc3f7").pack(pady=(30, 20))

        buttons = [
            ("Classify Image",   "classify",   "🔬"),
            ("Compare Classes",  "compare",    "📊"),
            ("Naming Criteria",  "criteria",   "📋"),
            ("Model Metrics",    "metrics",    "📈"),
        ]
        self._nav_btns = {}
        for label, key, icon in buttons:
            btn = ctk.CTkButton(
                sb, text=f"  {icon}  {label}", anchor="w",
                fg_color="transparent", hover_color="#16213e",
                font=ctk.CTkFont(size=13),
                command=lambda k=key: self._show_tab(k)
            )
            btn.pack(fill="x", padx=10, pady=4)
            self._nav_btns[key] = btn

        ctk.CTkLabel(sb, text="789 images  |  8 classes",
                     font=ctk.CTkFont(size=10), text_color="gray").pack(side="bottom", pady=15)

    # ── Main area ────────────────────────────────────────────
    def _build_main(self):
        self._main = ctk.CTkFrame(self, fg_color="#0f0f23")
        self._main.pack(side="left", fill="both", expand=True)

        self._frames = {}
        for key, builder in [
            ("classify", self._build_classify),
            ("compare",  self._build_compare),
            ("criteria", self._build_criteria),
            ("metrics",  self._build_metrics),
        ]:
            f = ctk.CTkFrame(self._main, fg_color="#0f0f23")
            f.place(relx=0, rely=0, relwidth=1, relheight=1)
            builder(f)
            self._frames[key] = f

    def _show_tab(self, key):
        for k, f in self._frames.items():
            f.lower()
        self._frames[key].lift()
        for k, b in self._nav_btns.items():
            b.configure(fg_color="#16213e" if k == key else "transparent")

    # ════════════════════════════════════════════════════════
    # TAB 1 – CLASSIFY IMAGE
    # ════════════════════════════════════════════════════════
    def _build_classify(self, parent):
        ctk.CTkLabel(parent, text="Classify a Microscopic Image",
                     font=ctk.CTkFont(size=20, weight="bold"),
                     text_color="#4fc3f7").pack(pady=(20, 10))

        top = ctk.CTkFrame(parent, fg_color="#16213e", corner_radius=12)
        top.pack(fill="x", padx=20, pady=5)

        ctk.CTkButton(top, text="Browse Image", width=160,
                      command=self._browse_image).pack(side="left", padx=15, pady=12)
        self._file_label = ctk.CTkLabel(top, text="No file selected", text_color="gray")
        self._file_label.pack(side="left", padx=10)

        ctk.CTkButton(top, text="Random Test Image", width=160,
                      fg_color="#2d6a4f",
                      command=self._random_image).pack(side="right", padx=15, pady=12)

        # Result area
        result = ctk.CTkFrame(parent, fg_color="#16213e", corner_radius=12)
        result.pack(fill="both", expand=True, padx=20, pady=10)

        left = ctk.CTkFrame(result, fg_color="transparent")
        left.pack(side="left", fill="both", expand=True, padx=20, pady=20)

        self._img_label = ctk.CTkLabel(left, text="", width=300, height=300)
        self._img_label.pack()
        self._img_name_label = ctk.CTkLabel(left, text="", text_color="gray",
                                             font=ctk.CTkFont(size=11))
        self._img_name_label.pack(pady=5)

        right = ctk.CTkFrame(result, fg_color="transparent")
        right.pack(side="left", fill="both", expand=True, padx=20, pady=20)

        self._pred_label = ctk.CTkLabel(right, text="Prediction will appear here",
                                         font=ctk.CTkFont(size=22, weight="bold"),
                                         text_color="#4fc3f7")
        self._pred_label.pack(pady=(30, 5))

        self._conf_label = ctk.CTkLabel(right, text="", font=ctk.CTkFont(size=14))
        self._conf_label.pack()

        self._criteria_label = ctk.CTkLabel(right, text="", wraplength=380,
                                             font=ctk.CTkFont(size=12),
                                             text_color="#a0c4ff")
        self._criteria_label.pack(pady=10)

        # Probability bars
        self._bar_frame = ctk.CTkScrollableFrame(right, fg_color="transparent", height=250)
        self._bar_frame.pack(fill="x", pady=10)

    def _browse_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if path:
            self._run_prediction(path)

    def _random_image(self):
        cls  = random.choice(CLASS_NAMES)
        imgs = get_samples(cls, 10)
        if imgs:
            self._run_prediction(str(random.choice(imgs)))

    def _run_prediction(self, path):
        self._file_label.configure(text=Path(path).name, text_color="white")
        self._img_name_label.configure(text=Path(path).name)

        photo = ImageTk.PhotoImage(Image.open(path).convert("RGB").resize((300, 300)))
        self._photo_refs.append(photo)
        self._img_label.configure(image=photo, text="")

        predicted, conf, probs = predict_image(path)
        color = CLASS_COLORS.get(predicted, "#4fc3f7")

        self._pred_label.configure(text=predicted, text_color=color)
        self._conf_label.configure(text=f"Confidence: {conf*100:.1f}%",
                                    text_color="#2ecc71" if conf > 0.5 else "#e74c3c")
        self._criteria_label.configure(
            text=f"Criteria: {NAMING_CRITERIA.get(predicted, '')}")

        # Clear old bars
        for w in self._bar_frame.winfo_children():
            w.destroy()

        sorted_idx = np.argsort(probs)[::-1]
        for idx in sorted_idx:
            cls   = CLASS_NAMES[idx]
            p     = float(probs[idx])
            row   = ctk.CTkFrame(self._bar_frame, fg_color="transparent")
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=cls, width=160, anchor="w",
                         font=ctk.CTkFont(size=11)).pack(side="left")
            bar = ctk.CTkProgressBar(row, width=200,
                                      progress_color=CLASS_COLORS.get(cls, "#4fc3f7"))
            bar.set(p)
            bar.pack(side="left", padx=5)
            ctk.CTkLabel(row, text=f"{p*100:.1f}%",
                         font=ctk.CTkFont(size=11)).pack(side="left")

    # ════════════════════════════════════════════════════════
    # TAB 2 – COMPARE CLASSES
    # ════════════════════════════════════════════════════════
    def _build_compare(self, parent):
        ctk.CTkLabel(parent, text="Comparative Image Samples",
                     font=ctk.CTkFont(size=20, weight="bold"),
                     text_color="#4fc3f7").pack(pady=(20, 5))

        # Class selector
        sel_frame = ctk.CTkFrame(parent, fg_color="#16213e", corner_radius=10)
        sel_frame.pack(fill="x", padx=20, pady=5)

        ctk.CTkLabel(sel_frame, text="Select Class:").pack(side="left", padx=15, pady=8)
        self._compare_var = ctk.StringVar(value=CLASS_NAMES[0])
        menu = ctk.CTkOptionMenu(sel_frame, values=CLASS_NAMES,
                                  variable=self._compare_var,
                                  command=self._load_compare)
        menu.pack(side="left", padx=10)

        ctk.CTkButton(sel_frame, text="Refresh Samples", width=140,
                      command=lambda: self._load_compare(self._compare_var.get())
                      ).pack(side="right", padx=15, pady=8)

        # Grid area
        self._compare_scroll = ctk.CTkScrollableFrame(parent, fg_color="#16213e",
                                                       corner_radius=12)
        self._compare_scroll.pack(fill="both", expand=True, padx=20, pady=10)
        self._load_compare(CLASS_NAMES[0])

    def _load_compare(self, cls):
        for w in self._compare_scroll.winfo_children():
            w.destroy()

        color = CLASS_COLORS.get(cls, "#4fc3f7")
        ctk.CTkLabel(self._compare_scroll,
                     text=f"  {cls}  —  {NAMING_CRITERIA.get(cls,'')}",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=color, anchor="w").pack(fill="x", pady=(5, 10))

        samples = get_samples(cls, 8)
        if not samples:
            ctk.CTkLabel(self._compare_scroll, text="No test images found.",
                         text_color="gray").pack()
            return

        grid = ctk.CTkFrame(self._compare_scroll, fg_color="transparent")
        grid.pack()

        for i, img_path in enumerate(samples):
            col = i % 4
            row = i // 4
            card = ctk.CTkFrame(grid, fg_color="#0f0f23", corner_radius=10)
            card.grid(row=row, column=col, padx=8, pady=8)

            photo = ImageTk.PhotoImage(
                Image.open(img_path).convert("RGB").resize((160, 160)))
            self._photo_refs.append(photo)
            ctk.CTkLabel(card, image=photo, text="").pack(padx=6, pady=6)

            predicted, conf, _ = predict_image(img_path)
            correct = predicted == cls
            tag_color = "#2ecc71" if correct else "#e74c3c"
            tag_text  = f"{'OK' if correct else 'X'}  {predicted}  {conf*100:.0f}%"

            ctk.CTkLabel(card, text=img_path.name[:20],
                         font=ctk.CTkFont(size=10), text_color="gray").pack()
            ctk.CTkLabel(card, text=tag_text,
                         font=ctk.CTkFont(size=11, weight="bold"),
                         text_color=tag_color).pack(pady=(0, 6))

    # ════════════════════════════════════════════════════════
    # TAB 3 – NAMING CRITERIA
    # ════════════════════════════════════════════════════════
    def _build_criteria(self, parent):
        ctk.CTkLabel(parent, text="Naming Criteria Reference",
                     font=ctk.CTkFont(size=20, weight="bold"),
                     text_color="#4fc3f7").pack(pady=(20, 10))

        scroll = ctk.CTkScrollableFrame(parent, fg_color="#0f0f23")
        scroll.pack(fill="both", expand=True, padx=20, pady=10)

        for cls, criteria in NAMING_CRITERIA.items():
            color = CLASS_COLORS.get(cls, "#4fc3f7")
            card  = ctk.CTkFrame(scroll, fg_color="#16213e", corner_radius=12)
            card.pack(fill="x", pady=6, padx=4)

            bar = ctk.CTkFrame(card, width=6, fg_color=color, corner_radius=3)
            bar.pack(side="left", fill="y", padx=(8, 0), pady=8)

            info = ctk.CTkFrame(card, fg_color="transparent")
            info.pack(side="left", fill="both", expand=True, padx=12, pady=10)

            ctk.CTkLabel(info, text=cls,
                         font=ctk.CTkFont(size=15, weight="bold"),
                         text_color=color, anchor="w").pack(fill="x")
            ctk.CTkLabel(info, text=criteria,
                         font=ctk.CTkFont(size=12),
                         text_color="#cccccc", anchor="w",
                         wraplength=700).pack(fill="x", pady=2)

            # Show 1 sample image
            samples = get_samples(cls, 1)
            if samples:
                try:
                    photo = ImageTk.PhotoImage(
                        Image.open(samples[0]).convert("RGB").resize((80, 80)))
                    self._photo_refs.append(photo)
                    ctk.CTkLabel(card, image=photo, text="").pack(
                        side="right", padx=12, pady=8)
                except Exception:
                    pass

    # ════════════════════════════════════════════════════════
    # TAB 4 – MODEL METRICS
    # ════════════════════════════════════════════════════════
    def _build_metrics(self, parent):
        ctk.CTkLabel(parent, text="Model Performance Metrics",
                     font=ctk.CTkFont(size=20, weight="bold"),
                     text_color="#4fc3f7").pack(pady=(20, 10))

        report = load_report()
        overall_acc = report.get("accuracy", 0)

        # Summary cards
        summary = ctk.CTkFrame(parent, fg_color="transparent")
        summary.pack(fill="x", padx=20, pady=5)

        for label, value, color in [
            ("Overall Accuracy", f"{overall_acc*100:.1f}%", "#4fc3f7"),
            ("Total Classes",    str(len(CLASS_NAMES)),      "#2ecc71"),
            ("Test Images",      "126",                      "#e67e22"),
            ("Backend",          "sklearn MLP",              "#9b59b6"),
        ]:
            card = ctk.CTkFrame(summary, fg_color="#16213e", corner_radius=12)
            card.pack(side="left", expand=True, fill="x", padx=6)
            ctk.CTkLabel(card, text=value,
                         font=ctk.CTkFont(size=26, weight="bold"),
                         text_color=color).pack(pady=(12, 2))
            ctk.CTkLabel(card, text=label,
                         font=ctk.CTkFont(size=11),
                         text_color="gray").pack(pady=(0, 12))

        # Per-class table
        ctk.CTkLabel(parent, text="Per-Class Metrics",
                     font=ctk.CTkFont(size=14, weight="bold"),
                     text_color="white").pack(anchor="w", padx=24, pady=(15, 5))

        scroll = ctk.CTkScrollableFrame(parent, fg_color="#16213e",
                                         corner_radius=12, height=350)
        scroll.pack(fill="both", expand=True, padx=20, pady=(0, 15))

        headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
        for col, h in enumerate(headers):
            ctk.CTkLabel(scroll, text=h,
                         font=ctk.CTkFont(size=12, weight="bold"),
                         text_color="#4fc3f7").grid(
                row=0, column=col, padx=20, pady=8, sticky="w")

        skip = {"accuracy", "macro avg", "weighted avg"}
        for row_i, cls in enumerate(CLASS_NAMES, start=1):
            if cls not in report:
                continue
            d     = report[cls]
            color = CLASS_COLORS.get(cls, "white")
            vals  = [cls,
                     f"{d['precision']:.2f}",
                     f"{d['recall']:.2f}",
                     f"{d['f1-score']:.2f}",
                     str(int(d['support']))]
            bg = "#1a1a2e" if row_i % 2 == 0 else "#16213e"
            for col_i, val in enumerate(vals):
                ctk.CTkLabel(scroll, text=val,
                             font=ctk.CTkFont(size=12),
                             text_color=color if col_i == 0 else "white").grid(
                    row=row_i, column=col_i, padx=20, pady=6, sticky="w")

        # Avg rows
        for avg_key in ["macro avg", "weighted avg"]:
            if avg_key in report:
                d = report[avg_key]
                row_i += 1
                vals = [avg_key,
                        f"{d['precision']:.2f}",
                        f"{d['recall']:.2f}",
                        f"{d['f1-score']:.2f}",
                        str(int(d['support']))]
                for col_i, val in enumerate(vals):
                    ctk.CTkLabel(scroll, text=val,
                                 font=ctk.CTkFont(size=11, weight="bold"),
                                 text_color="#aaaaaa").grid(
                        row=row_i, column=col_i, padx=20, pady=6, sticky="w")


# ── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    app = MicroscopeApp()
    app.mainloop()
