# Morphological Species Identifier

A Flask-based web application for classifying microscopic organisms using machine learning. Upload images of microscopic samples and get AI-powered species identification with confidence scores and morphological descriptions.

## Features

- 🦠 **AI-Powered Classification**: Uses machine learning to identify microscopic species
- 📊 **Confidence Scoring**: Provides prediction confidence percentages
- 🔬 **Morphological Analysis**: Displays detailed morphological characteristics
- 📸 **Sample Visualization**: Shows similar samples from the training dataset
- 🎯 **Ground Truth Comparison**: Compare predictions with expected results
- 📱 **Responsive Design**: Modern, mobile-friendly interface
- ⚡ **AJAX Upload**: Seamless file upload without page refresh

## Installation

1. **Clone or download** the project files to your local machine

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model files exist**:
   - The application expects `outputs/microscopic_classifier.joblib` (trained ML model)
   - Dataset should be in the `dataset/` directory with subfolders for each species

## Usage

1. **Start the application**:
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to `http://localhost:5000`

3. **Upload an image**:
   - Click "Choose Image" or drag and drop an image file
   - Optionally enter ground truth (expected species name)
   - Click "🔍 Analyze Species"

4. **View results**:
   - Predicted species with confidence score
   - Morphological description
   - Your uploaded image
   - Similar samples from the dataset
   - Ground truth comparison (if provided)

## Project Structure

```
Micro_Organism/
├── app.py                 # Main Flask application
├── micro.py              # Alternative Flask app (legacy)
├── mlb.py                # ML model training script
├── generate_visuals.py   # Data visualization script
├── sample_output.py      # Sample generation script
├── gui_app.py           # GUI application (Tkinter)
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html       # Main web interface
├── static/              # Static files (CSS, JS, images)
│   └── uploads/        # Uploaded images
├── outputs/             # Generated outputs and model files
│   ├── microscopic_classifier.joblib  # Trained ML model
│   ├── class_names.json               # Species class names
│   └── classification_report.json     # Model performance report
└── dataset/             # Training/validation data
    ├── test/
    ├── train/
    └── val/
```

## API Endpoints

- `GET /`: Main web interface
- `POST /analyze`: Analyze uploaded image
  - **Input**: `multipart/form-data` with `image` file and optional `ground_truth`
  - **Output**: JSON response with prediction results

## Supported Species

The model can classify the following microscopic organisms:
- Amoeba
- Euglena
- Hydra
- Paramecium
- Rod bacteria
- Spherical bacteria
- Spiral bacteria
- Yeast

## Requirements

- Python 3.8+
- Flask
- scikit-learn
- Pillow (PIL)
- NumPy
- Joblib

## Troubleshooting

**Site not reachable**:
- Ensure Flask app is running (`python app.py`)
- Check that port 5000 is not blocked
- Verify all dependencies are installed

**Model loading errors**:
- Ensure `outputs/microscopic_classifier.joblib` exists
- Check file permissions
- Verify scikit-learn and joblib versions

**Image processing errors**:
- Ensure uploaded images are valid image files
- Check file size limits
- Verify Pillow installation

## Development

To modify the application:

1. **Backend changes**: Edit `app.py`
2. **Frontend changes**: Edit `templates/index.html`
3. **Model training**: Use `mlb.py` to retrain the model
4. **Data visualization**: Run `generate_visuals.py`

## License

This project is for educational and research purposes.