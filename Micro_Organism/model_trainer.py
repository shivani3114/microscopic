import os
import numpy as np
import pandas as pd
try:
    from PIL import Image
except ImportError:
    print("PIL (Pillow) is not installed. Please install it with: pip install Pillow")
    exit(1)
try:
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score
except ImportError:
    print("scikit-learn is not installed. Please install it with: pip install scikit-learn")
    exit(1)
import json
from pathlib import Path

class MicroscopicClassifier:
    def __init__(self):
        self.model = None
        self.class_names = []
        self.categories = {
            'Protists': ['Amoeba', 'Euglena', 'Paramecium'],
            'Invertebrates': ['Hydra'],
            'Bacteria': ['Rod_bacteria', 'Spherical_bacteria', 'Spiral_bacteria'],
            'Fungi': ['Yeast']
        }

    def extract_features(self, image_path):
        """Extract basic features from image"""
        try:
            img = Image.open(image_path).convert('L')  # Convert to grayscale
            img = img.resize((64, 64))  # Resize for consistency

            # Convert to numpy array
            img_array = np.array(img)

            # Extract features
            features = []

            # Basic statistical features
            features.extend([img_array.mean(), img_array.std(), img_array.min(), img_array.max()])

            # Histogram features (16 bins)
            hist, _ = np.histogram(img_array, bins=16, range=(0, 255))
            features.extend(hist / hist.sum())  # Normalize

            # Shape features
            features.extend([img_array.shape[0], img_array.shape[1]])

            return np.array(features)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def load_dataset(self, dataset_path):
        """Load and prepare dataset"""
        X = []
        y = []

        print(f"Loading dataset from: {dataset_path}")
        print(f"Absolute path: {os.path.abspath(dataset_path)}")

        for category, species_list in self.categories.items():
            print(f"\nProcessing category: {category}")
            for species in species_list:
                species_path = os.path.join(dataset_path, 'train', species)
                print(f"  Checking species path: {species_path}")
                print(f"  Absolute species path: {os.path.abspath(species_path)}")
                if os.path.exists(species_path):
                    print(f"  Found {species} directory")
                    img_files = [f for f in os.listdir(species_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    print(f"  Found {len(img_files)} image files")
                    for img_file in img_files:
                        img_path = os.path.join(species_path, img_file)
                        features = self.extract_features(img_path)
                        if features is not None:
                            X.append(features)
                            y.append(species)
                        else:
                            print(f"    Failed to extract features from {img_file}")
                else:
                    print(f"  Species directory not found: {species}")

        self.class_names = sorted(list(set(y)))
        print(f"\nLoaded {len(X)} samples from {len(self.class_names)} classes")
        print(f"Classes: {self.class_names}")

        return np.array(X), np.array(y)

    def train_model(self, dataset_path):
        """Train the classification model"""
        print("Training model...")

        X, y = self.load_dataset(dataset_path)

        if len(X) == 0:
            raise ValueError("No training data found!")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(".2f")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        return accuracy

    def predict(self, image_path):
        """Predict species from image"""
        if self.model is None:
            raise ValueError("Model not trained or loaded!")

        features = self.extract_features(image_path)
        if features is None:
            raise ValueError("Could not extract features from image")

        # Get prediction probabilities
        probabilities = self.model.predict_proba([features])[0]

        # Get top prediction
        predicted_idx = np.argmax(probabilities)
        predicted_species = self.class_names[predicted_idx]
        confidence = probabilities[predicted_idx] * 100

        # Get category
        category = "Unknown"
        for cat, species_list in self.categories.items():
            if predicted_species in species_list:
                category = cat
                break

        # Get morphological description
        morphology = self.get_morphology_description(predicted_species)

        return {
            'species': predicted_species,
            'category': category,
            'confidence': round(confidence, 2),
            'morphology': morphology,
            'all_probabilities': {
                self.class_names[i]: round(probabilities[i] * 100, 2)
                for i in range(len(self.class_names))
            }
        }

    def get_morphology_description(self, species):
        """Get morphological description for species"""
        descriptions = {
            'Amoeba': 'Single-celled protist with irregular shape, moves by pseudopodia. No fixed shape, amorphous appearance.',
            'Euglena': 'Single-celled protist with spindle shape, has flagella for movement and chloroplasts for photosynthesis.',
            'Paramecium': 'Ciliated protist with elongated oval shape, covered with cilia for movement. Has oral groove and contractile vacuoles.',
            'Hydra': 'Small freshwater cnidarian with cylindrical body, tentacles for capturing prey. Shows radial symmetry.',
            'Rod_bacteria': 'Bacterial cells with elongated rod-shaped morphology. Gram-positive or negative staining possible.',
            'Spherical_bacteria': 'Bacterial cells with spherical (cocci) morphology. May appear in chains, clusters, or pairs.',
            'Spiral_bacteria': 'Bacterial cells with spiral or helical morphology. Includes spirilla and spirochetes.',
            'Yeast': 'Single-celled fungi with oval to spherical shape. Reproduce by budding. Important in fermentation.'
        }
        return descriptions.get(species, 'Morphological characteristics not available.')

    def save_model(self, output_path):
        """Save trained model and metadata"""
        if self.model is None:
            raise ValueError("No model to save!")

        os.makedirs(output_path, exist_ok=True)

        # Save model
        model_path = os.path.join(output_path, 'microscopic_classifier.joblib')
        joblib.dump(self.model, model_path)

        # Save metadata
        metadata = {
            'class_names': self.class_names,
            'categories': self.categories,
            'model_type': 'RandomForestClassifier',
            'feature_count': 20  # Based on our feature extraction
        }

        with open(os.path.join(output_path, 'model_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Model saved to {output_path}")

    def load_model(self, model_path):
        """Load trained model and metadata"""
        model_file = os.path.join(model_path, 'microscopic_classifier.joblib')
        metadata_file = os.path.join(model_path, 'model_metadata.json')

        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")

        self.model = joblib.load(model_file)

        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                self.class_names = metadata.get('class_names', [])
                self.categories = metadata.get('categories', {})

        print("Model loaded successfully")

if __name__ == "__main__":
    # Initialize classifier
    classifier = MicroscopicClassifier()

    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script directory: {script_dir}")

    # Try different dataset paths
    possible_paths = [
        "dataset",  # relative to script
        os.path.join(script_dir, "dataset"),  # absolute path to dataset in same dir
        os.path.join(script_dir, "..", "dataset"),  # parent directory dataset
    ]

    dataset_found = False
    for dataset_path in possible_paths:
        print(f"Trying dataset path: {dataset_path}")
        if os.path.exists(dataset_path):
            print(f"Found dataset at: {dataset_path}")
            try:
                accuracy = classifier.train_model(dataset_path)
                classifier.save_model("outputs")
                print("Training completed!")
                dataset_found = True
                break
            except ValueError as e:
                print(f"Error with {dataset_path}: {e}")
                continue
        else:
            print(f"Dataset not found at: {dataset_path}")

    if not dataset_found:
        print("No valid dataset found. Please ensure dataset folder exists with training images.")
        print("Expected structure:")
        print("dataset/")
        print("  train/")
        print("    Amoeba/")
        print("    Euglena/")
        print("    ... (other species folders)")