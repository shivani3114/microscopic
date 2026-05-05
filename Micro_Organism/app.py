from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import json
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import joblib
from pathlib import Path
import base64
from io import BytesIO
from pathlib import Path
import base64 
from io import BytesIO

app = Flask(__name__)

# Ensure we always use the app file directory as the base path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
UPLOAD_DIR = os.path.join(STATIC_DIR, 'uploads')

# Configuration
app.static_folder = STATIC_DIR
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class MicroscopicPredictor:
    def __init__(self):
        self.model = None
        self.class_names = []
        self.categories = {}
        self.load_model()

    def load_model(self):
        """Load the trained model"""
        try:
            model_path = 'outputs/microscopic_classifier.joblib'
            metadata_path = 'outputs/model_metadata.json'

            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                print("Model loaded successfully")

                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        self.class_names = metadata.get('class_names', [])
                        self.categories = metadata.get('categories', {})
            else:
                print("Warning: Model not found. Please train the model first.")

        except Exception as e:
            print(f"Error loading model: {e}")

    def extract_features(self, image):
        """Extract features from PIL Image object"""
        try:
            # Convert to grayscale if needed
            if image.mode != 'L':
                image = image.convert('L')

            # Resize
            image = image.resize((64, 64))

            # Convert to numpy array
            img_array = np.array(image)

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
            print(f"Error extracting features: {e}")
            return None

    def predict(self, image):
        """Predict species from image"""
        if self.model is None:
            return {
                'error': 'Model not loaded. Please train the model first.',
                'success': False
            }

        features = self.extract_features(image)
        if features is None:
            return {
                'error': 'Could not process image.',
                'success': False
            }

        try:
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
                'success': True,
                'species': predicted_species,
                'category': category,
                'confidence': round(confidence, 2),
                'morphology': morphology,
                'all_probabilities': {
                    self.class_names[i]: round(probabilities[i] * 100, 2)
                    for i in range(len(self.class_names))
                }
            }

        except Exception as e:
            return {
                'error': f'Prediction error: {str(e)}',
                'success': False
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

    def generate_sample_grid(self, species, grid_size=(3, 3)):
        """Generate a grid of sample images for the predicted species"""
        try:
            print(f"Generating sample grid for species: {species}")
            # Get the directory where the script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            print(f"Script directory: {script_dir}")

            # Look for sample images in dataset/test/species/
            sample_dir = os.path.join(script_dir, "dataset", "test", species)
            print(f"Looking for samples in: {sample_dir}")

            if not os.path.exists(sample_dir):
                print(f"Sample directory does not exist: {sample_dir}")
                return None

            # Get all image files
            image_files = [f for f in os.listdir(sample_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            print(f"Found {len(image_files)} image files in {sample_dir}")
            if not image_files:
                print(f"No image files found in {sample_dir}")
                return None

            # Select random samples
            num_samples = min(grid_size[0] * grid_size[1], len(image_files))
            if num_samples == 0:
                print("No images available for sample grid")
                return None

            selected_files = np.random.choice(image_files, num_samples, replace=False)
            print(f"Selected {num_samples} sample files: {selected_files[:3]}")

            # Adjust grid size based on available images
            if num_samples < 9:
                if num_samples <= 1:
                    grid_size = (1, 1)
                elif num_samples <= 4:
                    grid_size = (2, 2)
                else:
                    grid_size = (3, 3)  # Keep 3x3 but will have empty spaces

            # Create grid
            grid_width = 300
            grid_height = 300
            cell_width = grid_width // grid_size[1]
            cell_height = grid_height // grid_size[0]

            grid_image = Image.new('RGB', (grid_width, grid_height), 'white')

            for i, img_file in enumerate(selected_files):
                try:
                    img_path = os.path.join(sample_dir, img_file)
                    print(f"Processing sample image: {img_path}")
                    img = Image.open(img_path)

                    # Convert to RGB if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    # Resize to fit cell
                    img.thumbnail((cell_width-4, cell_height-4))

                    # Calculate position
                    row = i // grid_size[1]
                    col = i % grid_size[1]
                    x = col * cell_width + (cell_width - img.width) // 2
                    y = row * cell_height + (cell_height - img.height) // 2

                    grid_image.paste(img, (x, y))
                    print(f"Added image {img_file} to grid at position ({x}, {y})")

                except Exception as e:
                    print(f"Error processing sample {img_file}: {e}")
                    continue

            print("Sample grid generated successfully")
            return grid_image

        except Exception as e:
            print(f"Error generating sample grid: {e}")
            import traceback
            traceback.print_exc()
            return None

# Initialize predictor
predictor = MicroscopicPredictor()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze uploaded image"""
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            })

        file = request.files['image']
        ground_truth = request.form.get('ground_truth', '').strip()

        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No image selected'
            })

        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Please upload PNG, JPG, JPEG, or GIF files.'
            })

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Open and process image
        image = Image.open(filepath)

        # Get prediction
        result = predictor.predict(image)

        if not result['success']:
            return jsonify(result)

        # Generate sample grid
        print(f"Generating sample grid for species: {result['species']}")
        sample_grid = predictor.generate_sample_grid(result['species'])
        sample_image_url = None

        if sample_grid:
            # Save sample grid
            sample_filename = f"sample_{result['species'].lower()}_{np.random.randint(1000)}.png"
            sample_path = os.path.join(app.config['UPLOAD_FOLDER'], sample_filename)
            sample_grid.save(sample_path)
            sample_image_url = f"/static/uploads/{sample_filename}"
            print(f"Sample grid saved to: {sample_path}")
            print(f"Sample image URL: {sample_image_url}")
        else:
            print("Sample grid generation failed - no similar images found")
            # Fallback to show one existing sample image if possible
            species_candidate = result['species']
            for candidate_dir in [
                os.path.join('dataset', 'test', species_candidate),
                os.path.join('dataset', 'train', species_candidate)
            ]:
                if os.path.exists(candidate_dir):
                    candidate_images = [f for f in os.listdir(candidate_dir)
                                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    if candidate_images:
                        first_sample = candidate_images[0]
                        fallback_src = os.path.join(candidate_dir, first_sample)
                        try:
                            fallback_img = Image.open(fallback_src)
                            fallback_filename = f"sample_fallback_{species_candidate.lower()}_{np.random.randint(1000)}.png"
                            fallback_path = os.path.join(app.config['UPLOAD_FOLDER'], fallback_filename)
                            fallback_img.convert('RGB').save(fallback_path)
                            sample_image_url = f"/static/uploads/{fallback_filename}"
                            print(f"Fallback sample image saved to: {fallback_path}")
                            break
                        except Exception as ex:
                            print(f"Unable to save fallback sample image: {ex}")
                            sample_image_url = None

        # Prepare response
        response = {
            'success': True,
            'predicted_species': result['species'],
            'category': result['category'],
            'confidence': result['confidence'],
            'criteria': result['morphology'],
            'image_url': f"/static/uploads/{filename}",
            'sample_image': sample_image_url,
            'all_probabilities': result['all_probabilities']
        }

        # Add ground truth comparison if provided
        if ground_truth:
            is_correct = ground_truth.lower() == result['species'].lower()
            response['ground_truth'] = ground_truth
            response['result'] = 'Correct' if is_correct else 'Incorrect'

        return jsonify(response)

    except Exception as e:
        print(f"Analysis error: {e}")
        return jsonify({
            'success': False,
            'error': f'Analysis failed: {str(e)}'
        })

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting Microscopic Species Identifier...")
    print("Make sure to train the model first using: python model_trainer.py")
    app.run(debug=True, host='0.0.0.0', port=5000)