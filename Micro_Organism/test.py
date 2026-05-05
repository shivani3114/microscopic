import os
import numpy as np
import joblib
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from model_trainer import MicroscopicClassifier


def resolve_path(*parts):
    return os.path.abspath(os.path.join(*parts))


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Model and dataset paths
    model_dir = resolve_path(script_dir, 'outputs')
    model_file = resolve_path(model_dir, 'microscopic_classifier.joblib')
    metadata_file = resolve_path(model_dir, 'model_metadata.json')
    test_dataset_dir = resolve_path(script_dir, 'dataset', 'test')

    print('Script Directory :', script_dir)
    print('Model Directory  :', model_dir)
    print('Model File       :', model_file)
    print('Metadata File    :', metadata_file)
    print('Test Dataset Dir :', test_dataset_dir)

    if not os.path.exists(model_file):
        print('ERROR: Model not found.')
        print('Please run model_trainer.py to train and save the model to outputs/.')
        return
    if not os.path.exists(test_dataset_dir):
        print('ERROR: Test dataset not found.')
        print('Expected path:', test_dataset_dir)
        return

    # Load classifier with metadata
    classifier = MicroscopicClassifier()
    classifier.load_model(model_dir)

    y_true = []
    y_pred = []
    errors = []

    species_dirs = [d for d in os.listdir(test_dataset_dir) 
                    if os.path.isdir(os.path.join(test_dataset_dir, d))]

    if not species_dirs:
        raise ValueError(f"No species directories found inside {test_dataset_dir}")

    for species in sorted(species_dirs):
        species_path = os.path.join(test_dataset_dir, species)
        image_files = [f for f in os.listdir(species_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

        for image_file in image_files:
            image_path = os.path.join(species_path, image_file)
            features = classifier.extract_features(image_path)
            if features is None:
                errors.append((image_path, 'feature extraction failed'))
                continue

            # predict with loaded model
            try:
                proba = classifier.model.predict_proba([features])[0]
                idx = np.argmax(proba)
                predicted = classifier.class_names[idx]
            except Exception as e:
                errors.append((image_path, str(e)))
                continue

            y_true.append(species)
            y_pred.append(predicted)

    if not y_true:
        print('ERROR: No test samples were processed. Check your test dataset and categories.')
        print('Species folders found:', species_dirs)
        return

    acc = accuracy_score(y_true, y_pred)

    print('=' * 80)
    print('Microscopic Classifier Test Report')
    print(f'Test samples evaluated: {len(y_true)}')
    print(f'Correct predictions   : {sum(1 for truth, pred in zip(y_true, y_pred) if truth == pred)}')
    print(f'Accuracy              : {acc * 100:.2f}%')
    print('=' * 80)

    print('Classification report:')
    print(classification_report(y_true, y_pred, zero_division=0))

    print('Confusion matrix:')
    print(confusion_matrix(y_true, y_pred, labels=sorted(set(y_true))))

    if errors:
        print('\nErrors encountered:')
        for path, err in errors:
            print(f' - {path}: {err}')

    print('\nIf accuracy is unexpectedly low:')
    print(' - Verify that test dataset structure is dataset/test/<species>/*.jpg')
    print(' - Check that species names match the classifier classes exactly')


if __name__ == '__main__':
    main()
