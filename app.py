import os
import re
import json
import joblib
import warnings
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Suppress sklearn version warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

app = Flask(__name__)

# Load models and preprocessing objects
BASE_DIR = "saved_ml_models"
MODELS_DIR = "all_models"

# Load TF-IDF and Label Encoder
tfidf = joblib.load(f"{BASE_DIR}/tfidf_vectorizer.joblib")
label_encoder = joblib.load(f"{BASE_DIR}/label_encoder.joblib")

# Load metadata
with open(f"{BASE_DIR}/metadata.json", "r") as f:
    metadata = json.load(f)

# Load all models with error handling for version incompatibility
models = {}
all_model_names = ["Logistic Regression", "SVM", "Naive Bayes", "Decision Tree", "Random Forest"]
failed_models = []

for model_name in all_model_names:
    safe_name = model_name.replace(" ", "_").lower()
    model_path = f"{MODELS_DIR}/{safe_name}.joblib"
    if os.path.exists(model_path):
        try:
            models[model_name] = joblib.load(model_path)
            print(f"[OK] Successfully loaded: {model_name}")
        except (ValueError, AttributeError) as e:
            print(f"[WARNING] Could not load {model_name} due to version incompatibility")
            print(f"  Error: {str(e)[:100]}...")
            failed_models.append(model_name)
            continue
    else:
        print(f"[WARNING] Model file not found: {model_path}")
        failed_models.append(model_name)

# Keep all model names for display, but track which are available
available_models = [name for name in all_model_names if name in models]
print(f"\n[OK] Successfully loaded {len(models)} models: {', '.join(available_models)}")
if failed_models:
    print(f"[WARNING] Failed to load {len(failed_models)} models: {', '.join(failed_models)}")
    if failed_models:
        print("  (Due to file not found or version incompatibility)")
print(f"[OK] Best model available: {metadata['best_model']}\n")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Routes
@app.route('/')
def index():
    return render_template('index.html', 
                         best_model=metadata['best_model'],
                         best_accuracy=round(metadata['accuracy'] * 100, 2),
                         total_models=len(metadata['all_results']))

@app.route('/prediction')
def prediction():
    # Create model info with availability status
    model_info = []
    for model_name in all_model_names:
        model_info.append({
            'name': model_name,
            'available': model_name in models
        })
    return render_template('prediction.html', 
                         models=all_model_names,
                         model_info=model_info,
                         available_models=available_models,
                         failed_models=failed_models,
                         best_model=metadata['best_model'])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data.get('text', '')
        model_name = data.get('model', metadata['best_model'])
        
        if not text:
            return jsonify({'error': 'Please enter some text'}), 400
        
        if model_name not in models:
            if model_name in failed_models:
                return jsonify({
                    'error': f'{model_name} is currently unavailable due to scikit-learn version incompatibility. Please select another model like {metadata["best_model"]} (recommended).'
                }), 400
            return jsonify({'error': 'Invalid model selected'}), 400
        
        # Clean and vectorize text
        cleaned_text = clean_text(text)
        X = tfidf.transform([cleaned_text])
        
        # Get prediction
        model = models[model_name]
        prediction = model.predict(X)[0]
        sentiment = label_encoder.inverse_transform([prediction])[0]
        
        # Get probability if available
        confidence = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
            confidence = round(float(max(proba)) * 100, 2)
        elif hasattr(model, 'decision_function'):
            # For SVM
            decision = model.decision_function(X)[0]
            # Normalize decision scores to pseudo-probabilities
            exp_scores = np.exp(decision - np.max(decision))
            proba = exp_scores / exp_scores.sum()
            confidence = round(float(max(proba)) * 100, 2)
        
        return jsonify({
            'sentiment': sentiment,
            'confidence': confidence,
            'model_used': model_name,
            'cleaned_text': cleaned_text
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/documentation')
def documentation():
    # Load classification reports
    reports_data = {}
    for model_name in all_model_names:
        safe_name = model_name.replace(" ", "_")
        csv_path = f"{BASE_DIR}/reports/{safe_name}_classification_report.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            reports_data[model_name] = df.to_dict('records')
    
    return render_template('documentation.html',
                         metadata=metadata,
                         reports=reports_data,
                         model_names=all_model_names)

@app.route('/get_model_accuracy')
def get_model_accuracy():
    return jsonify(metadata['all_results'])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

