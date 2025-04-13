import os
import pickle
import numpy as np
import warnings
import time
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

# Model paths
MODEL_DIR = './models_labeled'
MODEL_MAPPING = {
    "all": {
        "model": os.path.join(MODEL_DIR, "bilstm_model_all_features.keras"),
        "scaler": os.path.join(MODEL_DIR, "scaler_all_features.pkl"),
        "features": os.path.join(MODEL_DIR, "features_all_features.pkl")
    },
    "kda": {
        "model": os.path.join(MODEL_DIR, "bilstm_model_hero_kda.keras"),
        "scaler": os.path.join(MODEL_DIR, "scaler_hero_kda.pkl"),
        "features": os.path.join(MODEL_DIR, "features_hero_kda.pkl")
    },
    "hero_picks": {
        "model": os.path.join(MODEL_DIR, "bilstm_model_hero_only.keras"),
        "scaler": os.path.join(MODEL_DIR, "scaler_hero_only.pkl"),
        "features": os.path.join(MODEL_DIR, "features_hero_only.pkl")
    }
}

# Caching loaded models/scalers/features
loaded_models = {}
loaded_scalers = {}
loaded_features = {}

def load_model_and_scaler(model_name):
    if model_name in loaded_models:
        return loaded_models[model_name], loaded_scalers[model_name], loaded_features[model_name]

    if model_name not in MODEL_MAPPING:
        raise ValueError(f"Model '{model_name}' not found")

    paths = MODEL_MAPPING[model_name]
    if not all(os.path.exists(p) for p in paths.values()):
        raise FileNotFoundError("Model, scaler, or feature list missing")

    model = load_model(paths['model'])
    with open(paths['scaler'], 'rb') as f:
        scaler = pickle.load(f)
    with open(paths['features'], 'rb') as f:
        feature_list = pickle.load(f)

    loaded_models[model_name] = model
    loaded_scalers[model_name] = scaler
    loaded_features[model_name] = feature_list

    return model, scaler, feature_list

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print("Incoming POST data:", data)

    if not data or 'features' not in data or 'model_name' not in data:
        return jsonify({'error': "Missing 'features' or 'model_name' in request"}), 400

    try:
        model_name = data['model_name']
        model, scaler, feature_order = load_model_and_scaler(model_name)

        raw_features = data['features']
        if isinstance(raw_features, list):
            if len(raw_features) != len(feature_order):
                return jsonify({'error': f"Expected {len(feature_order)} features but received {len(raw_features)}."}), 400
            input_dict = dict(zip(feature_order, raw_features))
        elif isinstance(raw_features, dict):
            input_dict = raw_features
        else:
            return jsonify({'error': "Invalid type for 'features'. Must be list or dictionary."}), 400

        missing = [f for f in feature_order if f not in input_dict]
        if missing:
            return jsonify({'error': f"Missing required features: {missing}"}), 400

        ordered_values = [input_dict[f] for f in feature_order]
        feature_mapping = dict(zip(feature_order, ordered_values))

        ordered_array = np.array(ordered_values).reshape(1, -1)
        scaled_input = scaler.transform(ordered_array)
        lstm_input = scaled_input.reshape(1, 1, scaled_input.shape[1])

        prob = model.predict(lstm_input)[0][0]
        label = 'Radiant' if prob >= 0.5 else 'Dire'

        time.sleep(2.5)

        return jsonify({
            'prediction': label,
            'model_used': model_name,
            'feature_mapping': feature_mapping
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7860)