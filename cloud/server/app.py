# app.py
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)   # Allow all origins (for development)

# Load the model bundle once at startup
with open('model.pkl', 'rb') as f:
    bundle = pickle.load(f)

model = bundle['model']
label_encoder = bundle['label_encoder']
feature_names = bundle['feature_names']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get JSON input
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Required fields
        required_fields = ['cpu_peak', 'cpu_avg', 'variance', 'pattern',
                           'duration_hours', 'instance_count']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # 2. Create DataFrame from input (single row)
        input_df = pd.DataFrame([data])

        # 3. One‑hot encode categorical columns (variance and pattern)
        input_encoded = pd.get_dummies(input_df)

        # 4. Reindex to match training feature set (fill missing columns with 0)
        input_encoded = input_encoded.reindex(columns=feature_names, fill_value=0)

        # 5. Predict class and probability
        pred_encoded = model.predict(input_encoded)[0]
        pred_proba = model.predict_proba(input_encoded)[0]

        # Confidence for the predicted class (max probability)
        confidence = float(max(pred_proba))

        # Decode class label
        strategy = label_encoder.inverse_transform([pred_encoded])[0]

        # 6. Return response
        return jsonify({
            'predicted_strategy': strategy,
            'confidence': confidence,
            # Optionally include all class probabilities
            'probabilities': {
                label_encoder.inverse_transform([i])[0]: float(prob)
                for i, prob in enumerate(pred_proba)
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
