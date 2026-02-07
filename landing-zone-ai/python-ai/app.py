from flask import Flask, request, jsonify
from services.inference import InferenceService
from config import Config

app = Flask(__name__)
inference_service = InferenceService()

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    # Save temp file or process in memory
    # ...
    
    result = inference_service.predict(file)
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
