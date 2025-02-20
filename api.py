from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from PIL import Image
import io
from resnet_model import ResNetModel
from data_loader import get_transforms
from config import Config

app = Flask(__name__)
CORS(app)

# Load the model
model = ResNetModel().to(Config.DEVICE)
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

transform = get_transforms()


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    try:
        # Read and transform image
        image_file = request.files['image']
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(Config.DEVICE)

        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item() * 100

        result = {
            'prediction': 'fake' if prediction == 0 else 'real',
            'confidence': round(confidence, 2)
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000)