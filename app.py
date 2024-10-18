from flask import Flask, request, jsonify
from flask_cors import CORS 
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('model.keras')

app = Flask(__name__)
CORS(app) 

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    print("data", data)
    print("data['features']", data['features'])

    try:
        features = np.array(data['features'], dtype=float).reshape(1, -1)
    except ValueError as e:
        return jsonify({'error': f'Invalid input format: {e}'}), 400

    print("Features shape:", features.shape)

    try:
        prediction = model.predict(features)
        print("Prediction:", prediction)

        output_array = [0] * len(prediction[0])
        for index, elem in enumerate(prediction[0]):
            if elem > 0.5:
                output_array[index] = 1
        if not any(output_array):
            predicted_index = np.argmax(prediction)
            output_array[predicted_index] = 1  

    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({'error': str(e)}), 500

    return jsonify({'output_array': output_array})

if __name__ == '__main__':
    app.run(debug=True)
