from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

def predict_sequence(sequence, num_predictions=5):
    sequence = np.array(sequence)
    x = np.arange(len(sequence)).reshape(-1, 1)
    y = sequence

    poly = PolynomialFeatures(degree=2)
    x_poly = poly.fit_transform(x)
    model = LinearRegression()
    model.fit(x_poly, y)

    future_x = np.arange(len(sequence), len(sequence) + num_predictions).reshape(-1, 1)
    future_x_poly = poly.transform(future_x)
    predictions = model.predict(future_x_poly)
    return predictions.tolist()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sequence = data.get('sequence', [])

    if not sequence or len(sequence) < 2:
        return jsonify({"error": "Введите последовательность из как минимум двух чисел."}), 400

    try:
        sequence = list(map(float, sequence))
        predictions = predict_sequence(sequence)
        return jsonify({"input": sequence, "predictions": predictions})
    except Exception as e:
        return jsonify({"error": f"Ошибка обработки: {str(e)}"}), 400

if __name__ == '__main__':
    app.run(debug=True)
