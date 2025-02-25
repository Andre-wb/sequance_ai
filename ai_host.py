from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

def detect_pattern(sequence):
    """Определяет закономерность в последовательности."""
    diffs = np.diff(sequence)

    if np.all(diffs == diffs[0]):  # Арифметическая прогрессия (сложение/вычитание)
        return "addition"

    if np.all(diffs[1:] / diffs[:-1] == diffs[1] / diffs[0]):  # Геометрическая прогрессия (умножение/деление)
        return "multiplication"

    if np.all(np.sqrt(sequence) == np.round(np.sqrt(sequence))):  # Последовательность квадратов
        return "squares"

    return None  # Неизвестный паттерн

def predict_sequence(sequence, num_predictions=5):
    pattern = detect_pattern(sequence)
    sequence = np.array(sequence)

    if pattern == "addition":  # Арифметическая прогрессия
        step = sequence[1] - sequence[0]
        predictions = [sequence[-1] + step * (i + 1) for i in range(num_predictions)]
    elif pattern == "multiplication":  # Геометрическая прогрессия
        ratio = sequence[1] / sequence[0]
        predictions = [sequence[-1] * (ratio ** (i + 1)) for i in range(num_predictions)]
    elif pattern == "squares":  # Квадраты чисел
        n = int(np.sqrt(sequence[-1]))
        predictions = [(n + i + 1) ** 2 for i in range(num_predictions)]
    else:
        raise ValueError("Неизвестная последовательность. Поддерживаются только сложение, умножение и квадраты чисел.")

    return predictions

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
