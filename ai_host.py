from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

def detect_pattern(sequence):
    """Определяет закономерность в последовательности."""
    diffs = np.diff(sequence)

    if np.all(diffs == diffs[0]):  # Арифметическая прогрессия
        return "addition"

    if np.all(diffs[1:] / diffs[:-1] == diffs[1] / diffs[0]):  # Геометрическая прогрессия
        return "multiplication"

    if np.all(np.sqrt(sequence) == np.round(np.sqrt(sequence))):  # Последовательность квадратов
        return "squares"

    return "unknown"  # Неизвестный паттерн, но без ошибки

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
        # Если паттерн не распознан, просто продолжаем последовательность разницей последних двух чисел
        step = sequence[-1] - sequence[-2]
        predictions = [sequence[-1] + step * (i + 1) for i in range(num_predictions)]

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
        return jsonify({"error": f"Что-то пошло не так, но нейросеть продолжает работать!", "details": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
