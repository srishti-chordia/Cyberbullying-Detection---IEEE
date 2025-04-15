from flask import Flask, request, jsonify
from model_utils.predictor import load_model_and_tokenizer, predict_texts

app = Flask(__name__)
model, tokenizer = load_model_and_tokenizer("model.h5")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    texts = data.get("texts", [])
    predictions = predict_texts(model, tokenizer, texts)
    return jsonify(predictions)


if __name__ == "__main__":
    app.run(debug=True)
