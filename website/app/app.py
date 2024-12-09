from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/data-preprocessing")
def data_preprocessing():
    return render_template("data_preprocessing.html")

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        # Handle image upload and model prediction logic
        prediction = "organic"  # Replace with model inference
        confidence = 0.95  # Replace with actual confidence score
        return render_template("prediction.html", prediction=prediction, confidence=confidence)
    return render_template("prediction.html")

@app.route("/retrain", methods=["GET", "POST"])
def retrain():
    if request.method == "POST":
        # Handle model retraining logic
        return render_template("retrain.html", retrained=True)
    return render_template("retrain.html")

@app.route("/evaluate", methods=["POST"])
def evaluate():
    # Handle evaluation logic
    evaluation = {"prediction": "recyclable", "confidence": 0.85}
    return render_template("retrain.html", evaluation=evaluation)

if __name__ == "__main__":
    app.run(debug=True)
