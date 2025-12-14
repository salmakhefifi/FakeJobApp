from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Charger TF-IDF et modèle
tfidf = joblib.load("tfidf.pkl")
model = joblib.load("model.pkl")

# Route pour afficher le formulaire
@app.route("/")
def home():
    return render_template("index.html")

# Route pour prédiction
@app.route("/predict", methods=["POST"])
def predict():
    company = request.form["company"]
    position = request.form["position"]
    location = request.form["location"]
    requirements = request.form["requirements"]
    benefits = request.form["benefits"]

    text = f"{company} {position} {location} {requirements} {benefits}"

    X = tfidf.transform([text])
    prediction = model.predict(X)[0]

    result = "Fake Job ❌" if prediction == 1 else "Real Job ✅"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
