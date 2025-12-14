from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Charger TF-IDF et modèle
tfidf = joblib.load("tfidf.pkl")
model = joblib.load("model.pkl")

# Route principale pour afficher le formulaire
@app.route("/")
def home():
    return render_template("index.html")

# Route pour prédiction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Récupérer chaque champ du formulaire
        company = request.form["company"]
        position = request.form["position"]
        location = request.form["location"]
        requirements = request.form["requirements"]
        benefits = request.form["benefits"]

        # Concaténer tous les champs en une seule chaîne
        text = f"{company} {position} {location} {requirements} {benefits}"

        # Transformer et prédire
        X = tfidf.transform([text])
        prediction = model.predict(X)[0]

        # Convertir la prédiction en texte
        if prediction == 1:
            result = "Fake Job ❌"
        else:
            result = "Real Job ✅"

        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

# Lancer le serveur Flask
if __name__ == "__main__":
    app.run(debug=True)
