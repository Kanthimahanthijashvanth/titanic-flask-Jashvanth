from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained Titanic model
model = joblib.load("titanic_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    pclass = int(request.form["pclass"])
    sex = int(request.form["sex"])
    age = float(request.form["age"])
    sibsp = int(request.form["sibsp"])
    parch = int(request.form["parch"])
    fare = float(request.form["fare"])
    embarked = int(request.form["embarked"])

    # Create input array for model
    input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

    # Predict
    prediction = model.predict(input_data)[0]
    result = "Survived ✅" if prediction == 1 else "Did not survive ❌"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
