from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model/hate_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    return text

@app.route("/", methods=["GET","POST"])
def index():
    prediction = ""

    if request.method == "POST":
        tweet = request.form["tweet"]

        tweet = clean_text(tweet)
        vector = vectorizer.transform([tweet])

        result = model.predict(vector)

        if result[0] == 0:
            prediction = "Hate Speech"
        elif result[0] == 1:
            prediction = "Offensive Language"
        else:
            prediction = "Neither"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)