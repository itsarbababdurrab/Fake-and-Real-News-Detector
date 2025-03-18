from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['news']
    transformed_text = vectorizer.transform([text])
    prediction = model.predict(transformed_text)[0]
    result = "Real News" if prediction == 1 else "Fake News"
    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)