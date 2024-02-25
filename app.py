from flask import Flask, render_template, request
import model_loader
import requests
from io import BytesIO

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' in request.files:
        image = request.files['image']
        prediction = model_loader.predict(image)
    elif 'image_url' in request.form:
        image_url = request.form['image_url']
        response = requests.get(image_url)
        image = BytesIO(response.content)
        prediction = model_loader.predict(image)
    else:
        return "No image provided", 400
    return str(prediction)

if __name__ == '__main__':
    app.run(debug=True)
