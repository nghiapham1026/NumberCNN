from flask import Flask, render_template, request
import model_loader

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']
    prediction = model_loader.predict(image)
    return str(prediction)

if __name__ == '__main__':
    app.run(debug=True)
