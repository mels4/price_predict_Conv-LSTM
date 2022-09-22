from urllib import request
from flask import Flask, render_template, jsonify, request

app = Flask(__name__, template_folder='template')

@app.route('/', methods=['GET', 'POST'])
def predict_data():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")