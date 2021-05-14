import os
from flask import Flask, jsonify, request, render_template
import pandas as pd
import json
import pickle
import json
from serializer import serializerJson

HEADERS = {'Content-type': 'application/json', 'Accept': 'text/plain'}


def flask_app():
    app = Flask(__name__)

    @app.route('/', methods=['GET'])
    def server_is_up():
        return render_template('index.html')

    @app.route('/result', methods=['POST'])
    def predict_churn():
        to_form = request.form

        to_predict = serializerJson(to_form)

        prediction = 'Congratulations! The customer will not churn 😀'
        if to_predict[1] == '1':
            prediction = 'Oh no! The customer will churn 😢'
        else:
            prediction = 'Congratulations! The customer will not churn 😀'
        return render_template('result.html', prediction=prediction)
    return app


if __name__ == '__main__':
    app = flask_app()
    app.run(debug=True, host='0.0.0.0')
