from flask import Flask, request, jsonify, render_template
import os
import yaml
import numpy as np
import joblib

param_path = 'params.yaml'
webapp_root = 'webapp'

static_dir = os.path.join(webapp_root, 'static')
template_dir = os.path.join(webapp_root, 'templates')

app = Flask(__name__ , static_folder= static_dir, template_folder= template_dir)

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def prediction(data):
    config = read_params(param_path)
    model_dir = config["webapp_model_dir"]
    model = joblib.load(model_dir)
    pred = model.predict(data)
    # print(pred)
    return pred[0]

def api_responce(api_input):
    try:
        data = np.array([list(api_input.json.values())])
        response = prediction(data)
        response = {"response": response}
        return response
    except Exception as e:
        print(e)
        error = {"error": "Something went wrong! Please try again."}
        return error


@app.route("/", methods= ['GET', 'POST'])
def index():
    if request.method == "POST":
        try:
            if request.form:
                data = dict(request.form).values()
                data = [list(map(float, data))]
                response = prediction(data)
                return render_template("index.html", response= response)
            elif request.json:
                responce = api_responce(request)
                return jsonify(responce)
        except Exception as e:
            print(e)
            error ={"error": "Something went wrong! Please, Try again."}
            return render_template('404.html', error = error)
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)