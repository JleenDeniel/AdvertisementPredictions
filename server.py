from data_preprocessing import model
from flask import Flask

app = Flask(__name__)


@app.route('/')
def index():
    return 'To make a prediction send request to /predict/<bid>/<placement> like: predict/0.5/2'


@app.route('/predict/<bid>/<placement>')
def load_prediction(bid, placement):
    result = model.predict([[bid, placement]])[0]
    if result < 0:
        return "Too little, try another arguments to see a prediction"
    else:
        return str(round(model.predict([[bid, placement]])[0]))


if __name__ == "__main__":
    app.run()
