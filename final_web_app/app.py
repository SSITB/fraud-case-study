from flask import Flask, render_template
from flask_bootstrap import Bootstrap
import pickle
from mongo import update_db, pull_values
app = Flask(__name__)
Bootstrap(app)

@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")


@app.route('/score', methods=['POST'])
def score():
   try:
      with open('model.pkl', 'rb') as f:
          model = pickle.load(f)
      update_db(model)
   except:
      pass

   low, medium, high = pull_values()

   return render_template("score.html", low=low, medium=medium, high=high)


if __name__ == '__main__':

   app.run(host='0.0.0.0', port=8080, debug=True)


