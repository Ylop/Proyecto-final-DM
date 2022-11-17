from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from pysentimiento import create_analyzer
import numpy as np
analyzer = create_analyzer(task="sentiment", lang="en")

app = Flask(__name__)

def filter(i:int):
    if i == 0:
        return "NEGATIVE"
    elif i == 1:
        return "NEUTRAL"
    else:
        return "POSITIVE"

def get_sentimal(prediction):
    # prediction['NEG'] = 0
    # prediction['NEU'] = 1
    # prediction['POS'] = 2
    probs = []
    for key, value in prediction.items():
        probs.append(value)
    probs = np.array(probs, dtype = np.float32)
    max_prob = probs.max()
    max_prob_i = np.argmax(probs)
    return filter(max_prob_i), max_prob


@app.route('/', methods=['GET'])
def index():
   print('Request for index page received')
   return render_template('index.html')

@app.route('/', methods=['POST'])
def hello():
   name = request.form.get('name')

   if name:
       print('Request for hello page received with name=%s' % name)
    #    print(analyzer.predict(name).__dict__['probas'])
       
       name, prob = get_sentimal(analyzer.predict(name).__dict__['probas'])
       return render_template('index.html', name = name, prob = prob)
   else:
       print('Request for hello page received with no name or blank name -- redirecting')
       return redirect(url_for('index'))


if __name__ == '__main__':
   app.run(debug=True)