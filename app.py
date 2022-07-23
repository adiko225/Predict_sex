from cProfile import run
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn import preprocessing
import pickle
import string

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/predict',methods=['POST'])




def encod_names(name):
  alphabet=list(string.ascii_lowercase)
  name_split=list(name)
  name_encoded=[alphabet.index(name_split[i]) for i in range(len(name_split))]
  if len(name_encoded)<50:
    for i in range(50-len(name_encoded)):
      name_encoded+=[0]
  return name_encoded

def predict():
    feature_list=request.form.to_dict()
    feature_list = list(feature_list.values())
    feature_list = list(map(chr, feature_list))
    feature_list = [encod_names(feature_list[i]) for i in range(len(feature_list))]
    final_feature=np.array(feature_list)
    prediction= model.predict(final_feature)
    output=prediction[0]
    if output==1:
        text="Male"
    else:
        text="Female"
    return render_template('index.html', prediction_text='You are a {}'.format(text))

if __name__=="__main__":
    app.run(debug=True)
