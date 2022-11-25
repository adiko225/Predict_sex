import tensorflow as tf
import numpy as np
import string


## All the alphabetic charater
alphabet=list(string.ascii_lowercase)

alphabet=alphabet+[" "]

print("the lastest element in alphabet is ", f"this {alphabet[-1]}")

model=tf.keras.models.load_model("predict_name.h5")


def encod_names(name):
    name_split=list(name)
    name_encoded=[alphabet.index(name_split[i])+1 for i in range(len(name_split))]
    return name_encoded

def comp_list(x):
    if len(x)< 50:
        x=[0]*(50-len(x))+x
    return x





def nametonum(name: str):
    name=name.lower()
    name_encoded=encod_names(name)
    name_encoded=comp_list(name_encoded)
    return [name_encoded]


### function to predict

def predict_name(name: str):
    name=np.array(nametonum(name))
    p=model.predict(name)
    if p[0][0]>.3:
        return 'Male', p
    else:
        return 'Female', 1-p