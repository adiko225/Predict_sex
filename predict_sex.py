#Libraries

import pandas as pd
import string
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.optimizers import Adam
import pickle 


#Data importation

names=pd.read_csv(r"/mnt/c/Users/7MAKSACOD PC/Downloads/names_data.csv")

#names.head()

#Preprocessing

names["name"]=names["name"].str.lower()
data=names.iloc[:, :2]

## Binary encoder function

def encod_sex(ch):
  if ch=="F":
    return 0
  else:
    return 1

data["gender"]= data["gender"].apply(encod_sex)

## Encoder alphabet character

alphabet=list(string.ascii_lowercase)

def encod_names(name):
  name_split=list(name)
  name_encoded=[alphabet.index(name_split[i]) for i in range(len(name_split))]
  if len(name_encoded)<50:
      for i in range(50-len(name_encoded)):
        name_encoded+=[0]
  return name_encoded

data["name_encoded"]=data["name"].apply(encod_names)

#print(data.head())
## Entrainement du modele

def lstm_model(num_alphabets=27, name_length=50, embedding_dim=256):
    model = Sequential([
        Embedding(num_alphabets, embedding_dim, input_length=name_length),
        Bidirectional(LSTM(units=128, recurrent_dropout=0.2, dropout=0.2)),
        Dense(1, activation="sigmoid")
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])

    return model

# Step 1: Instantiate the model
model = lstm_model(num_alphabets=27, name_length=50, embedding_dim=256)

# Step 2: Split Training and Test Data
X = np.asarray(data['name_encoded'].values.tolist())
y = np.asarray(data['gender'].values.tolist())

X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=42)


# Step 4: Training of model

callbacks = [
    EarlyStopping(monitor='val_accuracy',
                  min_delta=1e-3,
                  patience=5,
                  mode='max',
                  restore_best_weights=True,
                  verbose=1),
]

history = model.fit(x=X_train,
                    y=y_train,
                    batch_size=64,
                    epochs=50,
                    validation_data=(X_test, y_test),
                    callbacks=callbacks)


pickle.dump(model, open('model.pkl', 'wb'))












