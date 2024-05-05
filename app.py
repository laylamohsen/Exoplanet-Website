from flask import Flask, render_template, request
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd

model = load_model('Finale_Exoplanet_Model.h5')
# model = pickle.load(open('finale_model.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['Not_Transit-like_False_Positive_Flag']
    data2 = request.form['Orbit_Period']
    data3 = request.form['Transit_Epoch']
    data4 = request.form['Transit_Duration']
    data5 = request.form['Transit_Depth']
    data6 = request.form['Planetary_Radius']
    data7 = request.form['Insolation_Flux']
    data8 = request.form['Transit_Signal-to-Noise']
    data9 = request.form['Stellar_Surface_Gravity']
    data10 = request.form['Stellar_Radius']

    arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8, data9, data10]]).astype(float)
    arr[0] = preprocess_input(arr[0])
    pred = model.predict(arr)
    print(pred)
    if (pred < 0.5):
        pred = 0
    else:
        pred = 1

    return render_template('after.html', data=pred)

# def standardScaler(input):
    

#     # return input


def preprocess_input(input):
    df = pd.read_csv('NoScaling_NasaExoplanetArchive.csv')
    min = df['Not_Transit-like_False_Positive_Flag'].min()
    max = df['Not_Transit-like_False_Positive_Flag'].max()
    input[0] = (input[0]-min)/(max-min)

    min = df['Orbit_Period'].min()
    max = df['Orbit_Period'].max()
    input[1] = (input[1]-min)/(max-min)

    min = df['Transit_Epoch'].min()
    max = df['Transit_Epoch'].max()
    input[2] = (input[2]-min)/(max-min)

    min = df['Transit_Duration'].min()
    max = df['Transit_Duration'].max()
    input[3] = (input[3]-min)/(max-min)

    min = df['Transit_Depth'].min()
    max = df['Transit_Depth'].max()
    input[4] = (input[4]-min)/(max-min)

    min = df['Planetary_Radius'].min()
    max = df['Planetary_Radius'].max()
    input[5] = (input[5]-min)/(max-min)

    min = df['Insolation_Flux'].min()
    max = df['Insolation_Flux'].max()
    input[6] = (input[6]-min)/(max-min)

    min = df['Transit_Signal-to-Noise'].min()
    max = df['Transit_Signal-to-Noise'].max()
    input[7] = (input[7]-min)/(max-min)

    min = df['Stellar_Surface_Gravity'].min()
    max = df['Stellar_Surface_Gravity'].max()
    input[8] = (input[8]-min)/(max-min)

    min = df['Stellar_Radius'].min()
    max = df['Stellar_Radius'].max()
    input[9] = (input[9]-min)/(max-min)


    # mean = df['Not_Transit-like_False_Positive_Flag'].mean()
    # std = df['Not_Transit-like_False_Positive_Flag'].std()
    # input[0] = (input[0]-mean)/std

    # mean = df['Orbit_Period'].mean()
    # std = df['Orbit_Period'].std()
    # input[1] = (input[1]-mean)/std

    # mean = df['Transit_Epoch'].mean()
    # std = df['Transit_Epoch'].std()
    # input[2] = (input[2]-mean)/std

    # mean = df['Transit_Duration'].mean()
    # std = df['Transit_Duration'].std()
    # input[3] = (input[3]-mean)/std

    # mean = df['Transit_Depth'].mean()
    # std = df['Transit_Depth'].std()
    # input[4] = (input[4]-mean)/std

    # mean = df['Planetary_Radius'].mean()
    # std = df['Planetary_Radius'].std()
    # input[5] = (input[5]-mean)/std

    # mean = df['Insolation_Flux'].mean()
    # std = df['Insolation_Flux'].std()
    # input[6] = (input[6]-mean)/std

    # mean = df['Transit_Signal-to-Noise'].mean()
    # std = df['Transit_Signal-to-Noise'].std()
    # input[7] = (input[7]-mean)/std

    # mean = df['Stellar_Surface_Gravity'].mean()
    # std = df['Stellar_Surface_Gravity'].std()
    # input[8] = (input[8]-mean)/std

    # mean = df['Stellar_Radius'].mean()
    # std = df['Stellar_Radius'].std()
    # input[9] = (input[9]-mean)/std
    print(input)

    return input

if __name__ == "__main__":
    app.run(debug=True)















