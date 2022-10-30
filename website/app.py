import json
from flask import Flask, render_template, jsonify, request, Response, Markup

import datetime
import io
from keras.models import Sequential
from keras.layers import Conv1D, LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from sklearn.model_selection import train_test_split
from plotly.offline import plot
from plotly.graph_objs import Scatter
import plotly
import plotly.express as px

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np

app = Flask(__name__, template_folder='template')

data_pangan = pd.read_csv('data_harga_bahan_pangan_indonesia.csv')
data_pangan.columns = data_pangan.columns.str.replace(' (kg)', '', regex=False).str.replace('(kg)', '',regex=False).str.replace(' ', '_').str.lower()
data_pangan = data_pangan.set_index('date')
window_size_train= 500
window_size_test= 150
window_size_val= 80
batch_size_train=30
batch_size=5
num_shuffle=1000

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_data():
    global types, date, predicti, dates, my_plot, my_plot_2

    if request.method == "POST":
        date = request.form['date']
        types = request.form['bahan']
        print(date)
        predicti = types.lower()

        type = types.lower()
        start_date = datetime.datetime.today()
        end_date = datetime.datetime.strptime(date, '%Y-%m-%d')

        sector_chose = chose_sector(type)

        x_train, x_test = train_test_split(sector_chose, test_size=0.2, random_state=False, shuffle=False)
        x_train, x_val = train_test_split(x_train, test_size=0.1, random_state=False, shuffle=False)

        train = scaler_data(x_train)
        val = scaler_data(x_val)
        test = scaler_data(x_test)
    
        predicti = make_prediction(train, test, val, start_date, end_date)

        date_list = [start_date + datetime.timedelta(days=x) for x in range(day)]
        # dates = pd.DataFrame(predicted, index=date_list, columns=['predict'])
        dates = pd.DataFrame(date_list, columns=['date'])
        dates['predict'] = predicted
        print(dates)

        fig = px.scatter(dates, x='date', y='predict')

        my_plot_2 = json.dumps(fig, cls= plotly.utils.PlotlyJSONEncoder)
        print(my_plot_2)

        return render_template('index.html')
    else:
        return jsonify({"predict": predicti, "jenis_bahan": types, "date": date, "message": "200", "plot": my_plot_2})


def time_step_generator(data, time_size, batch_size, shuffle_data):
    generate_data = tf.data.Dataset.from_tensor_slices(data)
    add_window_time_step = generate_data.window(time_size+1, shift=1, drop_remainder=True)
    flatten_window = add_window_time_step.flat_map(lambda window : window.batch(time_size+1)) 
    tuple_feature_label = flatten_window.map(lambda window: (window[:-1], window[-1])) 
    shuffle_data = tuple_feature_label.shuffle(shuffle_data)
    batch_window = shuffle_data.batch(batch_size).prefetch(1)
    
    return batch_window


def chose_sector(type):
    beras = data_pangan.beras.values
    beras_kualitas_bawah_1 = data_pangan.beras_kualitas_bawah_i.values
    beras_kualitas_bawah_2 = data_pangan.beras_kualitas_bawah_ii.values
    beras_kualitas_medium_1 = data_pangan.beras_kualitas_medium_i.values
    beras_kualitas_medium_2 = data_pangan.beras_kualitas_medium_ii.values
    beras_kualitas_super_1 = data_pangan.beras_kualitas_super_i.values
    beras_kualitas_super_2 = data_pangan.beras_kualitas_super_ii.values
    daging_ayam = data_pangan.daging_ayam.values
    daging_ayam_ras_segar = data_pangan.daging_ayam_ras_segar.values
    daging_sapi = data_pangan.daging_sapi.values
    daging_sapi_kualitas_1 = data_pangan.daging_sapi_kualitas_1.values
    daging_sapi_kualitas_2 = data_pangan.daging_sapi_kualitas_2.values
    telur_ayam = data_pangan.telur_ayam.values
    telur_ayam_ras_segar = data_pangan.telur_ayam_ras_segar.values
    bawang_merah = data_pangan.bawang_merah.values
    bawang_merah_ukuran_sedang = data_pangan.bawang_merah_ukuran_sedang.values
    bawang_putih = data_pangan.bawang_putih.values
    bawang_putih_ukuran_sedang = data_pangan.bawang_putih_ukuran_sedang.values
    cabai_merah = data_pangan.cabai_merah.values
    cabai_merah_besar = data_pangan.cabai_merah_besar.values
    cabai_merah_keriting = data_pangan.cabai_merah_keriting_.values
    cabai_rawit = data_pangan.cabai_rawit.values
    cabai_rawit_hijau = data_pangan.cabai_rawit_hijau.values
    cabai_rawit_merah = data_pangan.cabai_rawit_merah.values
    minyak_goreng = data_pangan.minyak_goreng.values
    minyak_goreng_curah = data_pangan.minyak_goreng_curah.values
    minyak_goreng_kemasan_bermerk_1 = data_pangan.minyak_goreng_kemasan_bermerk_1.values
    minyak_goreng_kemasan_bermerk_2 = data_pangan.minyak_goreng_kemasan_bermerk_2.values
    gula_pasir = data_pangan.gula_pasir.values
    gula_pasir_kualitas_premium = data_pangan.gula_pasir_kualitas_premium.values
    gula_pasir_lokal = data_pangan.gula_pasir_lokal.values
    
    switch={
        "beras": beras,
        "beras kualitas bawah 1": beras_kualitas_bawah_1,
        "beras kualitas bawah 2": beras_kualitas_bawah_2,
        "beras kualitas medium 1": beras_kualitas_medium_1,
        "beras kualitas medium 2": beras_kualitas_medium_2,
        "beras kualitas super 1": beras_kualitas_super_1,
        "beras kualitas super 2": beras_kualitas_super_2,
        "cabai rawit": cabai_rawit,
        "cabai rawit hijau": cabai_rawit_hijau,
        "cabai rawit merah": cabai_rawit_merah,
        "cabai merah" : cabai_merah,
        "cabai merah besar" : cabai_merah_besar,
        "cabai merah keriting" : cabai_merah_keriting,
        "gula pasir" : gula_pasir,
        "gula pasir lokal" : gula_pasir_lokal,
        "gula pasir kualitas premium" : gula_pasir_kualitas_premium,
        "bawang merah": bawang_merah,
        "bawang merah ukuran sedang": bawang_merah_ukuran_sedang,
        "bawang putih": bawang_putih,
        "bawang putih ukuran sedang": bawang_putih_ukuran_sedang,
        "minyak goreng": minyak_goreng,
        "minyak goreng curah": minyak_goreng_curah,
        "minyak goreng kemasan bermerk 1": minyak_goreng_kemasan_bermerk_1,
        "minyak goreng kemasan bermerk 2": minyak_goreng_kemasan_bermerk_2,
        "daging ayam": daging_ayam,
        "daging ayam ras segar": daging_ayam_ras_segar,
        "daging sapi": daging_sapi,
        "daging sapi kualitas 1": daging_sapi_kualitas_1,
        "daging sapi kualitas 2": daging_sapi_kualitas_2,
        "telur ayam": telur_ayam,
        "telur ayam ras segar": telur_ayam_ras_segar 
    }
    return switch.get(type, "invalid input")


def scaler_data(type):
    global scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler_train = scaler.fit_transform(type.reshape(-1, 1))

    return scaler_train


def make_prediction(train, test, val, date_start, date_end):
    global day, predicted
    data_train = time_step_generator(train, time_size= window_size_train, batch_size= batch_size_train, shuffle_data=num_shuffle)
    data_val = time_step_generator(val, time_size= window_size_val, batch_size =batch_size, shuffle_data=num_shuffle)
    
    model = Sequential([
        Conv1D(16, kernel_size=2, activation='relu', strides=2, input_shape=[None, 1]),
        LSTM(24, return_sequences=True),
        LSTM(20),
        Dense(64, activation='relu'),
        Dense(1)
    ])

    model.compile(tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse', metrics=['mape', 'mae'])
    history = model.fit(data_train, epochs=5, verbose=1, validation_data=data_val)
    mae = history.history['mae']
    mape = history.history['mape']

    predict_list = []
    data_out = []


    test = test.reshape(1, -1)
    data_test = list(test)
    data_test = data_test[0].tolist()

    step = 234 
    i = 1

    day = (date_end - date_start)
    day = day.days
    day = day + 2
    print(day)

    while(i <= day):
        if(len(data_test) > 100):
            data_in = np.array(data_test[1:])
            data_in = data_in.reshape(1, -1)
            data_in = data_in.reshape((1, step, 1))
            predict = model.predict(data_in, verbose=0)
            print(data_in)
            data_test.extend(predict[0].tolist())
            data_test= data_test[1:]
            data_out.extend(predict.tolist())
            i = i+1
        else:
            data_in = np.array(data_test)
            data_test = data_in.reshape(1, step, 1)
            predict = model.predict(data_in, verbose=0)
            data_test.extend(predict[0].tolist())
            data_out.extend(predict.tolist())
            i= i+1

    predicted = scaler.inverse_transform(np.array(data_out).reshape(-1, 1))
    output = int(predicted[-1])
    
    return output


plt.rcParams["figure.figsize"] = [7, 3.50]
plt.rcParams["figure.autolayout"] = True    

@app.route('/plot')   
def make_plot():
    fig = figures()
    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def figures():
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(dates.date, dates.predict.round())
    return fig


if __name__ == '__main__':
    app.run(debug=True)