import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('model_beras.h5')

def make_prediction(date, type):
    print('a')