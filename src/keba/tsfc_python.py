import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, HistoricalAverage, WindowAverage, AutoARIMA, ExponentialSmoothing, Theta, Croston, DampedTrend, SeasonalExponentialSmoothing

horizon = 24
model = StatsForecast(models=[
    Naive(),
    SeasonalNaive(season_length=24),
    HistoricalAverage(),
    WindowAverage(window_size=24),
])
                      
sf = StatsForecast(models=[model], freq='H')

sf.fit(df_train, id_col='id', time_col='ds', value_col='y_b')
preds = sf.predict(horizon=horizon)