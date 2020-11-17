from telegram.ext import Updater, CommandHandler
import pandas as pd
import numpy as np
import requests
import re
import joblib as jb
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date, timedelta
from tensorflow.keras.models import load_model

model = load_model('model.h5')
scaler = jb.load("scaler.pkl.z")
ticker = 'AAPL'

today = date.today()
tomorrow = today + timedelta(days=1)
past = today - timedelta(days=20)

# Gets past closing prices
def get_stock_data(ticker):
    data = yf.Ticker(ticker)
    df = data.history(period='1d', start=past, end=tomorrow)
    df = df.reset_index(drop=False)

    lag_1 = df.iloc[df.shape[0]-1,3] # close_today
    lag_2 = df.iloc[df.shape[0]-2,3] # close_yesterday
    lag_3 = df.iloc[df.shape[0]-3,3]
    lag_4 = df.iloc[df.shape[0]-4,3]
    rolling_mean = df.iloc[df.shape[0]-10:df.shape[0],3].mean()
    
    raw_data = np.array([lag_4, lag_3, lag_2, lag_1, rolling_mean])
    
    return raw_data, df

# Predicts tomorrow's closing price
def determine_close(raw_data):
    # Data transformation
    X_raw = raw_data
    X_scaled = scaler.transform(X_raw.reshape(-1,1))
    X = np.array([[[X_scaled[0][0], X_scaled[1][0], X_scaled[2][0], X_scaled[3][0], X_scaled[4][0]]]])
    
    # Prediction
    pred = model.predict(X)[0]
    close_pred = scaler.inverse_transform(pred)[0]
    return close_pred

# Plots graph of past values with predicted next value and saves it locally
def plot_graph(df, close_pred):
    graph_df = pd.DataFrame()
    graph_df['date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    graph_df['close'] = df['Close'].copy()
    
    df.reset_index(drop=False, inplace=True)
    
    pred_dict = {}
    pred_dict['date'] = pd.to_datetime(tomorrow, format='%Y-%m-%d')
    pred_dict['close'] = close_pred
    
    pred_df = pd.DataFrame(pred_dict, index=[0])
    graph_df = pd.concat([graph_df, pred_df])
    graph_df.reset_index(drop=True, inplace=True)
    
    fig, ax = plt.subplots()
    # ls: linestyle
    # lw: linewidth
    line, = ax.plot(graph_df['date'], graph_df['close'], color='r', ls = '--', lw = 1, marker='o')
    ax.plot(graph_df['date'][:-1], graph_df['close'][:-1], color='b', lw = 1.5, marker='o')
    plt.savefig('plot.png')
    
    return True

# Called when bot receives message '/pred': computes prediction and sends messages with it and with graph to user
def pred(bot, updater):
    raw_data, df = get_stock_data(ticker)
    close_pred = determine_close(raw_data)
    close_pred = float(str(round(close_pred, 2)))
    plot_graph(df, close_pred)
    
    chat_id = updater.message.chat_id
    bot.send_message(chat_id=chat_id, text='The close price prediction for tomorrow is {} USD'.format(close_pred))
    bot.send_photo(chat_id=chat_id, photo=open('plot.png', 'rb'))

# Establishes connection with bot and waits for '/pred' command
def main():
    updater = Updater('1491318185:AAGmh-3dx75GBlqKawgrzr_WS87LIg1qjF0')
    dp = updater.dispatcher
    dp.add_handler(CommandHandler('pred',pred))
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()