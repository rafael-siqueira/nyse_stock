# NYSE Stock Close Prices Prediction: RNN Practice

This project was created in order to implement my first RNN and hone my time-series forecasting skills using TF/Keras. After selecting a simple model, I also implemented it to run locally in order to be able to receive the prediction via a Telegram Bot on my phone.

A quick overview is detailed below and there are also explanation notes in the notebook where the project is implemented (NYSE_RNN_vgit.ipynb):

1) I used only close prices of a single stock (Apple: AAPL) to predict its closing price on the following day.
2) Used StandardScaler for price normalization and only implemented lag features and a rolling mean feature
3) Due to the nature of the features chosen, there were 3 parameters to pick to determine the final number and value of features:
- n_lags: number of past values used for predicting next one
- window: size of window used to compute rolling mean
- min_periods: minimum number of past values available needed for computting rolling mean
4) I initially chose n_lags = 2, window = 7 and min_periods = 3 to fit the initial model
5) In order to take advantage of and implement different types of RNN-NN cells, I chose a model with GRU and LSTM units, dropout for regularization and dense units with ReLU activation functions
6) After testing several combinations of feature parameters and model configurations, I ended up using n_lags = 4, window = 10 and min_periods = 3 for the final model
7) **The obtained training and validation MSE were 0.0122 and 6.2112e-04, respectively**

Since this is a simple personal project and an initial version, I only developed it to run locally (`python main.py`).

I pre-configured a bot on Telegram and when the script is run, a connection is established with this bot and it sits idle, waiting for a command. When the bot receives the message '/pred' it returns a closing price prediction for next day with a graph depicting the recent price movements and the predicted price. Find below a screenshot of the Telegram bot working:

![Screenshot](https://github.com/rafael-siqueira/nyse_stock/blob/main/pred_resize.png)









