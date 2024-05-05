# Financial market forecasting using ARIMA, LSTM models.

Authors: Damir Dairov, Kamila Amangeldinova.

# Introduction
In this repository you can find the implitation of machine learning techniques like ARIMA (baseline model) and LSTM for stock price forecasting. The models were tested on 6 financial markets: Amazon, CAC, IBM, Microsoft, Nasdaq, and S&P 500. Data was given by Euklid LTD for the time window of March 1994 to March 2024.

## Coding steps:
### Importing Libraries & Loading Data
All the required libraries are imported in one block of code such as numpy, pandas, matplotlib, seaborn, statsmodels, and pandas_ta. The data has been loaded into pandas dataframe.

### Preprocessing Data
for ARIMA: The 1st step was to check the shape of all the datasets, check for missing and duplicated values and clean them. After, there was seasonal decomposition (to check for the trends and seasonality), and log transformation (to make the data stationary). 
for LSTM: Adding technical indicator as new features for LSTM model like RSI (Relative Strength Index), and EMA (Exponential Moving Average). Next step is to range the data from 0 to 1 using MinMax Scaler and converting it into numpy array. Adding target columns: Target column represents up to how much the next close price have changed; Targetclass shows either it went up or down; TargetNextClose shows next week closing price.

### EDA
The distribution of data can be found in this section.

### Train/Test split
Splitting the data into train/test by 0.8/0.2 respectivly. (for LSTM only)

### Model selection & Model Fit
As the baseline there was selected ARIMA model because this is traditional model for time series; parameters p,d,q were selected with auto_arima. LSTM it’s type of RNN that designed to handle sequence data. Particularly, useful when dealing with sequences where long-term dependencies are important. LSTM networks are equipped with memory cells that can maintain information over long periods of time. LSTM consists of several gates(input, forget, output) that regulate the flow of information into and out of memory cell.
ARIMA and LSTM models are fit to the data, LSTM is done to compare the results with ARIMA. The results are assessed with the metrics RMSE, MAE (for LSTM). 

### Results & Conclusion
In this section, there is the difference with original and predicted data to compare. You can find one of the easiest models in this repository and its usage to predict stock prices and to see the difference in the datasets and in the models. 
