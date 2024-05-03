# Financial market forecasting using ARIMA, LSTM models.

Authors: Damir Dairov, Kamila Amangeldinova.

In this repository you can find the implitation of machine learning techniques like ARIMA (baseline model) and LSTM for stock price forecasting. The models were tested on 6 financial markets: Amazon, CAC, IBM, Microsoft, Nasdaq, and S&P 500. Data was given by Euklid LTD for the time window of March 1994 to March 2024.

## Coding steps:
### Importing Libraries & Loading Data
All the required libraries are imported in one block of code such as numpy, pandas, matplotlib, seaborn, statsmodels, and pandas_ta. The data has been loaded into pandas dataframe.

### Preprocessing Data
The 1st step was to check the shape of all the datasets, check for missing and duplicated values and clean them. After, there was seasonal decomposition (to check for the trends and seasonality), and log transformation (to make the data stationary). Adding technical indicator as new features for LSTM model like RSI (Relative Strength Index), and EMA (Exponential Moving Average).

### EDA
The distribution of data can be found in this section.

### Model selection & Model Fit
As the baseline there was selected ARIMA model, parameters p,d,q were selected with auto_arima. ARIMA and LSTM models are fit to the data, the results are assessed with the metrics RMSE, MAE (for LSTM).

### Results & Conclusion
In this section, there is the difference with original and predicted data to compare. You can find one of the easiest models in this repository and its usage to predict stock prices and to see the difference in the datasets and in the models. 
