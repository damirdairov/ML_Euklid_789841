# Financial market forecasting using ARIMA/LSTM models.

Authors: Damir Dairov, Kamila Amangeldinova.

In this repository you can find the implitation of machine learning techniques like ARIMA (baseline model) and LSTM for stock price forecasting. The models were tested on 6 financial markets: Amazon, CAC, IBM, Microsoft, Nasdaq, and S&P 500. Data was given by Euklid LTD for the time window of March 1994 to March 2024.

## Coding steps:
### Importing Libraries
All the required libraries are imported in one block of code such as numpy, pandas, matplotlib, seaborn, and statsmodels.

### Loading Data
The data has been loaded into pandas dataframe.

### Preprocessing Data
The 1st step was to check the shape of all the datasets, check for missing and duplicated values and clean them. After, there was seasonal decomposition, log transformation to make the data stationary.

### EDA
The distribution of data can be found in this section.

### Model selection
As the baseline there was selected ARIMA model, parameters p,d,q were selected with auto_arima.

### Model Fit
ARIMA model is fit to the data, the results are shown with the metrics RMSE.

### Results
In this section, there is the difference with original and predicted data to compare.

## Conclusion
You can find one of the easiest models in this repository and its usage to predict stock prices and to see the difference in the datasets and in the models.
