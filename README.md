# Financial market forecasting using ARIMA, LSTM models.

Authors: Damir Dairov, Kamila Amangeldinova.

# Introduction
In this repository you can find the implitation of machine learning techniques like ARIMA (baseline model) and LSTM for stock price forecasting. The models were tested on 6 financial markets: Amazon, CAC, IBM, Microsoft, Nasdaq, and S&P 500. Data was given by Euklid LTD for the time window of March 1994 to March 2024.

## Coding steps:
### Importing Libraries & Loading Data
All the required libraries are imported in one block of code such as numpy, pandas, matplotlib, seaborn, statsmodels, and pandas_ta. The data has been loaded into pandas dataframe.

### Preprocessing Data
The 1st step was to check the shape of all the datasets, check for missing and duplicated values and drop them. For ARIMA model, it was better to merge all the datasets by outer join by 'Date' column and take only 'Close' prices from all the datasets. After, there was seasonal decomposition (to check for the trends and seasonality), and log transformation (to make the data stationary). Before log transformation, p-value (a p-value below 0.05 is commonly used as a threshold to determine statistical significance) in all datasets were bigger than the significant value (>0.05), after tranforming, it became less than 0.05. So, now, we made sure that our data is stationary, and we can fit the model. However, we wanted also to check for the normality of data, where it showed that data is not normally distributed, but, by plotting the distribution graphs (that you can see in this step), data seems to be normally distributed. After that we started fitting the model.
For LSTM: Adding technical indicator as new features for LSTM model like RSI (Relative Strength Index), and EMA (Exponential Moving Average). Next step is to range the data from 0 to 1 using MinMax Scaler and converting it into numpy array. Adding target columns: Target column represents up to how much the next close price have changed; Targetclass shows either it went up or down; TargetNextClose shows next week closing price.

### EDA
This section is done just to see the trend of all the datasets. As you can see, the overall trend for all the stocks is rising (so actually we expect them to increase), however, for Amazon, in 2022 there was a sharp decrease.

### Train/Test split
Splitting the data into train/test was done by 80-20% (for both models), first 80% of sequential data was taken as train, and other 20% as test set. The split is not random. Instead, it's sequential, it means that the training set consists of the initial part of the sequence, and the test set consists of the remaining part. This ensures that the model is evaluated on unseen data that comes after the data it was trained on (mimicking real-world scenarios where the model is used to make predictions on future data).

### Model selection & Model Fit
As the baseline there was selected ARIMA model because this is traditional model for time series. AutoRegressive (AR): It's a model that uses the relationship between an
observation and a number of previous time steps. Integrated (I): It's the differencing of raw observations (subtracting an observation from an observation at the previous time step) in order to make the time series stationary. Moving Average (MA): It's a model that uses the dependency between an observation and a residual error from a moving average model applied to previous time steps. So, all  best parameters (AR, I, MA = p, d, q) were selected with auto_arima (with p,q values minimum 0 and maximum 3, because when adjusted to make it maximum 10, for example, it still gave the same RMSE, so we stopped doing it like maximum of 3). For ARIMA, there was created the function, that consists of choosing p,d,q values, plotting the graph, and calculating RMSE, where, then all the datasets were just called in the function.
LSTM it’s type of RNN that designed to handle sequence data. Particularly, useful when dealing with sequences where long-term dependencies are important. LSTM networks are equipped with memory cells that can maintain information over long periods of time. LSTM consists of several gates(input, forget, output) that regulate the flow of information into and out of memory cell.
ARIMA and LSTM models are fit to the data, LSTM is done to compare the results with ARIMA. The results are assessed with the metrics RMSE, MAE (for LSTM). 

### Results & Conclusion
In ipynb file, you can see the results with RMSE values for each dataset, and conclusion for after each model. Overall, for ARIMA and LSTM, the datasets act similarly, Amazon shows the worst accuracy and SP500 shows the best accuracy among all those compared datasets. The reason for Amazon to show the worst performance could not be found in the dataset itself, but it's due to an external factors that affect this financial market. For example, in 2022, Amazon announced a 20-for-1 split of common shares, this decrease in the prices affects overall trend. As the purpose of this project was 1) to compare the datasets, why they perform the worst, and the best; we concluded it depends on the external factors. 2) to compare the models ARIMA and LSTM, the performance of LSTM is better; the reasons are: LSTMs can capture complex patterns and dependencies in sequential data, they are good at capturing nonlinear relationships and long-term dependencies in time series data, while for ARIMA, we need to tranform the data to make it stationary, removing trends and seasonality.

If we had more time, future research would be devoted to the development of other models and algorithms to compare the results between them, and to understand how models work. In the last section in ipynb, you can see the references for our work that helped us develop the models.
