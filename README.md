# Mlops_jaaa
# Stock Market Prediction and Optimal Portfolio Construction

## Project Description

The goal of our project is to use deep learning to predict stock market returns and volatility for a selection of five companies. We then use these predictions to construct an optimal investment portfolio that maximizes returns while minimizing risk (volatility). 

### Overall Goal of the Project

The primary objective of our project is to develop a predictive model that can forecast stock returns and volatility over a one-year horizon. By using historical stock market data, our model will guide the user in predicting future stock price behavior. This will assist with the construction of an optimal portfolio with respect to risk and return expectations. 

### Framework and Integration

We utilize the [PyTorch(https://pytorch.org/docs/stable/index.html)] framework for constructing our deep neural network for predictions.

### Data Collection and Initial Dataset

We construct the dataset using the [yfinance(https://github.com/ranaroussi/yfinance)] python package which allows us to extract our stock data from Yahoo Finance. 
Initially, we collect historical stock market data for five companies. This data will include daily stock prices, trading volumes, and other relevant financial indicators over a multi-year period. 

By preprocessing and analyzing this data, we will generate the necessary input features for our deep neural network.
We will use stock data from 2015 to 2022 for our model training. And subsequently use 2023 data to evaluate our model.

(As the project progresses, we may expand our dataset to include additional companies and/or longer time periods to further enhance our predictions.)

### Model(s)

We use a: 
1) **Deep neural network**

and perhaps compare the result with the following models:
2) **Linear regression**
3) **Time series**
4) **Reinforcement learning**

By experimenting with and refining these models, we aim to develop a robust predictive system that delivers accurate forecasts of stock returns and volatility. This will enable the construction of an optimal portfolio that balances potential returns against associated risks.


### Team members

Anjali Sarawgi 
Ali Najibpour Nashi
Annas Namouchi
John-Pierre Weideman
