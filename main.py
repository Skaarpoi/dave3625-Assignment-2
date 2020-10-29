# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as panda
from datetime import datetime as dt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# prepare Dataset
df = panda.read_csv('NAS.csv')

# Drop unwanted columns and rows with no data
df = df.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'])
df = df.dropna()

# Convert dates to useable data in sklearn
df['Date'] = panda.to_datetime(df['Date'])
df['Date'] = df['Date'].map(dt.toordinal)

# Creating the x and y values
X = df.drop(columns='Close')
y = df['Close']

# Normalizing the data, normalizer will be used later for user inputted dates
normalizer = X.max()[0]
X = X / X.max()

# Fitting the dataset
poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X)
poly.fit(X_poly, y)
lin2 = LinearRegression()
lin2.fit(X_poly, y)

# Plotting the actual prices
plt.scatter(X, y, color='green')

# Plotting predicted prices
plt.plot(X, lin2.predict(poly.fit_transform(X)), color='red')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


# Method to fit the input date to the prediction model
def predict_price(date):
    d = dt.strptime(date, '%Y-%m-%d')
    do = d.toordinal()
    do = np.array([[do]])
    do = do / normalizer
    return lin2.predict(poly.fit_transform(do))


# listens constantly for user input
while True:
    print('Enter date formatted by YYYY-MM-DD: \n')
    input_date = input('')
    price = predict_price(input_date)
    print(f'Predicted price: {price} \n')
