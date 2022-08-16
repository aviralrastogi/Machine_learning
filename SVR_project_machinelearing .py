#Description this program predicts the facebook price for specfic day
from sklearn.svm import SVR  #it will import the support vector machine algorithim from sklearn
import numpy as np
from datetime import datetime       #done after debugging
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
#Load the data
from sklearn import datasets
d1=pd.read_csv('Facebook_predictor.csv')
#Store the data and show the data
#print(d1)
#get the number of rows and and columns
print(d1.shape)
#get and the print the last row of data
actual_price=d1.tail(1)
print(actual_price)
#Prepare the data for training the SVR Models
#Get all the data except not the last row
d1=d1.head(len(d1)-1)
# Data to print not last row
print(d1)
#Create empty list to store the independent and dependent data
days=list()

adj_close_prices=list()

#Get the data and adjusted close price
d1_days=d1.loc[:,'Date']


d1_adj_close=d1.loc[:,'Adj Close']     #Taking two parameters in the dataset to train the data


#get the independent data set
for day in d1_days:
    days.append([int(day.split('-')[2])])         #slicing the value of the days
#Create the dependent data set

for adj_close in d1_adj_close:
    adj_close_prices.append(float(adj_close))
print(adj_close_prices)
#Write the days and the adjusted and the adjusted close prices

#print(days)

#print(adj_close_prices)

#Create the 3 support Vector Regression Models




#Create and train a SVR Model using linear kernel
lin_svr=SVR(kernel='linear',C=1000.0)
lin_svr.fit(days,adj_close_prices)
#Create and train a SVR Model using polynomial kernel
poly_svr=SVR(kernel='poly',C=1000.0,degree=2)
poly_svr.fit(days,adj_close_prices)

#Create and train a SVR Model using rbf kernel
rbf_svr=SVR(kernel='rbf',C=1000.0,gamma=0.15)
rbf_svr.fit(days,adj_close_prices)

#ploting could be done later

#figssize is a tuple of the width and height of the figure in inches
plt.figure(figsize=(16,8))
plt.scatter(days,adj_close_prices,color='red',label='Data')
plt.plot(days,rbf_svr.predict(days),color="orange",label='RBF Model')
plt.plot(days,rbf_svr.predict(days),color="green",label='Polynomial Model')
plt.plot(days,rbf_svr.predict(days),color="blue",label='Linear Model')
plt.legend()
plt.show()














