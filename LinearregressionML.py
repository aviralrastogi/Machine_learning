import matplotlib.pyplot as mp
import numpy as np
import pandas as pd
print('helo')
#inserting the data in the numpy array
#left row indicates squarefeet house
#right row indicates price of the house
dataset=np.array([[1500,158900],
[1700,169850],
[1750,178950],
[1800,178650],
[1820,180000],
[1920,186850],
[1450,150000],
[1590,149870],
[1596,158620],
[1623,159990],
[1878,189680],
[1658,168980],
[1720,170000],
[1985,190000],
[2000,198510],
[2100,200000],
[2050,193580],
[1990,200000],
[1965,195180],
[1970,198680],
[2120,201650],
[2200,220000],
[2156,216510],
[1269,138550],
[1489,149850],
[1785,179850],
[1965,196280],
[1948,195680],
[2008,200000],
[2079,205880],
[2116,210000],
[2230,220000],
[2200,219850],
[2220,222000],
[2365,235680],
[2325,239580],
[2396,240000],
[2489,248850],
[2420,245590],
[2398,240000],
[2350,236840],
[2375,230000],
[2236,226260],
[2347,220590],
[2459,239840]])
#slicing the array in left-row
Xaxis=dataset[:,0]
#slicing the array in right-row
Yaxis=dataset[:,1]
Yaxis=Yaxis.reshape(Yaxis.size,1)

print(Xaxis)
print(Yaxis)
Xaxis=np.vstack((np.ones((Xaxis.size,)),Xaxis)).T #getting the value in the estimated site
print(Xaxis)
#plotting the graph for the Xaxis and Yaxis
mp.scatter(Xaxis[:,1],Yaxis)
mp.show()
def model(Xaxis,Yaxis,learning_rate,iteration):
    m=Yaxis
    theta=np.zeros((2,1))    #initalize theta with zeroes in the matrix
    for i in range(iteration):
        Y_predicatable=np.dot([1,15450],theta)
        cost=(1/(2*m))*np.sum(np.square(Y_predicatable-Yaxis))

        d_theta=(1/m)*np.dot(Xaxis.T,Y_predicatable-Yaxis)
        theta=theta-learning_rate*d_theta

    return theta
learning_rate=0.0000000005
iteration=100
model(Xaxis,Yaxis,learning_rate,iteration)
