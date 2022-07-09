import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

from lib.ols import ols_regression

if __name__ == '__main__':
  data = pd.read_csv("womens100.csv",header=None)
  data.columns = ['years','times']

  # TODO: Make modular
  
  t = np.array(data['times'])
  X = np.array(data['years'])
  ones = np.ones(len(X))
  X = np.vstack((ones, X)).T

  # Compute and plot regression with sklearn
  
  model = LinearRegression()
  model = LinearRegression().fit(X,t)
  print("sklearn:")
  print(model.predict(X))
  print("")

  # Personal OLS

  model = ols_regression()
  model = ols_regression().fit(X,t)
  print("personal:")
  print(model.predict(X))
  print("")

  plt.scatter(np.array(data['years']), np.array(data['times']), color='g')
  plt.plot(np.array(data['years']), model.predict(X), color='k')
  plt.xlabel('Years')
  plt.ylabel('Times')
  plt.show()
