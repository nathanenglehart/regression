import numpy as np
from math import exp

class logit():
	
	def __init__(self, alpha=0.3, epoch=100):
		
		""" Logistic regression class based on sklearn functionality 
			
			Args:
				alpha::[Float]
					Learning rate for stochastic gradient descent algorithm

				epoch::[Int]
					Number of iterations for stochastic gradient descent algorithm

		"""

		self.alpha = alpha
		self.epoch = epoch
	
	def fit(self,X,t):

		""" Fits logistic regression model with given regressor/train matrix and target vector 

			Args:
				X::[Numpy Array]
					Regressor/train matrix

				t::[Numpy Array]
					Target vector

		"""

		self.coef_ = self.sgd(X, t)

		return self

	def predict(self, X):


		""" Generates predictions for the given matrix based on model

			Args:
				X::[Numpy Array]
					Test matrix

		"""

		preds = list()

		for row in X:
			t_hat = self.coef_[0]
			for i in range(len(row)-1):
				t_hat += self.coef_[i + 1] * row[i]
			preds.append(1.0 / (1.0 + exp(-t_hat)))

		return preds
	
	def sgd(self, X, t):
	
		""" Performs stochastic gradient descent to find optimal coefficients for logit model within fit function

			Args:
				X::[Numpy Array]
					Regressor/train matrix
				
				t::[Numpy Array]
					Target vector

		"""
		
		coef = [0.0 for i in range(len(X[0]))]

		for epoch in range(self.epoch):
			for j in range(len(X)):
				t_hat = coef[0]
				for i in range(len(X[j])-1):
					t_hat += coef[i+1] * X[j][i]
				t_hat = 1.0 / (1.0 + exp(-t_hat))
				error = t[j] - t_hat 
				coef[0] = coef[0] + self.alpha * error * t_hat * (1.0 - t_hat)
				for i in range(len(X[j])-1):
					coef[i + 1] = coef[i + 1] + self.alpha * error * t_hat * (1.0 - t_hat) * X[j][i]
		return coef
