import numpy as np

# Nathan Englehart (Summer, 2022)

class logit_regression():
	
	def __init__(self, alpha=0.1, epoch=100):
		
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
					Regressor/train matrix that already has column of ones for intercept

				t::[Numpy Array]
					Target vector

		"""

		self.gd(X, t)

		return self
	
	def gd(self, X, t):
	
		""" Performs gradient descent to find optimal coefficients for logit model within fit function

			Args:
				X::[Numpy Array]
					Regressor/train matrix that already has column of ones for intercept
				
				t::[Numpy Array]
					Target vector

		"""
		
		self.coef_ = [0.0 for i in range(len(X[0]))]  

		for i in range(self.epoch):
		
			for i in X:

				t_hat = self.predict(X)

				gradient = np.dot(X.T, (t_hat - t)) / t.size
				
				self.coef_ = self.coef_ - (self.alpha * gradient)

		return self
	
	def sigmoid(self,z):
		
		""" Sigmoid function (also called the logistic function) where P(t = 1) = sigma(coefs * x) and P(t = 0) = 1 - sigma(coefs * x) """

		return 1.0 / (1.0 + np.exp(z))

	def predict(self, X):


		""" Generates predictions for the given matrix based on model

			Args:
				X::[Numpy Array]
					Test matrix that already has column of ones for intercept

		"""

		return 1 - self.sigmoid(np.dot(X,self.coef_))
	
