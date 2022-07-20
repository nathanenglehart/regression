import numpy as np
import pandas as pd

# Nathan Englehart (Summer, 2022)

class ridge_regression():

  def __init__(self, lam=1.0):
      
      """ Ridge regression class based on sklearn functionality (for modular use)
		
		Args:
			
			lam::[Float]
				Weight penalty for high order polynomials
      """

      self.lam = lam

  def fit(self, X, t):

      """ Fits ridge regression model with given regressor/train matrix and target vector
      
		Args:
			
			X::[Numpy Array]
				Regressor/train matrix (must be built before entering as parameter with method such as sklearn.PolynomialFeatures fit_transform)
			
			t::[Numpy Array]
				Target vector
      """

      I = np.identity(X.shape[1])
      I[0, 0] = 0
      lam_matrix = self.lam * I
      theta = np.linalg.inv(X.T.dot(X) + lam_matrix).dot(X.T).dot(t)
      
      self.theta = theta
      self.coef_ = theta
      
      return self

  def predict(self, X):
      
      """ Generates predictions for the given matrix based on model.
      		Args:
			
			X::[Numpy Array]
				Test matrix (must be built before entering as parameter with method such as sklearn.PolynomialFeatures fit_transform)
      """

      self.predictions = X.dot(self.theta)

      return self.predictions
