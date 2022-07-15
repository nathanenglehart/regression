import numpy as np
import pandas as pd

# Nathan Englehart (Summer, 2022)

class lasso_regression() :
      
  def __init__(self, learning_rate=0.01, iterations=1000, l1_penality=500) :
          
       """ Lasso regression class based on sklearn functionality using gradient descent """

       self.learning_rate = learning_rate 
       self.iterations = iterations
       self.l1_penality = l1_penality
          
  def fit( self, X, t ) :
         
       """ Fits lasso regression model with given train matrix and target vector
	  	 
		 Args:
			
			 X::[Numpy Array]
				  Train matrix (build before putting into function)
			
			 t::[Numpy Array]
				  Target vector
       """
          
       self.m, self.n = X.shape
          
       self.theta = np.zeros( self.n )
       self.intercept = 0
       self.X = X
       self.t = t
          
       self.gradient_descent()
              
       return self

  def gradient_descent():
      
      """ Computes optimal weights for lasso function using gradient descent algorithm """

      for i in range(self.iterations):
          
          t_hat = self.predict( self.X )
          
          # compute gradients  
          
          d_theta = np.zeros(self.n)
          
          for j in range(self.n) :
            if self.theta[j] > 0:  
              d_theta[j] = ( - ( 2 * ( self.X[:, j] ).dot( self.t - t_hat ) ) + self.l1_penality ) / self.m
            else:
              d_theta[j] = ( - ( 2 * ( self.X[:, j] ).dot( self.t - t_hat ) ) - self.l1_penality ) / self.m
       
          d_intercept = - 2 * np.sum( self.t - t_hat ) / self.m 
          
          # update weights
      
          self.theta = self.theta - self.learning_rate * d_theta 
          self.intercept = self.intercept - self.learning_rate * d_intercept
       
  def predict( self, X ) :
      
      """ Generates predictions for the given matrix based on model.
      		
		Args:
			
			X::[Numpy Array]
				Test matrix (build before putting into function)
      """
      return X.dot( self.theta ) + self.intercept
