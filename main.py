#!/bin/bash python3

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.preprocessing import PolynomialFeatures

from lib.ols import ols_regression
from lib.ridge import ridge_regression
from lib.logit import logit_regression

plt.style.use('seaborn-poster')

if __name__ == '__main__':

	# DRIVER FOR VARIOUS REGRESSION EXAMPLES
	# UNCOMMENT plt.show FOR VISUALIZATIONS

	data = pd.read_csv("data/mpg.csv", sep=",")
	
	# SIMPLE OLS REGRESSION
	
	t = np.array(data['mpg'])
	x = np.array(data['weight'])

	X = np.array([np.ones(len(t)), x]).T

	model = ols_regression()
	model = ols_regression().fit(X,t)

	t_hat = model.predict(X)
	
	plt.scatter(np.array(data['weight']), np.array(data['mpg']), color='g')
	plt.plot(np.array(data['weight']), t_hat, color='k')
	plt.xlabel('weight')
	plt.ylabel('mpg')
	plt.show()

	# POLYNOMIAL OLS REGRESSION 
	
	t = np.array(data['mpg'])
	x = np.array(data['weight'])

	X = np.array([np.ones(len(t)), x, np.square(x)]).T

	model = ols_regression()
	model = ols_regression().fit(X,t)

	t_hat = model.predict(X)

	plt.scatter(x, t, color='g')
	x, t_hat = zip(*sorted(zip(x,t_hat))) # plot points in order
	plt.plot(x, t_hat, color='k')
	plt.xlabel('weight')
	plt.ylabel('mpg')
	plt.show()

	# MULTIVARIATE OLS REGRESSION

	t = np.array(data['mpg'])
	x = np.array(data['weight'])
	y = np.array(data['displacement'])

	X = np.array([np.ones(len(t)), x, y]).T

	model = ols_regression()
	model = ols_regression().fit(X,t)

	t_hat = model.predict(X)

	# Create set of ordered pairs to work with on graph

	x_pts = np.linspace(x.min(), x.max(), 30)
	y_pts = np.linspace(y.min(), y.max(), 30)
	x_pairs, y_pairs = np.meshgrid(x_pts,y_pts)

	# Get values for all ordered pairs in set using model

	z = model.coef_[0] + model.coef_[1] * x_pairs + model.coef_[2] * y_pairs

	# Graph

	fig = plt.figure(figsize = (1000,1000))
	ax = plt.axes(projection='3d')
	ax.plot_surface(x_pairs,y_pairs,z, rstride=1, cstride=1, color='teal', alpha=0.4, antialiased=False)
	ax.scatter(x,y,t, c = 'r')
	ax.set_ylabel('displacement')
	ax.set_title('mpg', fontsize=20)
	plt.xlabel('\n\n\nweight', fontsize=18)
	plt.ylabel('\n\n\ndisplacement', fontsize=16)
	plt.show()

	# POLYNOMIAL MULTIVARIATE OLS REGRESSION

	t = np.array(data['mpg'])
	x = np.array(data['weight'])
	y = np.array(data['displacement'])
	X = np.array([x, y]).T

	degree = 2
	poly = PolynomialFeatures(degree)
	X = poly.fit_transform(X)

	model = ols_regression()
	model = ols_regression().fit(X,t)

	t_hat = model.predict(X)

	# Create set of ordered pairs to work with on graph

	x_pts = np.linspace(x.min(), x.max(), 30)
	y_pts = np.linspace(y.min(), y.max(), 30)
	x_pairs, y_pairs = np.meshgrid(x_pts,y_pts)

	# Get values for all ordered pairs in set using model

	z = model.coef_[0] + model.coef_[1] * x_pairs + model.coef_[2] * y_pairs + model.coef_[3] * x_pairs**2 + (model.coef_[4] * y_pairs * x_pairs) + (model.coef_[5] * y_pairs**2)

	# Graph

	fig = plt.figure(figsize = (1000,1000))
	ax = plt.axes(projection='3d')
	ax.plot_surface(x_pairs,y_pairs,z, rstride=1, cstride=1, color='teal', alpha=0.4, antialiased=False)
	ax.scatter(x,y,t, c = 'r')
	ax.set_ylabel('displacement')
	ax.set_title('mpg', fontsize=20)
	plt.xlabel('\n\n\nweight', fontsize=18)
	plt.ylabel('\n\n\ndisplacement', fontsize=16)
	plt.show()
