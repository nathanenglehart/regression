#!/usr/bin/env python3

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.preprocessing import PolynomialFeatures

from lib.ols import ols_regression
from lib.ridge import ridge_regression
from lib.logit import logit_regression

plt.style.use('seaborn-poster')	

def logit_driver():
	
	# DRIVER FOR LOGIT REGRESSION EXAMPLE
	# UNCOMMENT plt.show FOR VISUALIZATIONS

	# SIMPLE LOGIT REGRESSION 

	x_1 = np.arange(10)
	t = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
	
	X = np.array([np.ones(len(t)),x_1]).T
	#X = np.array([np.ones(len(t)),x_1]).T # adds ones - dont think we need it 

	model = logit_regression()
	model = logit_regression().fit(X,t)

	t_hat = model.predict(X)
	print('preds:',t_hat)

	plt.scatter(x_1,t, color='tab:olive')
	plt.plot(x_1,t_hat, color='tab:cyan')
	plt.xlabel('x_1')
	plt.ylabel('t')
	plt.savefig('figs/simple_logit.png')


def ols_driver():

	# DRIVER FOR VARIOUS OLS REGRESSION EXAMPLES
	# UNCOMMENT plt.show FOR VISUALIZATIONS

	data = pd.read_csv('data/mpg.csv', sep=",")
	
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
	plt.savefig('figs/simple_ols.png')
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
	plt.savefig('figs/polynomial_ols.png')
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

	fig = plt.figure(figsize = (100,100))
	ax = plt.axes(projection='3d')
	ax.plot_surface(x_pairs,y_pairs,z, rstride=1, cstride=1, color='teal', alpha=0.4, antialiased=False)
	ax.scatter(x,y,t, c = 'r')
	ax.set_ylabel('displacement')
	ax.set_title('mpg', fontsize=20)
	plt.xlabel('\n\n\nweight', fontsize=18)
	plt.ylabel('\n\n\ndisplacement', fontsize=16)
	plt.savefig('figs/multivariate_ols.png')
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

	fig = plt.figure(figsize = (100,100))
	ax = plt.axes(projection='3d')
	ax.plot_surface(x_pairs,y_pairs,z, rstride=1, cstride=1, color='teal', alpha=0.4, antialiased=False)
	ax.scatter(x,y,t, c = 'r')
	ax.set_ylabel('displacement')
	ax.set_title('mpg', fontsize=20)
	plt.xlabel('\n\n\nweight', fontsize=18)
	plt.ylabel('\n\n\ndisplacement', fontsize=16)
	plt.savefig('figs/polynomial_multivariate_ols.png')
	plt.show()

def ridge_driver():

	# DRIVER FOR VARIOUS OLSREGRESSION EXAMPLES
	# UNCOMMENT plt.show FOR VISUALIZATIONS

	data = pd.read_csv("data/mpg.csv", sep=",")
	
	# SIMPLE RIDGE REGRESSION
	
	t = np.array(data['mpg'])
	x = np.array(data['weight'])

	X = np.array([np.ones(len(t)), x]).T

	model = ridge_regression()
	model = ridge_regression().fit(X,t)

	t_hat = model.predict(X)
	
	plt.scatter(np.array(data['weight']), np.array(data['mpg']), color='g')
	plt.plot(np.array(data['weight']), t_hat, color='k')
	plt.xlabel('weight')
	plt.ylabel('mpg')
	plt.savefig('figs/simple_ridge.png')
	plt.show()

	# POLYNOMIAL RIDGE REGRESSION 
	
	t = np.array(data['mpg'])
	x = np.array(data['weight'])

	X = np.array([np.ones(len(t)), x, np.square(x)]).T

	model = ridge_regression()
	model = ridge_regression().fit(X,t)

	t_hat = model.predict(X)

	plt.scatter(x, t, color='g')
	x, t_hat = zip(*sorted(zip(x,t_hat))) # plot points in order
	plt.plot(x, t_hat, color='k')
	plt.xlabel('weight')
	plt.ylabel('mpg')
	plt.savefig('figs/polynomial_ridge.png')
	plt.show()

	# MULTIVARIATE RIDGE REGRESSION

	t = np.array(data['mpg'])
	x = np.array(data['weight'])
	y = np.array(data['displacement'])

	X = np.array([np.ones(len(t)), x, y]).T

	model = ridge_regression()
	model = ridge_regression().fit(X,t)

	t_hat = model.predict(X)

	# Create set of ordered pairs to work with on graph

	x_pts = np.linspace(x.min(), x.max(), 30)
	y_pts = np.linspace(y.min(), y.max(), 30)
	x_pairs, y_pairs = np.meshgrid(x_pts,y_pts)

	# Get values for all ordered pairs in set using model

	z = model.coef_[0] + model.coef_[1] * x_pairs + model.coef_[2] * y_pairs

	# Graph

	fig = plt.figure(figsize = (100,100))
	ax = plt.axes(projection='3d')
	ax.plot_surface(x_pairs,y_pairs,z, rstride=1, cstride=1, color='teal', alpha=0.4, antialiased=False)
	ax.scatter(x,y,t, c = 'r')
	ax.set_ylabel('displacement')
	ax.set_title('mpg', fontsize=20)
	plt.xlabel('\n\n\nweight', fontsize=18)
	plt.ylabel('\n\n\ndisplacement', fontsize=16)
	plt.savefig('figs/multivariate_ridge.png')
	plt.show()

	# POLYNOMIAL MULTIVARIATE RIDGE REGRESSION

	t = np.array(data['mpg'])
	x = np.array(data['weight'])
	y = np.array(data['displacement'])
	X = np.array([x, y]).T

	degree = 2
	poly = PolynomialFeatures(degree)
	X = poly.fit_transform(X)

	model = ridge_regression()
	model = ridge_regression().fit(X,t)

	t_hat = model.predict(X)

	# Create set of ordered pairs to work with on graph

	x_pts = np.linspace(x.min(), x.max(), 30)
	y_pts = np.linspace(y.min(), y.max(), 30)
	x_pairs, y_pairs = np.meshgrid(x_pts,y_pts)

	# Get values for all ordered pairs in set using model

	z = model.coef_[0] + model.coef_[1] * x_pairs + model.coef_[2] * y_pairs + model.coef_[3] * x_pairs**2 + (model.coef_[4] * y_pairs * x_pairs) + (model.coef_[5] * y_pairs**2)

	# Graph

	fig = plt.figure(figsize = (100,100))
	ax = plt.axes(projection='3d')
	ax.plot_surface(x_pairs,y_pairs,z, rstride=1, cstride=1, color='teal', alpha=0.4, antialiased=False)
	ax.scatter(x,y,t, c = 'r')
	ax.set_ylabel('displacement')
	ax.set_title('mpg', fontsize=20)
	plt.xlabel('\n\n\nweight', fontsize=18)
	plt.ylabel('\n\n\ndisplacement', fontsize=16)
	plt.savefig('figs/polynomial_multivariate_ridge.png')
	plt.show()

if __name__ == '__main__':
	
	#ols_driver()
	#ridge_driver()
	logit_driver()
