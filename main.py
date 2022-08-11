#!/usr/bin/env python3

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing

from lib.closed.ols import ols_regression
from lib.closed.ridge import ridge_regression
from lib.gd.lasso import lasso_regression
from lib.gd.logit import logit_regression

plt.style.use('seaborn-poster')	

def lasso_driver():
	
	# DRIVER FOR LASSO REGRESSION EXAMPLE
	# UNCOMMENT plt.show FOR VISUALIZATIONS
	
	data = pd.read_csv('data/mpg.csv', sep=",")

	# SIMPLE LASSO REGRESSION
	
	t = preprocessing.scale(np.array(data['mpg']))
	x = preprocessing.scale(np.array(data['weight']))

	X = np.array([np.ones(len(t)), x]).T

	model = lasso_regression()
	model = lasso_regression().fit(X,t)

	t_hat = model.predict(X)

	plt.scatter(x, t, color='none', edgecolor='tab:cyan')
	plt.plot(x, t_hat, color='tab:orange', alpha=0.5)
	plt.xlabel('weight')
	plt.ylabel('mpg')
	plt.savefig('figs/simple_lasso.png')
	plt.show()

def logit_driver():
	
	# DRIVER FOR LOGIT REGRESSION EXAMPLE
	# UNCOMMENT plt.show FOR VISUALIZATIONS

	# (VERY) SIMPLE LOGIT REGRESSION 

	x_1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
	t = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
	
	X = np.array([np.ones(len(t)),x_1]).T
	
	model = logit_regression()
	model = logit_regression().fit(X,t)

	t_probs = model.predict_proba(X)
	print('prob preds:',t_probs)

	plt.scatter(x_1,t, color='tab:olive')
	plt.plot(x_1,t_probs, color='tab:cyan')
	plt.xlabel('x_1')
	plt.ylabel('t')
	plt.show()

	# SIMPLE LOGIT REGRESSION 

	data = pd.read_csv('data/pima.csv',sep=",")
	
	t = np.array(data['diabetes']) 
	x_1 = preprocessing.scale(np.array(data['mass'], dtype=np.float128)) # , dtype=np.float128 to prevent overflow
	
	X = np.array([np.ones(len(t)), x_1]).T

	model = logit_regression()
	model = logit_regression().fit(X,t)

	t_probs = model.predict_proba(X)
	print('prob preds',t_probs)

	plt.scatter(x_1,t, color='tab:olive')
	x_1, t_probs = zip(*sorted(zip(x_1,t_probs))) # plot points in order
	plt.plot(x_1,t_probs, color='tab:cyan')
	plt.xlabel('x_1')
	plt.ylabel('t')
	plt.savefig('figs/simple_logit.png')
	plt.show()

	# POLYNOMIAL LOGIT REGRESSION
	
	t = np.array(data['diabetes']) 
	x_1 = preprocessing.scale(np.array(data['glucose'], dtype=np.float128))
	
	X = np.array([np.ones(len(t)), x_1, np.square(x_1)]).T

	model = logit_regression()
	model = logit_regression().fit(X,t)

	t_probs = model.predict_proba(X)
	print('prob preds',t_probs)

	plt.scatter(x_1,t, color='tab:olive')
	x_1, t_probs = zip(*sorted(zip(x_1,t_probs))) # plot points in order
	plt.plot(x_1,t_probs, color='tab:cyan')
	plt.xlabel('x_1')
	plt.ylabel('t')
	plt.savefig('figs/polynomial_logit.png')
	plt.show()

	# MULTIVARIATE LOGIT REGRESSION

	t = np.array(data['diabetes']) 
	x_1 = preprocessing.scale(np.array(data['glucose'], dtype=np.float128))
	x_2 = preprocessing.scale(np.array(data['mass'], dtype=np.float128))
	
	X = np.array([np.ones(len(t)), x_1, x_2]).T

	model = logit_regression()
	model = logit_regression().fit(X,t)

	t_probs = model.predict_proba(X)
	print('prob preds',t_probs)

	x_pts = np.linspace(x_1.min(), x_1.max(), 30)
	y_pts = np.linspace(x_2.min(), x_2.max(), 30)
	x_pairs, y_pairs = np.meshgrid(x_pts,y_pts)

	# Get values for all ordered pairs in set using model

	z = 1 / (1 + np.e ** (model.coef_[0] + model.coef_[1] * x_pairs + model.coef_[2] * y_pairs))

	# Graph

	fig = plt.figure(figsize = (100,100))
	ax = plt.axes(projection='3d')
	ax.plot_surface(x_pairs,y_pairs,z, rstride=1, cstride=1, color='tab:cyan', alpha=0.4, antialiased=False)
	ax.scatter(x_1,x_2,t, c = 'tab:olive')
	ax.set_ylabel('mass')
	ax.set_title('pima', fontsize=20)
	plt.xlabel('\n\n\nglucose', fontsize=18)
	plt.ylabel('\n\n\nmass', fontsize=16)
	plt.savefig('figs/multivariate_logit.png')
	plt.show()

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
	#lasso_driver()
