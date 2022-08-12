#!/usr/bin/env python3

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing

from lib.closed.ols import ols_regression
from lib.closed.ridge import ridge_regression
from lib.gd.lasso import lasso_regression
from lib.gd.logit import logit_regression

plt.style.use('seaborn-poster')	

bool verbose = False

def compute_classification_error_rate(t,t_hat):
	""" Computes error rate for classification methods such as logistic regression.

		Args:
			
			t::[Numpy Array]
				Truth values

			t_hat::[Numpy Array]
				Prediction values
	
	"""

	error_rate = 0

	for i in range(len(t)):
		
		if(t[i] != t_hat[i]):
			error_rate += 1
	
	return error_rate / len(t) 

def r_squared(t, t_hat):
	
	""" Returns R-Squared for model with given truth values and prediction values. 

		Args:
			
			t::[Numpy Array]
				Truth values

			t_hat::[Numpy Array]
				Prediction values

	"""

	t_bar = t.mean()
	return 1 - ((((t-t_bar)**2).sum())/(((t-t_hat)**2).sum()))

def efron_r_squared(t, t_probs):

	""" Returns Efron's psuedo R-Squared for logistic regression. 

		Args:

			t::[Numpy Array]
				Truth values

			t_probs::[Numpy Array]
				Prediction value probabilities

	"""

	return 1.0 - ( np.sum(np.power(t - t_probs, 2.0)) / np.sum(np.power((t - (np.sum(t) / float(len(t)))), 2.0)) ) 

def mcfadden_r_squared(theta, X, t):

	""" Returns McFadden's psuedo R-Squared for logistic regression 
	
		Args:
			
			theta::[Numpy Array]
				Weights/coefficients for the given logistic regression model
			
			X::[Numpy Array]
				Regressor matrix

			t::[Numpy Array]
				Truth values corresponding to regressor matrix


	"""

	# Based on code from https://datascience.oneoffcoder.com/psuedo-r-squared-logistic-regression.html

	# Compute full log likelihood

	score = np.dot(X, theta)
	score = score.reshape(1, X.shape[0])
	
	full_log_likelihood = np.sum(-np.log(1 + np.exp(score))) + np.sum(t * score)

	# Compute null log likelihood

	z = list()

	for i, theta in enumerate(theta.reshape(1, X.shape[1])[0]):

		if(i == 0):
			z.append(theta)
		else:
			z.append(0.0)

	z = np.array(z)
	z = z.reshape(X.shape[1], 1)
	
	score = np.dot(X, z)
	score = score.reshape(1, X.shape[0])
	
	null_log_likelihood = np.sum(-np.log(1 + np.exp(score))) + np.sum(t * score)

	return 1.0 - (full_log_likelihood / null_log_likelihood)

def mcfadden_adjusted_rsquare(w, X, y):

	""" """

	score = np.dot(X, w).reshape(1, X.shape[0])
	full_log_likelihood = np.sum(-np.log(1 + np.exp(score))) + np.sum(y * score)

	z = np.array([w if i == 0 else 0.0 for i, w in enumerate(w.reshape(1, X.shape[1])[0])]).reshape(X.shape[1], 1)
	score = np.dot(X, z).reshape(1, X.shape[0])
	null_log_likelihood = np.sum(-np.log(1 + np.exp(score))) + np.sum(y * score)

	k = float(X.shape[1])
	return 1.0 - ((full_log_likelihood- k) / null_log_likelihood)

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

	if(verbose):
		print('prob preds:',t_probs)

	plt.scatter(x_1,t, color='tab:olive')
	plt.plot(x_1,t_probs, color='tab:cyan')
	plt.xlabel('x_1')
	plt.ylabel('t')
	#plt.show()
	plt.close()

	# SIMPLE LOGIT REGRESSION 

	data = pd.read_csv('data/pima.csv',sep=",")

	# pregnant,glucose,pressure,triceps,insulin,mass,pedigree,age,diabetes

	t = np.array(data['diabetes']) 
	x_1 = preprocessing.scale(np.array(data['mass'], dtype=np.float128)) # , dtype=np.float128 to prevent overflow
	
	X = np.array([np.ones(len(t)), x_1]).T

	model = logit_regression()
	model = logit_regression().fit(X,t)

	t_probs = model.predict_proba(X)

	if(verbose):
		print('prob preds',t_probs)

	plt.scatter(x_1, t, facecolors='none', edgecolor='tab:olive')
	x_1, t_probs = zip(*sorted(zip(x_1,t_probs))) # plot points in order
	plt.plot(x_1,t_probs, color='tab:cyan')
	plt.xlabel('glucose')
	plt.ylabel('diabetes')
	plt.savefig('figs/simple_logit.png')
	#plt.show()
	plt.close()

	# POLYNOMIAL LOGIT REGRESSION
	
	t = np.array(data['diabetes']) 
	x_1 = preprocessing.scale(np.array(data['glucose'], dtype=np.float128))
	
	X = np.array([np.ones(len(t)), x_1, np.square(x_1)]).T

	model = logit_regression()
	model = logit_regression().fit(X,t)

	t_probs = model.predict_proba(X)

	if(verbose):
		print('prob preds',t_probs)

	plt.scatter(x_1,t, color='tab:olive')
	x_1, t_probs = zip(*sorted(zip(x_1,t_probs))) # plot points in order
	plt.plot(x_1,t_probs, color='tab:cyan')
	plt.xlabel('glucose')
	plt.ylabel('diabetes')
	plt.savefig('figs/polynomial_logit.png')
	#plt.show()
	plt.close()

	# MULTIVARIATE LOGIT REGRESSION

	t = np.array(data['diabetes']) 
	x_1 = preprocessing.scale(np.array(data['glucose'], dtype=np.float128))
	x_2 = preprocessing.scale(np.array(data['mass'], dtype=np.float128))
	
	X = np.array([np.ones(len(t)), x_1, x_2]).T

	model = logit_regression()
	model = logit_regression().fit(X,t)

	t_probs = model.predict_proba(X)

	if(verbose):
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
	#plt.show()
	plt.close()

	# CLASSIFICATION

	# make train and test datasets

	train_test_split('data/pima.csv','pima-train.csv','pima-test.csv')

	train_data = pd.read_csv('data/pima-train.csv', sep=",") 
	test_data = pd.read_csv('data/pima-test.csv', sep=",")

	# graph true classifications

	sns.pairplot(test_data, hue="diabetes", palette=['lightcoral', 'skyblue'], plot_kws={'alpha':0.75})
	plt.savefig('figs/true-pima-pairplot.png')
	#plt.show()
	plt.close()

	# generate predicted classifications

	t = np.array(train_data['diabetes'])
	
	train_data = train_data.drop(['diabetes'], axis=1)

	x_1 = np.ones((len(t),1))
	x_n = preprocessing.scale(np.array(train_data))	
	X = np.hstack((x_1,x_n)) 
	
	model = logit_regression()
	model = logit_regression().fit(X,t)

	t = np.array(test_data['diabetes'])

	test_data = test_data.drop(['diabetes'], axis=1)

	x_1 = np.ones((len(t),1))
	x_n = preprocessing.scale(np.array(test_data))	
	X = np.hstack((x_1,x_n)) 

	t_hat = model.predict(X)
	t_probs = model.predict_proba(X)

	# graph predicted classifications

	test_data_with_pred_classifications = pd.read_csv('data/pima-test.csv', sep=",")
	test_data_with_pred_classifications['diabetes'] = t_hat

	sns.pairplot(test_data_with_pred_classifications, hue="diabetes", palette=['lightcoral', 'skyblue'], plot_kws={'alpha':0.75})
	plt.savefig('figs/pred-pima-pairplot.png')
	#plt.show()
	plt.close()

	error_rate = compute_classification_error_rate(t,t_hat)

	if(verbose):
		print('error rate:', error_rate)
		print('McFadden R-Squared:',mcfadden_r_squared(model.coef_, X, t))
		print('Efron R-Squared:',efron_r_squared(t,t_probs))




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

def train_test_split(data, train_filename, test_filename):
	
	""" Splits csv file into a train and test csv files
		
		Args:

			data::[String]
				Path to csv file to split

			train_filename::[String]
				Name for train csv file

			test_filename::[String]
				Name for test csv file


	"""

	df = pd.read_csv(data) 
		
	msk = np.random.rand(len(df)) < 0.8
	train = df[msk]
	test = df[~msk]

	train.to_csv(r'./data/' + train_filename, index=False)
	test.to_csv(r'./data/' + test_filename, index=False)

if __name__ == '__main__':
	
	#ols_driver()
	#ridge_driver()
	logit_driver()
	#lasso_driver()
