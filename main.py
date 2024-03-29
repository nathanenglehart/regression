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
from lib.gd.probit import probit_regression

plt.style.use('seaborn-poster')	

verbose = True


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

	return 1 - ((((t-t_hat)**2).sum()) / (((t-t_bar)**2).sum()))

def adj_r_squared(t,t_hat,m):

	""" Returns adjusted R-Squared for model with given truth values and prediction values.

		Args:
				
			t::[Numpy Array]
				Truth values

			t_hat::[Numpy Array]
				Prediction values
			
			m::[Integer]
				Number of features in the dataset
	
	"""

	n = len(t)
	return 1 - ((1 - r_squared(t,t_hat)) * ((n-1) / (n-m)))

def efron_r_squared(t, t_probs):

	""" Returns Efron's psuedo R-Squared for logistic regression. 

		Args:

			t::[Numpy Array]
				Truth values

			t_probs::[Numpy Array]
				Prediction value probabilities

	"""

	return 1.0 - ( np.sum(np.power(t - t_probs, 2.0)) / np.sum(np.power((t - (np.sum(t) / float(len(t)))), 2.0)) ) 

def mcfadden_r_squared(theta, X, t, model):

	""" Returns McFadden's psuedo R-Squared for logistic regression 
	
		Args:
			
			theta::[Numpy Array]
				Weights/coefficients for the given logistic regression model
			
			X::[Numpy Array]
				Regressor matrix

			t::[Numpy Array]
				Truth values corresponding to regressor matrix

	"""

	L_ul = model.log_likelihood(X,t,theta)
	theta_0 = np.zeros(theta.size)
	theta_0[0] = theta[0]
	L_0 = model.log_likelihood(X, t, theta_0)

	return 1 - (L_ul / L_0)

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

	if(verbose):
		print('R-Squared:', r_squared(t,t_hat))

def logit_driver():
	
	# DRIVER FOR LOGIT REGRESSION EXAMPLE
	# UNCOMMENT plt.show FOR VISUALIZATIONS

	# (VERY) SIMPLE LOGIT REGRESSION 

	x_1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
	t = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
	
	X = np.array([np.ones(len(t)),x_1]).T
	
	model = logit_regression()
	model = logit_regression().fit(X,t)
	coef = model.coef_

	t_probs = model.predict_proba(X)

	if(verbose):
		print('prob preds:',t_probs)
		print('McFadden R-Squared:',mcfadden_r_squared(model.coef_, X, t, model))
		print('Efron R-Squared:',efron_r_squared(t,t_probs))
		print('log likelihood',model.log_likelihood(X,t,coef))

	plt.scatter(x_1,t, color='tab:olive')
	plt.plot(x_1,t_probs, color='tab:cyan')
	plt.xlabel('x_1')
	plt.ylabel('t')
	plt.show()
	plt.close()

	# SIMPLE LOGIT REGRESSION 

	data = pd.read_csv('data/pima.csv',sep=",")

	# pregnant,glucose,pressure,triceps,insulin,mass,pedigree,age,diabetes

	t = np.array(data['diabetes']) 
	x_1 = preprocessing.scale(np.array(data['mass'], dtype=np.float128)) # , dtype=np.float128 to prevent overflow
	
	X = np.array([np.ones(len(t)), x_1]).T

	model = logit_regression()
	model = logit_regression().fit(X,t)
	coef = model.coef_

	t_probs = model.predict_proba(X)

	if(verbose):
		print('prob preds',t_probs)
		print('McFadden R-Squared:',mcfadden_r_squared(model.coef_, X, t, model))
		print('Efron R-Squared:',efron_r_squared(t,t_probs))
		print('log likelihood',model.log_likelihood(X,t,coef))

	plt.scatter(x_1, t, facecolors='none', edgecolor='tab:olive')
	x_1, t_probs = zip(*sorted(zip(x_1,t_probs))) # plot points in order
	plt.plot(x_1,t_probs, color='tab:cyan')
	plt.xlabel('glucose')
	plt.ylabel('diabetes')
	plt.savefig('figs/simple_logit.png')
	plt.show()
	plt.close()

	# MULTIVARIATE LOGIT REGRESSION

	t = np.array(data['diabetes']) 
	x_1 = preprocessing.scale(np.array(data['glucose'], dtype=np.float128))
	x_2 = preprocessing.scale(np.array(data['mass'], dtype=np.float128))
	
	X = np.array([np.ones(len(t)), x_1, x_2]).T

	model = logit_regression()
	model = logit_regression().fit(X,t)
	coef = model.coef_

	t_probs = model.predict_proba(X)

	if(verbose):
		print('prob preds',t_probs)
		print('McFadden R-Squared:',mcfadden_r_squared(model.coef_, X, t, model))
		print('Efron R-Squared:',efron_r_squared(t,t_probs))
		print('log likelihood',model.log_likelihood(X,t,coef))

	x_pts = np.linspace(x_1.min(), x_1.max(), 30)
	y_pts = np.linspace(x_2.min(), x_2.max(), 30)
	x_pairs, y_pairs = np.meshgrid(x_pts,y_pts)

	# Get values for all ordered pairs in set using model

	z = 1 / (-(1 + np.e ** (model.coef_[0] + model.coef_[1] * x_pairs + model.coef_[2] * y_pairs)))

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
		print('McFadden R-Squared:',mcfadden_r_squared(model.coef_, X, t, model))
		print('Efron R-Squared:',efron_r_squared(t,t_probs))

def probit_driver():
	
	# DRIVER FOR PROBIT REGRESSION EXAMPLE
	# UNCOMMENT plt.show FOR VISUALIZATIONS

	# (VERY) SIMPLE PROBIT REGRESSION 

	x_1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
	t = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
	
	X = np.array([np.ones(len(t)),x_1]).T
	
	model = probit_regression()
	model = probit_regression().fit(X,t)
	coef = model.coef_

	t_probs = model.predict_proba(X)

	if(verbose):
		print('prob preds:',t_probs)
		print('McFadden R-Squared:',mcfadden_r_squared(coef, X, t, probit_regression()))
		print('Efron R-Squared:',efron_r_squared(t,t_probs))
		print('log likelihood',model.log_likelihood(X,t,coef))

	plt.scatter(x_1,t, color='tab:olive')
	plt.plot(x_1,t_probs, color='tab:cyan')
	plt.xlabel('x_1')
	plt.ylabel('t')
	plt.show()
	plt.close()

	# SIMPLE PROBIT REGRESSION 

	data = pd.read_csv('data/pima.csv',sep=",")

	# pregnant,glucose,pressure,triceps,insulin,mass,pedigree,age,diabetes

	t = np.array(data['diabetes']) 
	x_1 = preprocessing.scale(np.array(data['mass'], dtype=np.float128)) # , dtype=np.float128 to prevent overflow
	
	X = np.array([np.ones(len(t)), x_1]).T

	model = probit_regression()
	model = probit_regression().fit(X,t)
	coef = model.coef_

	t_probs = model.predict_proba(X)

	if(verbose):
		print('prob preds',t_probs)
		print('McFadden R-Squared:',mcfadden_r_squared(coef, X, t, probit_regression()))
		print('Efron R-Squared:',efron_r_squared(t,t_probs))
		print('log likelihood',model.log_likelihood(X,t,coef))

	plt.scatter(x_1, t, facecolors='none', edgecolor='tab:olive')
	x_1, t_probs = zip(*sorted(zip(x_1,t_probs))) # plot points in order
	plt.plot(x_1,t_probs, color='tab:cyan')
	plt.xlabel('glucose')
	plt.ylabel('diabetes')
	plt.savefig('figs/simple_probit.png')
	plt.show()
	plt.close()

# MULTIVARIATE PROBIT REGRESSION

	t = np.array(data['diabetes']) 
	x_1 = preprocessing.scale(np.array(data['glucose'], dtype=np.float128))
	x_2 = preprocessing.scale(np.array(data['mass'], dtype=np.float128))
	
	X = np.array([np.ones(len(t)), x_1, x_2]).T

	model = probit_regression()
	model = probit_regression().fit(X,t)
	coef = model.coef_

	t_probs = model.predict_proba(X)

	if(verbose):
		print('prob preds',t_probs)
		print('McFadden R-Squared:',mcfadden_r_squared(model.coef_, X, t, model))
		print('Efron R-Squared:',efron_r_squared(t,t_probs))
		print('log likelihood',model.log_likelihood(X,t,coef))

	x_pts = np.linspace(x_1.min(), x_1.max(), 30)
	y_pts = np.linspace(x_2.min(), x_2.max(), 30)
	x_pairs, y_pairs = np.meshgrid(x_pts,y_pts)

	# Get values for all ordered pairs in set using model

	z = 1 - model.Phi(model.coef_[0] + model.coef_[1] * x_pairs + model.coef_[2] * y_pairs) #z = 1 / (1 + np.e ** (-(model.coef_[0] + model.coef_[1] * x_pairs + model.coef_[2] * y_pairs)))

	# Graph

	fig = plt.figure(figsize = (100,100))
	ax = plt.axes(projection='3d')
	ax.plot_surface(x_pairs,y_pairs,z, rstride=1, cstride=1, color='tab:cyan', alpha=0.4, antialiased=False)
	ax.scatter(x_1,x_2,t, c = 'tab:olive')
	ax.set_ylabel('mass')
	ax.set_title('pima', fontsize=20)
	plt.xlabel('\n\n\nglucose', fontsize=18)
	plt.ylabel('\n\n\nmass', fontsize=16)
	plt.savefig('figs/multivariate_probit.png')
	plt.show()
	plt.close()

	data = pd.read_csv('data/catanstats.csv')

	# CATAN EXAMPLE PROBIT REGRESSION 

	# Create bernoulli rv which is 1 when a player won and is 0 when a player lost

	# https://www.kaggle.com/datasets/lumins/settlers-of-catan-games
	# gameNum,player,points,me,2,3,4,5,6,7,8,9,10,11,12,settlement1,,,,,,settlement2,,,,,,production,tradeGain,robberCardsGain,totalGain,tradeLoss,robberCardsLoss,tribute,totalLoss,totalAvailable

	# production - total cards gained from settlements and cities during game
	# tradeGain - total cards gained from peer AND bank trades during game
	# robberCardsGain - total cards gained from stealing with the robber, plus cards gained with non-knight development cards. A road building card is +4 resources.
	# totalGain - sum of previous 3 columns.
	# tradeLoss - total cards lost from peer AND bank trades during game
	# robberCardsLoss - total cards lost from robbers, knights, and other players' monopoly cards
	# tribute - total cards lost when player had to discard on a 7 roll (separate from previous column.)
	# totalLoss - sum of previous 3 columns.
	# totalAvailable - totalGain minus totalLoss.

	data['won'] = 0
	for i in range(len(data)):
		if(data.loc[i, 'points'] >= 10):
			data.loc[i,'won'] = 1
	print(data['won'])

	t = np.array(data['won']) 
	x_1 = preprocessing.scale(np.array(data['totalAvailable'], dtype=np.float128)) # , dtype=np.float128 to prevent overflow
	
	X = np.array([np.ones(len(t)), x_1]).T

	model = probit_regression()
	model = probit_regression().fit(X,t)
	coef = model.coef_

	t_probs = model.predict_proba(X)

	if(verbose):
		print('prob preds',t_probs)
		print('McFadden R-Squared:',mcfadden_r_squared(coef, X, t, probit_regression()))
		print('Efron R-Squared:',efron_r_squared(t,t_probs))
		print('log likelihood',model.log_likelihood(X,t,coef))

	plt.scatter(x_1, t, facecolors='none', edgecolor='tab:olive')
	x_1, t_probs = zip(*sorted(zip(x_1,t_probs))) # plot points in order
	plt.plot(x_1,t_probs, color='tab:cyan')
	plt.xlabel('totalAvailable')
	plt.ylabel('won')
	plt.savefig('figs/catan_probit.png')
	plt.show()
	plt.close()

	x_1 = preprocessing.scale(np.array(data['totalLoss'], dtype=np.float128))
	x_2 = preprocessing.scale(np.array(data['totalGain'], dtype=np.float128))
	
	X = np.array([np.ones(len(t)), x_1, x_2]).T

	model = probit_regression()
	model = probit_regression().fit(X,t)
	coef = model.coef_

	t_probs = model.predict_proba(X)

	if(verbose):
		print('prob preds',t_probs)
		print('McFadden R-Squared:',mcfadden_r_squared(model.coef_, X, t, model))
		print('Efron R-Squared:',efron_r_squared(t,t_probs))
		print('log likelihood',model.log_likelihood(X,t,coef))

	x_pts = np.linspace(x_1.min(), x_1.max(), 30)
	y_pts = np.linspace(x_2.min(), x_2.max(), 30)
	x_pairs, y_pairs = np.meshgrid(x_pts,y_pts)

	# Get values for all ordered pairs in set using model

	z = 1 - model.Phi(model.coef_[0] + model.coef_[1] * x_pairs + model.coef_[2] * y_pairs) #z = 1 / (1 + np.e ** (-(model.coef_[0] + model.coef_[1] * x_pairs + model.coef_[2] * y_pairs)))

	# Graph

	fig = plt.figure(figsize = (100,100))
	ax = plt.axes(projection='3d')
	ax.plot_surface(x_pairs,y_pairs,z, rstride=1, cstride=1, color='tab:cyan', alpha=0.4, antialiased=False)
	ax.scatter(x_1,x_2,t, c = 'tab:olive')
	ax.set_ylabel('won')
	ax.set_title('catan', fontsize=20)
	plt.xlabel('\n\n\ntotal_gain', fontsize=18)
	plt.ylabel('\n\n\ntotal_loss', fontsize=16)
	plt.savefig('figs/catan_probit_multi.png')
	plt.show()
	plt.close()

"""
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

	if(verbose):
		print('R-Squared:',r_squared(t,t_hat))
		print('Adjusted R-Squared:',adj_r_squared(t,t_hat,X.shape[1]-1)) # - 1 to account for intercept column

	# POLYNOMIAL OLS REGRESSION 
	
	t = np.array(data['mpg'])
	x = np.array(data['weight'])

	X = np.array([np.ones(len(t)), x, np.square(x)]).T

	model = ols_regression()
	model = ols_regression().fit(X,t)

	t_hat = model.predict(X)

	r2 = r_squared(t,t_hat)
	adj_r2 = adj_r_squared(t,t_hat,X.shape[1]-1)

	plt.scatter(x, t, color='g')
	x, t_hat = zip(*sorted(zip(x,t_hat))) # plot points in order
	plt.plot(x, t_hat, color='k')
	plt.xlabel('weight')
	plt.ylabel('mpg')
	plt.savefig('figs/polynomial_ols.png')
	plt.show()

	if(verbose):
		print('R-Squared:',r2)
		print('Adjusted R-Squared',adj_r2)

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

	if(verbose):
		print('R-Squared:',r_squared(t,t_hat))
		print('Adjusted R-Squared:',adj_r_squared(t,t_hat,X.shape[1]-1))

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

	if(verbose):
		print('R-Squared:',r_squared(t,t_hat))
		print('Adjusted R-Squared:',adj_r_squared(t,t_hat,X.shape[1]-1))
"""

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

	if(verbose):
		print('R-Squared:',r_squared(t,t_hat))

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

	if(verbose):
		print('R-Squared:',r_squared(t,t_hat))

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

	if(verbose):
		print('R-Squared:',r_squared(t,t_hat))

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

	if(verbose):
		print('R-Squared:',r_squared(t,t_hat))

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

def ols_simulation():

	x1 = np.random.normal(size=1000, loc=1, scale=2)
	 
	theta0 = 1
	theta1 = 2
	 
	epsilon = np.random.normal(size=1000, loc=0, scale=1) # must be standard normal
	 
	t = theta0 + theta1 * x1 + epsilon
	 
	data = pd.DataFrame({'t': t, 'x1': x1})
	print(data)

	# SIMPLE OLS REGRESSION
	
	t = np.array(data['t'])
	x1 = np.array(data['x1'])

	X = np.array([np.ones(len(t)), x1]).T

	model = ols_regression()
	model = ols_regression().fit(X,t)

	t_hat = model.predict(X)
	
	plt.scatter(np.array(data['x1']), np.array(data['t']), color='g')
	plt.plot(np.array(data['x1']), t_hat, color='k')
	plt.xlabel('x1')
	plt.ylabel('t')
	#plt.savefig('figs/simple_ols.png')
	plt.show()



if __name__ == '__main__':
	
	#ols_driver()
	#ridge_driver()
	#lasso_driver()
	#logit_driver()
	#probit_driver()
	simulation()

