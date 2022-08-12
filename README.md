# Regression Implementations

Implementations of OLS, ridge, lasso, and logit regression classes using Python. 

## Author

Nathan Englehart (Summer, 2022)

## Usage

Open `main.py` and uncomment the drivers for each regression you would like to test in the main function. For instance to run the lasso regression driver:

```python
if __name__ == '__main__':
	
	#ols_driver()
	#ridge_driver()
	#logit_driver()
	lasso_driver()
```

## Documentation

OLS and ridge project documentation available at: <a href="https://nathanenglehart.github.io/regression/">https://nathanenglehart.github.io/regression/</a>.

Logit and lasso project documentation coming soon.

## Notes

This is a work in progress. OLS and ridge classes use their respective closed form solution. Logit and lasso classes use batch gradient descent to compute coefficients. 

## References

Rogers, Simon and Girolami, Mark. (2017). A First Course in Machine Learning Second Edition. Routledge.

Quinlan, Ross. (1983). UCI Machine Learning Repository <a href="https://archive.ics.uci.edu/ml/datasets/auto+mpg">https://archive.ics.uci.edu/ml/datasets/auto+mpg</a>. Irvine, CA: University of California, School of Information and Computer Science.

Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). Kaggle <a href="https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database">https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database</a>. San Francisco, CA.
