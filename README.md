# Machine-Learning-Algorithm-Implementation
This repository is for implementation of machine learning algorithms from scratch (using Python Numpy)
* Linear Regression (Notebook name: Linear Regeression.ipynb)
This jupyter notebook file implements linear regression algorithm using Python Numpy library. The notebook contains implementation of linear regression using Normal Equation, Batch Gradient Descent, Stochastic Gradient Descent and Newton's method.
Example of using LinearRegression class:
```python
lr1 = LinearRegression()
lr1.fit(X_train, y_train, dummy_feature_add = True)
y_test_pred = lr1.predict(X_test, dummy_feature_add = True)
r_square = lr1.r_square(y_test, y_test_pred)
```
* Locally Weighted Linear Regression: Implemented locally weighted linear regression using Batch Gradient Descent, Stochastic Gradient Descent and Normal Equations. Effect of bandwidth parameter on the predictions is also analyzed.
* Logistic Regression: Implemented Logistic Regression on Breast Cancer dataset using Batch / Stochastic gradient descent algorithms. Binary classification metrics like (accuracy, precision, recall and F1 score) has been calculated.
* Perceptron Algorithm: Implemented Perceptron algorithm for classification of synthetic data generated using Sciki-learn library.
* Softmax regression: Implemented softmax regression for multiclass classification. The dataset used for the implementation is a 3 class synthetic dataset created using Sciki-learn library. The implementation is inspired from Neural Network implementation.
* Support Vector Machine: Implemented Support Vector Machine algorithm using SMO optimization method. Implemented 'rbf', 'linear', 'polynomial' and 'sigmoid' kernels. Demonstrated capabilities of the kernels on a synthetic dataset created using Sci-kit learn.
* Gaussian Discriminant Analysis (GDA): Implemented GDA algorithm with the option of setting same /different covariance matrix for positive and negative examples. As it can be seen from the notebook that the decision boundary for same covariance case is linear (it should also be the same decision boundary as that of Logistic Regression). For different coviariance case, the decision boundary is non-linear which results into much better fit to the data.
* Naive Bayes algorithm: Implementation of Naive Bayes is to be added.
