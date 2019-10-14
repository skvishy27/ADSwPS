# Neural Networks

# Neural networks: Classification

# Synthetic dataset 1: single hidden layer
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

X_D2, y_D2 = make_blobs(n_samples = 100, n_features = 2,
                       centers = 8, cluster_std = 1.3,
                       random_state = 4)
y_D2 = y_D2 % 2

X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state = 0)
nnclf = MLPClassifier(hidden_layer_sizes = 10, solver ='lbfgs',
                     random_state = 0).fit(X_train, y_train)
train_score = nnclf.score(X_train, y_train)
test_score = nnclf.score(X_test, y_test)
print('Train Score: {:.2f}\nTest Score: {:.2f}'
     .format(train_score, test_score))


# synthetic dataset 1: two hidden layes
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

X_D2, y_D2 = make_blobs(n_samples = 100, n_features = 2,
                       centers = 8, cluster_std = 1.3,
                       random_state = 4)
y_D2 = y_D2 % 2

X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state = 0)
nnclf = MLPClassifier(hidden_layer_sizes = [10,10], solver = 'lbfgs',
                     random_state = 0).fit(X_train, y_train)
train_score = nnclf.score(X_train, y_train)
test_score = nnclf.score(X_test, y_test)
print('Train Score: {:.2f}\nTest Score: {:.2f}'
     .format(train_score, test_score))

# Regularization parameter: alpha
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

X_D2, y_D2 = make_blobs(n_samples = 100, n_features = 2,
                       centers = 8, cluster_std = 1.3,
                       random_state = 4)
y_D2 = y_D2 % 2

X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state = 0)
nnclf = MLPClassifier(solver = 'lbfgs',activation = 'tanh',
                     alpha = 0.1, hidden_layer_sizes = [100,100],
                     random_state = 0).fit(X_train, y_train)
train_score = nnclf.score(X_train, y_train)
test_score = nnclf.score(X_test, y_test)
print('Train Score: {:.2f}\nTest Score: {:.2f}'
     .format(train_score, test_score))

# The effect of different choices of activation function
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

X_D2, y_D2 = make_blobs(n_samples = 100, n_features = 2,
                       centers = 8, cluster_std = 1.3,
                       random_state = 4)
y_D2 = y_D2 % 2

X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state = 0)
nnclf = MLPClassifier(solver='lbfgs', activation = 'relu',
                         alpha = 0.1, hidden_layer_sizes = [10, 10],
                         random_state = 0).fit(X_train, y_train)
train_score = nnclf.score(X_train, y_train)
test_score = nnclf.score(X_test, y_test)
print('Train Score: {:.2f}\nTest Score: {:.2f}'
     .format(train_score, test_score))


# Neural networks: Regression
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np


X_R1, y_R1 = make_regression(n_samples = 100, n_features = 1,
                            n_informative=1, bias = 150.0,
                            noise = 30, random_state=0)

X_predict_input = np.linspace(-3, 3, 50).reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X_R1[0::5], y_R1[0::5], random_state = 0)
mlpreg = MLPRegressor(hidden_layer_sizes = [100,100], activation = 'relu',
                     alpha = 1.0,solver ='lbfgs').fit(X_train, y_train)
y_predict_output = mlpreg.predict(X_predict_input)
# check the notebook for graph



# Application to real-world dataset for classification
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

scaler = MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state =0)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = MLPClassifier(hidden_layer_sizes = [100,100], alpha = 5.0,
                   random_state = 0,solver ='lbfgs').fit(X_train_scaled,y_train)

print('Breast cancer dataset')
print('Accuracy of NN classifier on training set: {:.2f}'
     .format(clf.score(X_train_scaled, y_train)))
print('Accuracy of NN classifier on test set: {:.2f}'
     .format(clf.score(X_test_scaled, y_test)))
