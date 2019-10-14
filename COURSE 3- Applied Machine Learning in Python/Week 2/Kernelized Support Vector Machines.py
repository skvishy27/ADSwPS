# Kernelized Support Vector Machines (SVM)

# Classification
# Using Radial Basci Function Kernel
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs

X_D2, y_D2 = make_blobs(n_samples = 100, n_features = 2, centers =8,
                        cluster_std = 1.3, random_state = 4)
y_D2 = y_D2%2

X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state = 0)
clf = SVC().fit(X_train, y_train)

print('Accuracy of RBF Kernerl on training set: {:.2f}'
      .format(clf.score(X_train, y_train)))
print('Accuracy of RBF Kernel of testing set: {:.2f}'
      .format(clf.score(X_test, y_test)))

# Using Polynomial Kernel
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs

X_D2, y_D2 = make_blobs(n_samples = 100, n_features = 2, centers =8,
                        cluster_std = 1.3, random_state = 4)
y_D2 = y_D2%2

X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state = 0)
clf = SVC(kernel = 'poly',degree =3).fit(X_train, y_train)

print('Accuracy of Poly Kernerl on training set: {:.2f}'
      .format(clf.score(X_train, y_train)))
print('Accuracy of Poly Kernel of testing set: {:.2f}'
      .format(clf.score(X_test, y_test)))

# SVM using Gamma Parameter

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs

X_D2, y_D2 = make_blobs(n_samples = 100, n_features = 2, centers =8,
                        cluster_std = 1.3, random_state = 4)
y_D2 = y_D2%2

X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state = 0)
clf = SVC(kernel ='rbf',gamma =1.0).fit(X_train, y_train)

print('Accuracy of RBF Kernerl on training set: {:.2f}'
      .format(clf.score(X_train, y_train)))
print('Accuracy of RBF Kernel of testing set: {:.2f}'
      .format(clf.score(X_test, y_test)))

# Using gamma and c parameter
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs

X_D2, y_D2 = make_blobs(n_samples = 100, n_features = 2, centers =8,
                        cluster_std = 1.3, random_state = 4)
y_D2 = y_D2%2

X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state = 0)
clf = SVC(kernel ='rbf',gamma =0.01, C =0.1).fit(X_train, y_train)

print('Accuracy of RBF Kernerl on training set: {:.2f}'
      .format(clf.score(X_train, y_train)))
print('Accuracy of RBF Kernel of testing set: {:.2f}'
      .format(clf.score(X_test, y_test)))

# Application of SVMs to real dataset:unnormalized data
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state =0)
clf = SVC(C =10).fit(X_train, y_train)
print('Accuracy of RBF-kernel SVC on training set: {:.2f}'
      .format(clf.score(X_train, y_train)))
print('Accuracy of RBF-kernel SVC on testing set: {:.2f}'
      .format(clf.score(X_test, y_test)))

# Application of SVMs to a real dataset: normalized data with feature preprocessing using minmax scaling

from sklearn.svm import SVC 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y =True)

X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = SVC(C = 10).fit(X_train_scaled, y_train)

print('RBF-kernel SVC (with MinMaxScaling) training set accuracy: {:.2f}'
      .format(clf.score(X_train_scaled, y_train)))
print('RBF-kernel SVC (with MinMaxScaling) test set accuracy: {:.2f}'
      .format(clf.score(X_test_scaled, y_test)))

                                                    
