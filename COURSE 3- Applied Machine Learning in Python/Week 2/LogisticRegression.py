# Real Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

fruits = pd.read_table('fruit_data_with_colors.txt')

X_fruits = fruits[['height','width']]
y_fruits = fruits['fruit_label'] ==1

X_train, X_test, y_train, y_test = train_test_split(X_fruits.values, y_fruits.values, random_state=0)

clf = LogisticRegression(C=100).fit(X_train,y_train)

h = 6
w = 8
print('A fruit with height {} and width {} is predicted to be: {}'
     .format(h,w, ['not an apple', 'an apple'][clf.predict([[h,w]])[0]]))

print('Accuracy of Logistic Regression classifier on training set : {:.2f}'.
      format(clf.score(X_train,y_train)))
print('Accuracy of Logistic Regression classifier on test set: {:.2f}'.
      format(clf.score(X_test,y_test)))

#Synthetic Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
X_C2, y_C2 = make_classification(n_samples = 100,n_features = 2,
                                 n_redundant = 0,n_informative =2,
                                 n_clusters_per_class =1,flip_y=0.1,
                                 class_sep=0.5,random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2, random_state = 0)
clf = LogisticRegression().fit(X_train,y_train)

print('Accuracy of Logistic Regression classifier o training set :{:.2f}'
      .format(clf.score(X_train,y_train)))
print('Accuracy of Logistic Regression on test set : {:.2f}'
      .format(clf.score(X_test,y_test)))

# Application on Real Dataset

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_cancer, y_cancer = load_breast_cancer(return_X_y = True)

X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)

clf = LogisticRegression().fit(X_train, y_train)

print('Accuracy of Logistic Regression on training set :{:.2f}'
      .format(clf.score(X_train,y_train)))
print('Accuracy of Logistic Regression on testing set :{:.2f}'
      .format(clf.score(X_test,y_test)))
