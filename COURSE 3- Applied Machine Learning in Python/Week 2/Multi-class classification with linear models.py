# Multi-class Classification with linear models

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import pandas as pd

fruits = pd.read_table('fruit_data_with_colors.txt')

X_fruits = fruits[['height','width']]
y_fruits = fruits['fruit_label']

X_train, X_test, y_train, y_test = train_test_split(X_fruits, y_fruits, random_state=0)

clf = LinearSVC(C=5, random_state=67).fit(X_train, y_train)
print('coefficients:\n',clf.coef_)
print('Intercepts:\n',clf.intercept_)
