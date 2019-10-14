# CrossValidation

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import numpy as np

fruits = pd.read_table('fruit_data_with_colors.txt')
X_fruits = fruits[['height','width']]
y_fruits = fruits['fruit_label']

clf = KNeighborsClassifier(n_neighbors = 5)
X = X_fruits.values
y = y_fruits.values

cv_scores = cross_val_score(clf, X, y)

print('Cross-validation scores (3-fold):',cv_scores)
print('Mean cross-validation score (3-fold): {:.3f}'
      .format(np.mean(cv_scores)))

#validation curve example
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
import pandas as pd
import numpy as np

fruits = pd.read_table('fruit_data_with_colors.txt')
X_fruits = fruits[['height','width']]
y_fruits = fruits['fruit_label']

clf = KNeighborsClassifier(n_neighbors = 5)
X = X_fruits.values
y = y_fruits.values

param_range = np.logspace(-3,3,4)
train_scores, test_scores = validation_curve(SVC(),X, y,param_name= 'gamma',
                                             param_range=param_range,cv=3)
print(train_scores)
print(test_scores)
