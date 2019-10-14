# evaluation for classification
# accuracy and confusion metrix

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

dataset = load_digits()
X, y = dataset.data, dataset.target

for class_name, class_count in zip(dataset.target_names, np.bincount(dataset.target)):
    print(class_name, class_count)
    
# creating a dataset with imbalanced binary class:

y_binary_imbalanced = y.copy()
y_binary_imbalanced[y_binary_imbalanced != 1 ] = 0

print('Original labels:\t', y[1:30])
print('New binary labels:\t', y_binary_imbalanced[1:30])

# Negative class (0) is the most frequent class
np.bincount(y_binary_imbalanced)

X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state = 0)

# Accuracy of Support Vector Machine Classifier
from sklearn.svm import SVC
svm = SVC(kernel = 'rbf', C=1).fit(X_train, y_train)
svm.score(X_test, y_test)

svm = SVC(kernel ='linear',C=1).fit(X_train, y_train)
svm.score(X_test, y_test)

# Dummy Classifier

from sklearn.dummy import DummyClassifier

dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
y_dummy_predictions = dummy_majority.predict(X_test)
y_dummy_predicitons
dummy_majority.score(X_test,y_test)



# Confusion matrices
# Binary (two-class) confusion matrix

from sklearn.metrics import confusion_matrix
# Negative class (0) is most frequent
dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
y_majority_predicted = dummy_majority.predict(X_test)
confusion = confusion_matrix(y_test, y_majority_predicted)

print('Most frequent class (dummy classifier)\n', confusion)

# produces random predictions w/ same class proportion as training set
dummy_classprop = DummyClassifier(strategy='stratified').fit(X_train, y_train)
y_classprop_predicted = dummy_classprop.predict(X_test)
confusion = confusion_matrix(y_test, y_classprop_predicted)

print('Random class-proportional prediction (dummy classifier)\n', confusion)

#SVM
svm = SVC(kernel='linear', C=1).fit(X_train, y_train)
svm_predicted = svm.predict(X_test)
confusion = confusion_matrix(y_test, svm_predicted)

print('Support vector machine classifier (linear kernel, C=1)\n', confusion)

# Logistic Regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression().fit(X_train, y_train)
lr_predicted = lr.predict(X_test)
confusion = confusion_matrix(y_test, lr_predicted)

print('Logistic regression classifier (default settings)\n', confusion)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
tree_predicted = dt.predict(X_test)
confusion = confusion_matrix(y_test, tree_predicted)

print('Decision tree classifier (max_depth = 2)\n', confusion)


# Evaluation metrics for binary classification
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score

print('Accuracy: {:.2f}'.format(accuracy_score(y_test, tree_predicted)))
print('Precision: {:.2f}'.format(precision_score(y_test, tree_predicted)))
print('Recall: {:.2f}'.format(recall_score(y_test, tree_predicted)))
print('F1: {:.2f}'.format(f1_score(y_test, tree_predicted)))

# combined report with all above metrices

from sklearn.metrics import classification_report

print('Random class-proportional (dummy)\n', 
      classification_report(y_test, y_classprop_predicted, target_names=['not 1', '1']))
print('SVM\n', 
      classification_report(y_test, svm_predicted, target_names = ['not 1', '1']))
print('Logistic regression\n', 
      classification_report(y_test, lr_predicted, target_names = ['not 1', '1']))
print('Decision tree\n', 
      classification_report(y_test, tree_predicted, target_names = ['not 1', '1']))


# Decision Functions

X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state = 0 )
y_score_lr = lr.fit(X_train,y_train).decision_function(X_test)
y_score_list = list(zip(y_test[0:20],y_score_lr[0:20]))
y_score_list

# Predicted Probability 

X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state =0)
y_proba_lr = lr.fit(X_train, y_train).predict_proba(X_test)
y_prob_list = list(zip(y_test[0:20], y_proba_lr[0:20,1]))

y_prob_list


#precision-recall curves

from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_score_lr)


# ROC curves, Area-Under-Curve(AUC)
from sklearn.metrics import roc_curve, auc
X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state = 0)

y_score_lr = lr.fit(X_train, y_train).decision_function(X_test)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_score_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)


# Evaluation measures for multi-class classification

dataset = load_digits()
X, y = dataset.data, dataset.target
X_train_mc, X_test_mc, y_train_mc, y_test_mc = train_test_split(X, y, random_state = 0)

svm = SVC(kernel = 'linear').fit(X_train_mc, y_train_mc)
svm_predicted_mc = svm.predict(X_test_mc)
confusion_mc = confusion_matrix(y_test_mc, svm_predicted_mc)

svm = SVC(kernel = 'rbf').fit(X_train_mc, y_train_mc)
svm.predicted_mc = svm.predict(X_test_mc)
confusion_mc = confusion_matrix(y_test_mc,svm.predicted_mc)

# Multi-class classification report
print(classification_report(y_test_mc, svm.predicted_mc))


#Micro vs Macro averaged mmetrics

print('Micro-averaged precision = {:.2f} (treat instances equally)'
      .format(precision_score(y_test_mc, svm_predicted_mc, average = 'micro')))
print('Macro-averaged precision = {:.2f} (treat classes equally)'
      .format(precision_score(y_test_mc, svm_predicted_mc, average = 'macro')))

print('Micro-averaged f1 = {:.2f} (treat instances equally)'
      .format(f1_score(y_test_mc, svm_predicted_mc, average = 'micro')))
print('Macro-averaged f1 = {:.2f} (treat classes equally)'
      .format(f1_score(y_test_mc, svm_predicted_mc, average = 'macro')))


# Regression evaluation metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, r2_score

diabetes = load_diabetes()

X = diabetes.data[:,None, 6]
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

lm = LinearRegression().fit(X_train, y_train)
lm_dummy_mean = DummyRegressor(strategy = 'mean').fit(X_train, y_train)

y_predict = lm.predict(X_test)
y_predict_dummy_mean = lm_dummy_mean.predict(X_test)

print('Linear model, coefficients:',lm.coef_)

print('Mean squared error (linear model): {:.2f}'.format(mean_squared_error(y_test, y_predict)))
print('Mean squared error (dummy): {:.2f}'.format(mean_squared_error(y_test, y_predict_dummy_mean)))

print('r2_score (linear_model): {:.2f}'.format(r2_score(y_test, y_predict)))
print('r2_score (dummy): {:.2f}'.format(r2_score(y_test, y_predict_dummy_mean)))


# Model selcetion using evaluation metrics
# cross-validation example

from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_digits
from sklearn.svm import SVC

dataset = load_digits()

X, y = dataset.data, dataset.target ==1

clf = SVC(kernel = 'linear', C =1)

# accuracy is the default scoring metric
print('Cross-validation (accuracy)', cross_val_score(clf, X, y, cv =5))
#use AUC as scoring metric
print('Cross-validaton (AUC)',cross_val_score(clf, X, y, cv =5, scoring = 'roc_auc'))
# use recall as scoring metric
print('Cross-validation (recall)',cross_val_score(clf, X, y, cv =5, scoring ='recall'))


# Grid Search example
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.datasets import load_digits

dataset = load_digits()
X, y = dataset.data, dataset.target ==1

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state =0)

clf = SVC(kernel ='rbf')
grid_values = {'gamma': [0.001,0.01,0.05, 0.1, 1, 10, 100]}

#default metric to optimize over grid parameters: accuracy

grid_clf_acc = GridSearchCV(clf, param_grid = grid_values)
grid_clf_acc.fit(X_train, y_train)
y_decision_fn_scores_acc = grid_clf_acc.decision_function(X_test)

print('Grid best parameter (max. accuracy):',grid_clf_acc.best_params_)
print('Grid best score (accuracy): ', grid_clf_acc.best_score_)

# alternative metric to optimize over grid parameters: AUC
grid_clf_auc = GridSearchCV(clf, param_grid = grid_values, scoring = 'roc_auc')
grid_clf_auc.fit(X_train, y_train)
y_decision_fn_scores_auc = grid_clf_auc.decision_function(X_test) 

print('Test set AUC: ', roc_auc_score(y_test, y_decision_fn_scores_auc))
print('Grid best parameter (max. AUC): ', grid_clf_auc.best_params_)
print('Grid best score (AUC): ', grid_clf_auc.best_score_)


# Evaluation metrics supported for model selection
from sklearn.metrics.scorer import SCORERS

print(sorted(list(SCORERS.keys())))


# Two-feature classification example using the digits dataset
# Optimizing a classifier using different evaluation metrics
 # check Model Evaluation.ipyb file
    
    
# Precision-recall curve for the default SVC classifier (with balanced class weights)
  #  check Model Evaluation.ipyb file
    
