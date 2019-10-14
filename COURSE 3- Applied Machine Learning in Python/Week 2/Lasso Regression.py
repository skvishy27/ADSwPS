#Lassso Regression with feature Normalization

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

crime = pd.read_table('CommViolPredUnnormalizedData.txt',sep=',',na_values = '?')
columns_to_keep = [5,6] + list(range(11,26)) + list(range(32,103)) +[145]
crime = crime.iloc[:,columns_to_keep].dropna()
X_crime = crime.iloc[:,range(0,88)]
y_crime = crime['ViolentCrimesPerPop']

X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime,random_state=0)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linlasso = Lasso(alpha = 2.0, max_iter = 10000).fit(X_train_scaled,y_train)

print('lasso regression linear model intercept: {}'.format(linlasso.intercept_))
print('lasso regression linear model coeff: {}'.format(linlasso.coef_))
print('R-Squared Score (training) :{:.3f}'.format(linlasso.score(X_train_scaled,y_train)))
print('R-Squared score (test) :{:.3f}'.format(linlasso.score(X_test_scaled,y_test)))
