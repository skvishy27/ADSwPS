# Ridge Regression without normalization

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

import pandas as pd

crime = pd.read_table('CommViolPredUnnormalizedData.txt', sep =',',
                      na_values='?')
columns_to_keep = [5,6] + list(range(11,26)) + list(range(32,103)) +[145]
crime = crime.iloc[:,columns_to_keep].dropna()
X_crime = crime.iloc[:,range(0,88)]
y_crime = crime['ViolentCrimesPerPop']

X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime,random_state=0)
linridge = Ridge(alpha=20.0).fit(X_train, y_train)

print('ridge regression linear model intercept:{}'.format(linridge.intercept_))
print('ridge regression linear model coeff:{}'.format(linridge.coef_))
print('R-Squared score (training): {:.3f}'.format(linridge.score(X_train, y_train)))
print('R-Squared score (testing) :{:.3f}'.format(linridge.score(X_test, y_test)))


# Ridge Regression with feature normlization 
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

crime = pd.read_table('CommViolPredUnnormalizedData.txt', sep =',',
                      na_values='?')
columns_to_keep = [5,6] + list(range(11,26)) + list(range(32,103)) +[145]
crime = crime.iloc[:,columns_to_keep].dropna()
X_crime = crime.iloc[:,range(0,88)]
y_crime = crime['ViolentCrimesPerPop']

X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime,random_state=0)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linridge = Ridge(alpha=20.0).fit(X_train_scaled,y_train)

print('ridge regression linear model intercept:{}'.format(linridge.intercept_))
print('ridge regression linear model coeff:{}'.format(linridge.coef_))
print('R-Squared score (training): {:.3f}'.format(linridge.score(X_train_scaled, y_train)))
print('R-Squared score (testing) :{:.3f}'.format(linridge.score(X_test_scaled, y_test)))