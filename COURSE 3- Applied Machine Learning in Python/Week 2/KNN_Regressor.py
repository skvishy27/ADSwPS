from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import make_regression

X_R1, y_R1 = make_regression(n_samples = 100, n_features = 1,
                             n_informative =1, bias = 150.0,
                             noise = 30, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1, random_state=0)
knnreg = KNeighborsRegressor(n_neighbors = 5)
knnreg.fit(X_train, y_train)
print(knnreg.predict(X_test))
print('R-squared test score:{:.3f}'.format(knnreg.score(X_test,y_test)))