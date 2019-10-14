# K- Nearest Neighbors(KNN)

import pandas as pd
fruits = pd.read_table('fruit_data_with_colors.txt')
fruits.head()

X = fruits[['mass','width','height']]
y = fruits['fruit_label']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train,y_train)

knn.score(X_test,y_test)#accuracy
print("Accuracy of K-NN classifier on test set:", knn.score(X_test,y_test))

example_fruits = [[187,6.7,7.9]]
print("predicted fruit type for",example_fruits,"is",knn.predict(example_fruits))

