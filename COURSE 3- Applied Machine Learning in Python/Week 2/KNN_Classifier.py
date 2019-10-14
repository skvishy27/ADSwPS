from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification

X_C2, y_C2 = make_classification(n_samples=100, n_features=2,
                                 n_redundant=0, n_informative=2,
                                 n_clusters_per_class=1, flip_y=0.1,
                                 class_sep=0.5,random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2,
                                                    random_state=0)
clf = KNeighborsClassifier(n_neighbors=11, weights='uniform' )
clf.fit(X_train, y_train)
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
print('train_score ={0:.2f}, test_score={0:.2f}'
      .format(train_score,test_score))