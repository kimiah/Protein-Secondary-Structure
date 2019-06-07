# %%
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix

mlp = MLPClassifier(verbose=True, hidden_layer_sizes=(128, ))

mlp.fit(X_train, y_train)

# %%
y_pred = mlp.predict(X_test)

print(mlp.score(X_train, y_train))
print(mlp.score(X_test, y_test))

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# %%

kf = KFold(n_splits=5)
clf = MLPClassifier(hidden_layer_sizes=(128, ))

for train_indices, val_indices in kf.split(X_train):
    clf.fit(X_train[train_indices], y_train[train_indices])
    print(clf.score(X_train[val_indices], y_train[val_indices]))

