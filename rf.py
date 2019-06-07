# %%
from time import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

rfc = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=42)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

rfc.score(X_train, y_train)
rfc.score(X_test, y_test)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)

# %%
from sklearn.metrics import precision_recall_fscore_support

precision_recall_fscore_support(y_test, y_pred)


# %%

from sklearn.model_selection import cross_val_score, GridSearchCV

depths = [int(x) for x in np.linspace(10, 110, num=11)]  # tekrari
depths.append(None)
n_estimators = [int(x) for x in np.linspace(start=200, stop=1000, num=5)]

# %%
val_acc = np.zeros((len(n_estimators), len(depths)))

for i in range(len(n_estimators)):
    for j in range(len(depths)):
        rfc = RandomForestClassifier(n_estimators=n_estimators[i],
                                     max_depth=depths[j],
                                     criterion='entropy', random_state=42)
        val_acc[i][j] = np.mean(cross_val_score(rfc, X_train, y_train, cv=5, scoring='accuracy'))
        print(val_acc[i][j])


# use a full grid over all parameters

# %%
# def report(results, n_top=3):
#     for i in range(1, n_top + 1):
#         candidates = np.flatnonzero(results['rank_test_score'] == i)
#         for candidate in candidates:
#             print("Model with rank: {0}".format(i))
#             print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
#                 results['mean_test_score'][candidate],
#                 results['std_test_score'][candidate]))
#             print("Parameters: {0}".format(results['params'][candidate]))
#             print("")
#

param_grid = {
    # "max_features": [1, 3, 10],
    # "min_samples_split": [2, 3, 10],
    "max_depth": depths,
    "n_estimators": n_estimators,
    # "bootstrap": [True, False],
    # "criterion": ["gini", "entropy"]
}

# run grid search
grid_search = GridSearchCV(rfc, param_grid=param_grid, cv=5, return_train_score=True, verbose=2)
start = time()
grid_search.fit(X_train, y_train)

# print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
#       % (time() - start, len(grid_search.cv_results_['params'])))
# report(grid_search.cv_results_)

df_gridsearch = pd.DataFrame(grid_search.cv_results_)