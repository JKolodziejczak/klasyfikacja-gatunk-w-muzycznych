from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from xgboost import plot_tree, plot_importance

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import decomposition, datasets
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


def genre_classification():
    general_path = 'data'
    #data = pd.read_csv(f'{general_path}/features_5_sec_with_noise.csv')
    data = pd.read_csv(f'{general_path}/features_5_sec_original.csv')
    data = data.iloc[0:, 1:]
    data.head()

    y = data['label']  # genre variable.
    X = data.loc[:, data.columns != 'label']  # select all columns but not the labels
    #X = data.loc[:, "length":"mfcc20_var"]  # select all columns but not the labels

    #### NORMALIZE X ####
    # Normalize so everything is on the same scale.

    cols = X.columns
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X)

    # new data frame with the new scaled data.
    X = pd.DataFrame(np_scaled, columns=cols)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print("test length "+str(len(y_test)))
    print("train length " + str(len(y_train)))

    def model_assess(model, title="Default"):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        # print(confusion_matrix(y_test, preds))
        print('Accuracy', title, ':', round(accuracy_score(y_test, preds), 5))
        return round(accuracy_score(y_test, preds), 5)

    ne = [50,100,200,300,400,500,600,700,800,900,1000]
    lr = [0.01,0.05,0.1,0.15,0.2,0.25,0.3]
    max_acc = 0
    max_ne = 0
    max_lr = 0

    # Cross Gradient Booster
    # for n_estimators in ne:
    #     for learning_rate in lr:
    #         print("ne: "+str(n_estimators)+", lr: "+str(learning_rate))
    #         xgb = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
    xgb = XGBClassifier(n_estimators=300, learning_rate=0.25)
    accuracy_xgb = model_assess(xgb, "Cross Gradient Booster")
    #         if max_acc<accuracy_xgb:
    #             max_acc = accuracy_xgb
    #             max_ne = n_estimators
    #             max_lr = learning_rate
    # print("WINNER ne: " + str(max_ne) + ", lr: " + str(max_lr)+", accuracy: "+ str(max_acc))

    # Random Forest
    rforest = RandomForestClassifier(n_estimators=500, max_depth=20)
    accuracy_rforest = model_assess(rforest, "Random Forest")

    # KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    accuracy_knn = model_assess(knn, "KNN")

    # Support Vector Machine
    svm = SVC(kernel="poly", gamma=3)
    accuracy_svm = model_assess(svm, "Support Vector Machine")

    # Logistic Regression
    lg = LogisticRegression(solver='lbfgs', multi_class='ovr', penalty='none')
    accuracy_lg = model_assess(lg, "Logistic Regression")

    # Neural Nets
    nn = MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=(1000, 100))
    accuracy_nn = model_assess(nn, "Neural Nets")

    # # Stochastic Gradient Descent
    sgd = SGDClassifier(max_iter=500, alpha=1e-4, penalty="l1", loss="log_loss")
    accuracy_sgd = model_assess(sgd, "Stochastic Gradient Descent")

    # Decission trees
    tree = DecisionTreeClassifier(criterion="entropy", max_depth=20)
    accuracy_tree = model_assess(tree, "Decision trees")

    # Naive Bayes
    nb = GaussianNB()
    accuracy_nb = model_assess(nb, "Naive Bayes")

    ##########
    # Best model
    max_accuracy = max([accuracy_nb, accuracy_tree, accuracy_nn, accuracy_sgd, accuracy_lg, accuracy_svm, accuracy_rforest, accuracy_knn, accuracy_xgb])
    xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05)
    xgb.fit(X_train, y_train)

    preds2 = xgb.predict(X_test)

    print('Accuracy', ':', max_accuracy, '\n')

    # Confusion Matrix
    confusion_matr = confusion_matrix(y_test, preds2)  # normalize = 'true'
    plt.figure(figsize=(16, 9))
    sns.heatmap(confusion_matr, cmap="Blues", annot=True,
                xticklabels=["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae",
                             "rock"],
                yticklabels=["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae",
                             "rock"]);
    plt.savefig("conf matrix")
    #plt.show()

    import eli5
    from eli5.sklearn import PermutationImportance

    perm = PermutationImportance(estimator=xgb, random_state=1)
    perm.fit(X_test, y_test)

    eli5.show_weights(estimator=perm, feature_names=X_test.columns.tolist())


def optimal_params():
    general_path = 'data'
    data = pd.read_csv(f'{general_path}/features_30_sec_original.csv')
    data = data.iloc[0:, 1:]
    data.head()

    y = data['label']  # genre variable.
    X = data.loc[:, data.columns != 'label']
    cols = X.columns
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X)

    # new data frame with the new scaled data.
    X = pd.DataFrame(np_scaled, columns=cols)
    std_slc = StandardScaler()
    pca = decomposition.PCA()
    dec_tree = tree.DecisionTreeClassifier()
    pipe = Pipeline(steps=[('std_slc', std_slc),
                           ('pca', pca),
                           ('dec_tree', dec_tree)])
    criterion = ['gini', 'entropy']
    max_depth = [2, 4, 6, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    parameters = dict(dec_tree__criterion=criterion,
                      dec_tree__max_depth=max_depth)

    clf_GS = GridSearchCV(pipe, parameters)
    clf_GS.fit(X, y)
    print('Best Criterion:', clf_GS.best_estimator_.get_params()['dec_tree__criterion'])
    print('Best max_depth:', clf_GS.best_estimator_.get_params()['dec_tree__max_depth'])
    print();
    print(clf_GS.best_estimator_.get_params()['dec_tree'])
