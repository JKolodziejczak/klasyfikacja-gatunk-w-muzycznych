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


def genre_classification():
    general_path = 'data'
    data = pd.read_csv(f'{general_path}/features_30_sec_with_noise.csv')
    # data = pd.read_csv(f'{general_path}/features_30_sec_original.csv')
    data = data.iloc[0:, 1:]
    data.head()

    y = data['label']  # genre variable.
    X = data.loc[:, data.columns != 'label']  # select all columns but not the labels

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
        print(confusion_matrix(y_test, preds))
        print('Accuracy', title, ':', round(accuracy_score(y_test, preds), 5))

    # Cross Gradient Booster
    xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05)
    model_assess(xgb, "Cross Gradient Booster")

    # Random Forest
    rforest = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)
    model_assess(rforest, "Random Forest")

    # KNN
    knn = KNeighborsClassifier(n_neighbors=19)
    model_assess(knn, "KNN")

    # Support Vector Machine
    svm = SVC(decision_function_shape="ovo")
    model_assess(svm, "Support Vector Machine")

    # Logistic Regression
    lg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    model_assess(lg, "Logistic Regression")

    # Neural Nets
    nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5000, 10), random_state=1)
    model_assess(nn, "Neural Nets")

    # Stochastic Gradient Descent
    sgd = SGDClassifier(max_iter=5000, random_state=0)
    model_assess(sgd, "Stochastic Gradient Descent")

    # Decission trees
    tree = DecisionTreeClassifier()
    model_assess(tree, "Decision trees")

    # Naive Bayes
    nb = GaussianNB()
    model_assess(nb, "Naive Bayes")

    ##########
    # Final model
    xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05)
    xgb.fit(X_train, y_train)

    preds2 = xgb.predict(X_test)

    print('Accuracy', ':', round(accuracy_score(y_test, preds2), 5), '\n')

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
