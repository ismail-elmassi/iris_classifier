from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def train_evaluate_iris():

    # load iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Create a RF Classifier
    clf = RandomForestClassifier(n_estimators=100)

    #split data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Train the RF model using the training sets
    clf.fit(X_train, y_train)

    # prediction of test data
    y_pred = clf.predict(X_test)

    # caluclate and diplay model performane
    print('accuracy= ', round(accuracy_score(y_test, y_pred), 2))
    print("confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("classification metrics:")
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    train_evaluate_iris()


