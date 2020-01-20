import argparse
import os
import numpy as np
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import KFold
from scipy import stats


def accuracy(cm):
    ''' Computes accuracy given confusion matrix. '''
    accurate = 0
    num_classification = 0
    for i in range(2):
        accurate += cm[i][i]
        for j in range(2):
            num_classification += cm[i][j]

    return accurate / num_classification if num_classification > 0 else 0.0

def recall(cm):
    ''' Computes recall given confusion matrix. '''
    fractions = []
    for i in range(2):
        accurate =  cm[i][i]
        num_classifications = 0
        for j in range(2):
            num_classifications += cm[i][j]
        recall_num = accurate / num_classifications if num_classifications > 0 else 0.0
        fractions.append(recall_num)

    return fractions


def precision(cm):
    ''' Computes precision given confusion matrix. '''
    fractions = []
    for i in range(2):
        accurate =  cm[i][i]
        num_classifications = 0
        for j in range(2):
            num_classifications += cm[j][i]
        precision_num = accurate / num_classifications if num_classifications > 0 else 0.0
        fractions.append(precision_num)

    return fractions


def classify(output_dir, X_train, X_test, y_train, y_test):
    ''' This function performs experiment 3.1

    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes

    Returns:
       i: int, the index of the supposed best classifier
    '''
    # Reshape y_train
    y_train = np.ravel(y_train)

    # Best classifier as decided by accuracies

    # Get k best features
    selector = SelectKBest(f_classif, 10)
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)
    indices_full = selector.get_support(indices=True)
    top_5 = set(indices_full)

    # Train the model and write relevant results to file
    accuracy_, recall_, precision_, cm = train_classifier(RandomForestClassifier, X_train, X_test, y_train,
                                                          y_test)

    with open(f"{output_dir}/classify", "w") as outf:
        outf.write(f'Results for {RandomForestClassifier}:\n')
        outf.write(f'\tAccuracy: {accuracy_:.4f}\n')
        outf.write(f'\tRecall: {[round(item, 4) for item in recall_]}\n')
        outf.write(f'\tPrecision: {[round(item, 4) for item in precision_]}\n')
        outf.write(f'\tConfusion Matrix: \n{cm}\n\n')
        outf.write(f'Top-5 at higher: {top_5}\n')

        pass


def train_k_best(output_dir, X_train, X_test, y_train, y_test):
    selector = SelectKBest(f_classif, 5)
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)
    indices_full = selector.get_support(indices=True)

    accuracy_, recall_, precision_, cm = train_classifier(RandomForestClassifier, X_train, X_test, y_train,
                                                          y_test)

    with open(f"{output_dir}/k_best", "w") as outf:
        outf.write(f'Results for {RandomForestClassifier}:\n')
        outf.write(f'\tAccuracy: {accuracy_:.4f}\n')
        outf.write(f'\tRecall: {[round(item, 4) for item in recall_]}\n')
        outf.write(f'\tPrecision: {[round(item, 4) for item in precision_]}\n')
        outf.write(f'\tConfusion Matrix: \n{cm}\n\n')

        pass


def train_classifier(classifier, X_train, X_test, y_train, y_test):
    ''' Trains the specified classifier with the given data.
        Returns the accuracy, recall, precision and confusion matrix. '''

    # Initialize the classifier with the correct arguments
    model = classifier(n_estimators=10, max_depth=5)

    # Fit the model and make predictions on test data
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    # Get confusion matrix and relevant variables
    cm = confusion_matrix(y_test, prediction)
    cm_accuracy = accuracy(cm)
    cm_recall = recall(cm)
    cm_precision = precision(cm)
    return cm_accuracy, cm_recall, cm_precision, cm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()

    # Load data
    npz_path = os.path.join(os.getcwd(), args.input)
    npz_file = np.load(npz_path)
    features_array = npz_file["arr_0"]

    # Get X - features, and y - classes
    X, y = features_array[:, :-1], features_array[:, -1:]

    # Split into test and train data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

    print(X_train.shape)
    print(X_test.shape)

    #  Classification experiment 1
    classify(args.output_dir, X_train, X_test, y_train, y_test)


