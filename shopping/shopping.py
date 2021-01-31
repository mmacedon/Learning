import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import datetime

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    print("Loading Data...")
    evidence, labels = [], []
    evidencelist = []
    with open(filename, newline = '') as csvfile:
        reader = csv.reader(csvfile, delimiter = ' ')
        for row in reader:
            row = row[0].split(',')
            if 'a' in row[0]: continue
            #Correct Data
            evidence = row[0:17]
            labels.append(1 if row[17] == "TRUE" else 0)
            try:
                evidence[10] = datetime.datetime.strptime(evidence[10], "%b").month
            except ValueError as err:
                evidence[10] = 6
            evidence[15] = 1 if evidence[15] == "Returning_Visitor" else 0
            evidence[16] = 1 if evidence[16] == "TRUE" else 0
            for i in range(0, 6, 2):
                evidence[i] = int(evidence[i])
                evidence[i + 1] = float(evidence[i + 1])
            for i in range(6, 10, 1):
                evidence[i] = float(evidence[i])
            for i in range(11, 15, 1):
                evidence[i] = int(evidence[i])
            evidencelist.append(evidence)
    return tuple((evidencelist, labels))


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    classifier = KNeighborsClassifier(n_neighbors = 1)
    classifier.fit(evidence, labels)
    return classifier


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    positiveCorrectIdentified = negativeCorrectIdentified = 0
    totalPositive = totalNegative = 0

    if len(labels) != len(predictions):
        print("The length of labels does not match the length of predictions...")
        sys.exit(1)

    for i in range(len(labels)):
        if labels[i] == 1 and predictions[i] == 1:
            positiveCorrectIdentified += 1
        elif labels[i] == 0 and predictions[i] == 0:
            negativeCorrectIdentified += 1
        if labels[i] == 1:
            totalPositive += 1
        else:
            totalNegative += 1

    if totalPositive + totalNegative != len(labels):
        print("Error! Total Postive + Total Negative != Total Labels")
        sys.exit(1)

    sensitivity = float(positiveCorrectIdentified / totalPositive)
    specificty = float(negativeCorrectIdentified / totalNegative)

    return sensitivity, specificty


if __name__ == "__main__":
    main()
