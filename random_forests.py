"""
Implement random forests

tested in Python 3.6
"""

import argparse
import numpy as np
import random

import help_functions
import decision_tree as dt


def build_random_forests(features, labels, forests_size):
    """
    Create the random forests.

    Parameters
    -----------------
    features : 2-D array
        [n_sampels, m_features]

    labels : 1-D array
        binary labels for each sample

    forests_size : int
        how many decision trees that will be created in the random forests

    Returns
    -----------------
    A forest, i.e. a list of decision trees
    """
    forest = []
    # number of features to be used for measurement calculation for each node
    feature_batch_size = int(len(features[0]) / 5)
    number_of_samples = len(labels)
    for i in range(forests_size):
        # randomly select N (number of total training samples) records (with replacement)
        # to build each tree
        rand = [random.randint(0, number_of_samples - 1) for x in range(number_of_samples)]
        _features = [features[r] for r in rand]
        _labels = [labels[r] for r in rand]
        forest.append(dt.build_decision_tree(_features, _labels, feature_batch_size))

    return forest


def classify_testing_dataset(forests, t_features):
    """
    Classify each testing sample/record against each tree of the forests. Use majority voting
    to merge the result.

    Parameters
    -----------------
    forests : list of Node()
        A list of root node of decision trees in a forest

    t_features : 2-D array
        a list of testing samples, [num_of_samples, num_of_features]

    Returns
    -----------------
    result : list
        the classified label for each testing samples
    """
    result = []
    for row in t_features:
        classified_labels = []
        for tree in forests:
            classified_labels.append(dt.classify_testing_record(tree, row))

        result.append(dt.majority_voting(classified_labels))
    return result


if __name__ == "__main__":
    parameter_parser = argparse.ArgumentParser()
    parameter_parser.add_argument("-f", "--filename", default="project3_dataset2.txt", \
                                  help="file contains data")
    parameter_parser.add_argument("-k", "--num_of_folds", type=int, default=10, \
                                  help="number of fold-cross validation")
    parameter_parser.add_argument("-T", "--num_of_trees", type=int, default=20, \
                                  help="the number of trees to grow in each forest")
    args = parameter_parser.parse_args()

    features, labels, _ = help_functions.read_file(args.filename)

    overall_performance = []
    for i in range(args.num_of_folds):
        # split the data into training and testing
        training_features, testing_features = help_functions.get_traing_testing(features, args.num_of_folds, i)
        training_labels, testing_labels = help_functions.get_traing_testing(labels, args.num_of_folds, i)

        decision_trees = build_random_forests(training_features, training_labels, args.num_of_trees)

        classfied_labels = classify_testing_dataset(decision_trees, testing_features)
        performance = help_functions.compute_performance(testing_labels, classfied_labels)
        overall_performance.append(performance)
        print("fold " + str(i) + "'s performance: " + str(performance))

    overall_performance = np.array(overall_performance)
    print("\nOverall performance: " + str(overall_performance.mean(axis=0)))
