"""
Implement boosting

Tested in Python 3.6
"""

import argparse
import copy
import numpy as np

import help_functions
import decision_tree as dt


def sampling_by_weights(features, labels, weights):
    """
    Randomly sample the data according to their weights. One thing need to pay
    attention to is that the sampling indices should be same for features and
    labels since they have one-to-one relationship.

    Parameters
    --------------
    features : 2-D array
        [num_of_samples, num_of_features]

    labels : list
        labels for each sample

    Returns
    ---------------
    sampled_features : 2-D array
        Sampled data. Same size as the parameter features

    sampled_labels : list
        labels corresponding to sampled data. Same size as the parameter labels
    """
    num_of_sampels = len(labels)
    # weighted random, refer to
    # https://stackoverflow.com/questions/11373192/generating-discrete-random-variables-with-specified-weights-using-scipy-or-numpy
    rand_indicies = np.random.choice(range(num_of_sampels), size=num_of_sampels, p=weights)

    _sampled_features = copy.deepcopy(features)
    _sampled_features = [_sampled_features[i] for i in rand_indicies]

    _sampled_labels = copy.deepcopy(labels)
    _sampled_labels = [_sampled_labels[i] for i in rand_indicies]

    return _sampled_features, _sampled_labels


def boosting(features, labels, rounds):
    """
    Boosting process. Each round will build a decision tree.
    Then calculate the error and classifier's importance.
    Then update the weight on each sample.

    Parameters
    ------------------------
    features : 2-D array, numpy
        all training samples, [num_of_samples, num_of_features]

    labels : list/array, numpy
        labels for training samples

    rounds : int
        The number of times to run boosting

    Returns
    ------------------------
    trees : list[Node()]
        A list of trees (root node) that are generated in each round

    tree_weights : list, float
        Weights of each tree to be used in calculating the final prediction
    """
    trees = []
    tree_weights = []
    num_of_sampels = len(labels)
    sample_weights = [1.0 / num_of_sampels] * num_of_sampels

    for i in range(rounds):
        sampled_features, sampled_labels = sampling_by_weights(features, labels, sample_weights)
        tree = dt.build_decision_tree(sampled_features, sampled_labels)
        classified_labels = dt.classify_testing_dataset(tree, features)
        # calculate the error
        classifier_error = sum([sample_weights[i] if classified_labels[i] != labels[i] else 0\
                                for i in range(num_of_sampels)]) / sum(sample_weights)

        # print("classifier error rate is: %f" % classifier_error)
        # print(min(sample_weights), max(sample_weights))

        # if the error > 50%, start over
        if classifier_error > 0.5:
            continue

        # calculate the classifier's importance
        tree_weight = 1.0 / 2.0 * np.log((1.0 - classifier_error) / classifier_error)

        # update the weights of each record
        sample_weights = [sample_weights[i] * np.exp(-classifier_error) \
                     if classified_labels[i] == labels[i]\
                     else sample_weights[i] * np.exp(classifier_error)\
                     for i in range(num_of_sampels)]
        sample_weights = sample_weights / sum(sample_weights)

        trees.append(tree)
        tree_weights.append(tree_weight)
    return trees, tree_weights


def final_prediction(classifiers, classifier_weights, testing_features):
    """
    Carry out the final prediction on testing data according to the weighted classifiers

    Parameters
    --------------
    classifiers : list, Node()
        list of classifiers, of which each is a decision tree

    classifier_weights : list, float
        the weights for each classifer

    testing_features : 2-D array
        the testing data, [num_of_samples, num_of_features]

    Returns
    ---------------
    classified_labels : list
        list of labels for each testing data
    """
    _classified_labels = []

    # each column is the prediction results of a single record against all classifier
    # [num_of_classifiers, num_of_testing_samples]
    all_labels = []
    for _, classifier in enumerate(classifiers):
        all_labels.append(dt.classify_testing_dataset(classifier, testing_features))

    labels_set = set([0, 1])
    for sample in range(len(testing_features)):
        # find the label that yields the max value
        max_value = -1.0
        label = None
        for y in labels_set:
            temp = sum([classifier_weights[i] if all_labels[i][sample] == y else 0\
                    for i in range(len(classifiers))])
            if temp > max_value:
                max_value = temp
                label = y
        _classified_labels.append(label)

    return _classified_labels


if __name__ == "__main__":
    parameter_parser = argparse.ArgumentParser()
    parameter_parser.add_argument("-f", "--filename", default="project3_dataset2.txt", \
                                  help="file contains data")
    parameter_parser.add_argument("-k", "--num_of_folds", type=int, default=10, \
                                  help="number of fold-cross validation")
    parameter_parser.add_argument("-r", "--num_of_rounds", type=int, default=5, \
                                  help="the number of rounds to run")
    args = parameter_parser.parse_args()

    features, labels, _ = help_functions.read_file(args.filename)

    overall_performance = []
    for i in range(args.num_of_folds):
        # split the data into training and testing
        training_features, testing_features = help_functions.get_traing_testing(features, args.num_of_folds, i)
        training_labels, testing_labels = help_functions.get_traing_testing(labels, args.num_of_folds, i)

        # boosting to get a bunch of classifiers (decision trees) and their importance (weights)
        classifiers, classifier_weights = boosting(training_features, training_labels, args.num_of_rounds)

        classified_labels = final_prediction(classifiers, classifier_weights, testing_features)
        performance = help_functions.compute_performance(testing_labels, classified_labels)
        overall_performance.append(performance)
        print("fold " + str(i) + "'s performance: " + str(performance))

    overall_performance = np.array(overall_performance)
    print("\nOverall performance: " + str(overall_performance.mean(axis=0)))
