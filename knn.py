import pandas as pd
import argparse
import numpy as np
import scipy.spatial
from sklearn.model_selection import KFold


def read_file(filename):


    raw_data = pd.read_csv(filename, sep=r"\s+", header=None)
    _features = raw_data.iloc[:, 0:-1].values
    _labels = raw_data.iloc[:, -1].values
    _norminal_indices = []

    # Convert norminal features to enum
    for col in range(len(_features[0])):
        if not isinstance(_features[0][col], str):
            continue

        _norminal_indices.append(col)
        category = dict()
        category_id = 0
        for row in range(len(_features)):
            if _features[row][col] not in category:
                category[_features[row][col]] = category_id
                category_id += 1
            _features[row][col] = category[_features[row][col]]

    return _features, _labels, _norminal_indices


def _features_normalization_MinMax(features):
    _features_updated = features
    for col in range(len(features[0])):
        colMax = np.max(features[:,col])
        colMin = np.min(features[:,col])
        colDistance = colMax - colMin
        for row in range(len(features)):
            _features_updated[row][col] = (features[row][col] - colMin)/colDistance
    return _features_updated


def _features_normalization_ZScore(features):
    _features_updated = features
    for col in range(len(features[0])):
        colMean = np.mean(features[:,col])
        colStd = np.std(features[:,col])
        for row in range(len(features)):
            _features_updated[row][col] = (features[row][col] - colMean)/colStd
    return _features_updated


def _distance_cal(testing_features,training_features):
    return scipy.spatial.distance.euclidean(testing_features,training_features)

def knn_classify(testing_features, training_features, training_labels, K):
    testingSize = len(testing_features)
    trainingSize = len(training_features)
    classification = np.zeros(shape=(testingSize,1))
    for i in range(testingSize):
        _distance_label = {}
        distance = [0] * trainingSize
        testing_sample = testing_features[i]
        for j in range(trainingSize):
            distance[j] = _distance_cal(testing_sample,training_features[j])
            _distance_label[distance[j]] = training_labels[j]
        _distance_label = sorted(_distance_label.items(),key = lambda d:d[0])
        #print("dis",distance)
        #print(_distance_label)

        neighbour = {}
        for k in range(K):
            if (neighbour.__contains__(_distance_label[k][1])):
                neighbour[_distance_label[k][1]] += 1
            else:
                neighbour[_distance_label[k][1]] = 1
        neighbour = sorted(neighbour.items(),key = lambda d:d[1],reverse = True)
        classification[i] = neighbour[0][0]
    return classification

def compute_performance(ground_truth, classification):

    true_positive = sum([1 if a == 1 and b == 1 else 0 \
                            for (a, b) in zip(ground_truth, classification)])
    true_negative = sum([1 if a == 0 and b == 0 else 0 \
                            for (a, b) in zip(ground_truth, classification)])
    false_positive = sum([1 if a == 1 and b == 0 else 0 \
                            for (a, b) in zip(ground_truth, classification)])
    false_negative = sum([1 if a == 0 and b == 1 else 0 \
                            for (a, b) in zip(ground_truth, classification)])

    accuracy = 1.0 * (true_positive + true_negative) \
                   / (true_positive + true_negative + false_positive + false_negative)
    precision = 1.0 * true_positive / (true_positive + false_positive)
    recall = 1.0 * true_positive / (true_positive + false_negative)
    f_one_measure = 2.0 * true_positive / (2.0 * true_positive + false_positive + false_negative)

    return [accuracy, precision, recall, f_one_measure]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', default='project3_dataset1.txt', help='file contains data')
    args = parser.parse_args()
title_part = ': file = ' + args.filename
_features, _labels, _norminal_indices = read_file(args.filename)
_features_updated = _features_normalization_MinMax(_features)
#_features_updated = _features_normalization_ZScore(_features)

#cross validation
K=15 #dataset1-15 dataset2-65_MINMAX-25_ZSCORE
kf = KFold(n_splits=10) #10-fold
kf.get_n_splits(_features_updated)
#print(kf)

sum_accuracy = 0
sum_precision = 0
sum_recall = 0
sum_fOneMeasure = 0

for train_index, test_index in kf.split(_features_updated):
    #print("TRAIN:", len(train_index), "TEST:", len(test_index))
    _features_train, _features_test = _features_updated[train_index], _features_updated[test_index]
    _labels_train, _labels_test = _labels[train_index], _labels[test_index]
    classification = knn_classify(_features_test, _features_train, _labels_train, K)
    _accuracy, _precision, _recall, _f_one_measure = compute_performance(_labels_test,classification)

    sum_accuracy += _accuracy
    sum_precision += _precision
    sum_recall += _recall
    sum_fOneMeasure += _f_one_measure
accuracy = sum_accuracy/10
precision = sum_precision/10
recall = sum_recall/10
f_one_measure = sum_fOneMeasure/10
print("Accuracy:", accuracy, "Precision:", precision, "Recall:", recall, "F-1 measure", f_one_measure)




