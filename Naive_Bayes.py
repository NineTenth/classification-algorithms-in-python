# Implementation of Naive Bayes
# For continuous features, we assume they follow the Gaussion Distribution. And then apply MLE to estimate the parameters of the distribution.
import numpy as np
import argparse
import math
import help_functions
import sys


def feature_normalization (trainning_dataset_features, test_dataset_features, norminal_indices):
	#This function normalize the raw features.
	#normalized_features = (features - mean (features)) / std (features)

	mean_features = np.mean (trainning_dataset_features, axis = 0)
	var_features =np.var (trainning_dataset_features, axis = 0)
	std_features = []
	trainning_dataset_features = np.array (trainning_dataset_features)
	test_dataset_features = np.array (test_dataset_features)
	for i in range (0, var_features.shape[0]):
		std_features.append (math.sqrt (var_features [i]))

	normalized_trainning_dataset_features = []
	normalized_test_dataset_features = []
	for i in range (0, trainning_dataset_features.shape [0]):
		normalized_feature = (trainning_dataset_features [i] - mean_features) / std_features
		normalized_trainning_dataset_features.append (normalized_feature)
	for i in range (0, test_dataset_features.shape [0]):
		normalized_feature = (test_dataset_features [i] - mean_features) / std_features
		normalized_test_dataset_features.append (normalized_feature)
	normalized_test_dataset_features = np.array (normalized_test_dataset_features)
	normalized_trainning_dataset_features = np.array (normalized_trainning_dataset_features)
	for i in norminal_indices:
		normalized_trainning_dataset_features [:,i] = trainning_dataset_features [:,i]
		normalized_test_dataset_features [:,i] = test_dataset_features [:,i]
	#print normalized_trainning_dataset_features, normalized_test_dataset_features
	return normalized_trainning_dataset_features, normalized_test_dataset_features, mean_features, std_features



def cross_validation (K, features, lables, norminal_indices):

	n_split = features.shape [0] / K
	dimension = features.shape [1]
	accuracy = []
	precision = []
	recall = []
	f1 = []
	for i in range (0, K):
		trainning_dataset_lables = []
		trainning_dataset_features = []
		test_dataset_features = []
		test_dataset_lables = []
		for p in range (0, features.shape [0]):
			if p in range (i * n_split, (i + 1) * n_split):
				test_dataset_features.append (features [p])
				test_dataset_lables.append (lables [p])
			else:
				trainning_dataset_features.append (features [p])
				trainning_dataset_lables.append (lables [p])
		normalized_trainning_dataset_features, normalized_test_dataset_features, mean_features, std_features = \
		feature_normalization (trainning_dataset_features, test_dataset_features, norminal_indices)

		class_dict, para_dict = train (normalized_trainning_dataset_features, trainning_dataset_lables, norminal_indices)
		predict_lables = []

		for j in range (0, normalized_test_dataset_features.shape [0]):
			p_lable = predict (normalized_test_dataset_features[j,:], norminal_indices, class_dict, para_dict)
			predict_lables.append (p_lable)
		#print test_dataset_lables, predict_lables
		
		performance_matrix = help_functions.compute_performance (test_dataset_lables, predict_lables)
		accuracy.append (performance_matrix [0])
		precision.append (performance_matrix [1])
		recall.append (performance_matrix [2])
		f1.append (performance_matrix [3])
	average_accuracy = np.mean (accuracy, axis = 0)
	average_precision = np.mean (precision, axis = 0)
	average_recall = np.mean (recall, axis = 0)
	avearge_f1 = np.mean (f1, axis = 0)
	return [average_accuracy, average_precision, average_recall, avearge_f1]





def train(features, lables, norminal_indices):

	#This function is used to train the model including estimating parameters for Gaussion Distribution and calculating 
	#the descriptor posterior probability for the norminal features and the class prior probability 
	#For the continuous features, we assume they follow the Gaussion Distribution.
	#Apply maximum likelihood estimation to estimate the parameters of Gaussion distribution for continuous features.
	

	para_dict = {}
	#The keys are tuples: for continuous features elements in tuples are 
	#type of feature (0 for continuous features), class_id, the order of the feature and para_id ( 0 for mean and 1 for sd)
	#For norminal features elements in tuples are 
	#type of feature (1 for norminal features), class_id, the order of the feature and the value of the feature.
	#The values are corresponding parameters.
	
	class_dict = {}
	#The keys are class_id and the values are the number of samples in that class
	features = np.array(features)
	lables = np.array(lables)
	dimension = features.shape [1]
	samples_cnt = features.shape [0]

	for i in range (0, samples_cnt):
		class_id = lables[i]
		if class_id in class_dict:
			class_dict [class_id] = class_dict [class_id] + 1
		else:
			class_dict [class_id] = 1
		for j in range (0, dimension):
			if j in norminal_indices:
				n_tuple = (1, class_id, j, features [i,j])
				if n_tuple in para_dict:
					para_dict [n_tuple] = para_dict [n_tuple] + 1
				else:
					para_dict [n_tuple] = 1
			else:
				c_mean_tuple = (0, class_id, j, 0)
				c_sd_tuple = (0, class_id, j, 1)
				if c_mean_tuple in para_dict:
					para_dict [c_mean_tuple] = para_dict [c_mean_tuple] + features [i,j] 
					#sum of features. To calculate mean later
					para_dict [c_sd_tuple] = para_dict [c_sd_tuple] + features [i,j] * features [i,j]
					#sum of square of features. To calculate sd later
				else:
					para_dict [c_mean_tuple] = features [i,j]
					para_dict [c_sd_tuple] = features [i,j] * features [i,j]

	for t_tuple in para_dict:
		feature_cat = t_tuple [0]
		class_id = t_tuple [1]
		feature_id = t_tuple [2]
		feature_name = t_tuple [3]
		value = para_dict [t_tuple]
		if feature_cat == 0:
			#if it is a continuous feature
			if feature_name == 1:
				continue
			else:
				para_dict [t_tuple] = 1.0 * para_dict [t_tuple] / class_dict [class_id]
				t_sd_tuple = (feature_cat, class_id, feature_id, 1)
				para_dict [t_sd_tuple] = math.sqrt(1.0 * para_dict [t_sd_tuple] / class_dict [class_id] - para_dict [t_tuple] * para_dict [t_tuple])
		else:
			#if it is a norminal feature
			para_dict [t_tuple] = 1.0 * para_dict [t_tuple] / class_dict [class_id]
	for c_id in class_dict:
		class_dict [c_id] = 1.0 * class_dict [c_id] / samples_cnt
	return class_dict, para_dict
#Return class prior probability and parameters estimated for Gaussion Distribution for continuous features 
#and descriptor posterior probability for norminal features









def predict(features, norminal_indices, class_dict, para_dict):
	#This function is used to predicte lables in the test dataset based on the trainning model
	label_dict = {}
	dimension = features.shape[0]
	#print dimension
	max_p = -1
	label = -1

	#The keys are lables and the values are the class posterior probability
	for class_id in class_dict:
		#print class_id
		class_prior_p = class_dict [class_id]
		c_post_p = 1.0
		c_post_p = c_post_p * class_prior_p
		for i in range (0, dimension):
			if i in norminal_indices:
				t_tuple = (1, class_id, i, features[i])
				des_post_p = para_dict [t_tuple]
				c_post_p = c_post_p * des_post_p
			else:
				t_mean_tuple = (0, class_id, i, 0)
				t_sd_tuple = (0, class_id, i, 1)
				mean = para_dict [t_mean_tuple]
				sd = para_dict [t_sd_tuple]
				des_post_p = 1.0 / (math.sqrt (2 * math.pi) * sd) * math.e ** ( - (features [i] - mean) * (features [i] - mean) / (2 * sd * sd) )
				#Gaussian Distribution 
				c_post_p = c_post_p * des_post_p
				
		if c_post_p > max_p:
			max_p = c_post_p
			label = class_id
			
	return label


def Naive_Bayes (filename, K):
	features, lables, norminal_indices = help_functions.read_file (filename)
	average_performance_matrix = cross_validation (K, features, lables, norminal_indices)
	return average_performance_matrix


if __name__ == '__main__':
	
	if len(sys.argv) < 2:
		filename = "project3_dataset1.txt"
		K = 10 
	else:
		if (len(sys.argv) == 2):
			K = 10 
			filename = sys.argv [1]
		else:
			filename = sys.argv[1]
			K = int(sys.argv[2])
	average_performance_matrix = Naive_Bayes(filename, K)
	print("The overall performance is:")
	overall_performance = "Accuracy: " + str (average_performance_matrix [0]) + "\nPrecision: " + str (average_performance_matrix [1]) + \
	"\nRecall: " + str (average_performance_matrix [2]) + "\nF1: " + str (average_performance_matrix [3])

	print(overall_performance)
	
