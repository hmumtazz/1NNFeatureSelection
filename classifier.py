"""Modules for 1NN classifer and evaluation"""

import math
from collections import Counter


def euclidean_distance(instance_a, instance_b, feature_subset):
    """Calculate Euclidean distance between two instances"""

    dist_sq=0.0
    for f in feature_subset:
        diff = instance_a[f-1] - instance_b[f-1]
        dist_sq+= diff * diff
    return math.sqrt(dist_sq)

def nearest_neighbor_classifier(labels, features, test_index, feature_subset):
    "Classify single instance with 1nn with leave one out"

    best_label = None
    best_dist = float('inf')
    test_instance = features[test_index]
    

    for i in range(len(features)):
        if i == test_index:
            continue
        dist= euclidean_distance(test_instance,features[i],feature_subset)
        if dist<best_dist:
            best_dist =dist
            best_label =labels[i]

    return best_label

def leave_eval(labels, features, feature_subset):
    """Checking accuracy of 1nn"""
    n = len(labels)
    correct = 0

    for i in range(n):
        predict = nearest_neighbor_classifier(labels, features, i, feature_subset)
        if predict == labels[i]:
            correct+=1

    return correct/n

def default_rater(labels):

    counts = Counter(labels)
    return max(counts.values()) / len(labels)