"""
Functions created to use for the exam in the course 02450 Introduction to machine learning at the Technical University of Denmark

Author: Gabriele Turetta / Gabriel Lanaro
"""
__version__ = "Revision: 2023-12-14"


import numpy as np
from fractions import Fraction
from matplotlib.pyplot import figure, show
from scipy.cluster.hierarchy import linkage, dendrogram
from basic_operations import choose_mode
from scipy.spatial.distance import squareform
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.cluster import KMeans
import sklearn.metrics.cluster as cluster_metrics
import math
from exam_toolbox import *


##THEORY FOR EXERCISES
# SUPPORT: the support of an itemset is the number of rows containing all items in the itemset divided by the total number of rows.
# JACCARD: n11/(N-n00)     n11=coordinates where both vectors are non-zero
# 

# decision (classification) tree 
def entropy(v):
    v = np.array(v)
    l2 = [math.log2(v[i] / sum(v)) for i in range(len(v))]
    l2 = np.array(l2)

    return 1 - np.sum(((v / sum(v)) * l2))

def impurity(class_counts, mode='gini'):
    total_elements = np.sum(class_counts)
    probabilities = class_counts / total_elements

    if mode == 'gini':
        impurity_value = 1 - np.sum(probabilities**2)
    elif mode == 'class_error':
        impurity_value = 1 - np.max(probabilities)
    elif mode == 'entropy':
        impurity_value = -np.sum(probabilities * np.log2(probabilities))
    else:
        raise ValueError("Invalid impurity mode. Use 'gini', 'class_error', or 'entropy'.")

    return impurity_value

def purity_gain(root, children, purity_measure, accuracy=False): #negli esercizi d'esame lo chiama impurity gain ma il nome giusto è purity gain 
    """
    root = list with the size of each class
    children = list of arrays, each array representing the size of each class in a branch after the split
    purity_measure = string with the purity measure to use
    """
    root = np.array(root)
    children = [np.array(child) for child in children]

    v = np.array(children)

    acc = np.sum(np.max(children, axis=1)) / np.sum(v)
    if accuracy:
        print("The accuracy of the split is {}".format(acc))

    Iv = 0

    print("Root Impurity:", impurity(root, purity_measure))

    for i in range(len(children)):
        child_impurity = impurity(children[i], purity_measure)
        Iv += child_impurity * sum(children[i]) / sum(root)
        print(f"Child {i + 1} Impurity:", child_impurity)

    purity = impurity(root, purity_measure) - Iv

    print("The purity gain for the split is {} with the purity measure {}".format(purity, purity_measure))
    return purity

# Esempio di utilizzo
# root_classes = np.array([2, 5, 4])
# child1_classes = np.array([2, 4, 0])
# child2_classes = np.array([0, 1, 4])

# gain = purity_gain(root_classes, [child1_classes, child2_classes], "gini") #accetta qualsiasi numero di child_classes

import numpy as np

def empirical_correlation(covariance_matrix, feature_index_1, feature_index_2):
    """
    Compute empirical correlation between two features given a covariance matrix.

    Parameters:
    -----------
    covariance_matrix : np.ndarray
        The covariance matrix containing variances and covariances of features.
    feature_index_1 : int
        Index of the first feature. (0-index)
    feature_index_2 : int
        Index of the second feature. (0-index)

    Returns:
    --------
    correlation : float
        Empirical correlation between the specified features.
    """
    # Calculate variance of the two features
    var_feature_1 = covariance_matrix[feature_index_1, feature_index_1]
    var_feature_2 = covariance_matrix[feature_index_2, feature_index_2]
    
    # Calculate covariance between the two features
    cov_features = covariance_matrix[feature_index_1, feature_index_2]
    
    # Calculate empirical correlation
    correlation = cov_features / np.sqrt(var_feature_1 * var_feature_2)
    
    return correlation

# emp = np.array(
#     [
#         [143.0, 39.0, -0.0, 253.0, 142.0],
#         [39.0, 415.0, -7.0, -6727.0, 143.0],
#         [-0.0, -7.0, 1.0, 108.0, -2.0],
#         [253.0, -6727.0, 108.0, 370027.0, -1403.0],
#         [142.0, 143.0, -2.0, -1403.0, 171.0],
#     ]
# )

# empirical_correlation(emp, 1, 2)


def calculate_rss(values):
    return np.sum((values - np.mean(values)) ** 2)


## TO USE FOR REGRESSION TREES
def purity_gain_rss(parent_values, left_child_values, right_child_values):
    """
    Calculates the purity gain using RSS (Residual Sum of Squares) for a given split.

    Parameters:
    - parent_values (numpy.ndarray): Values of the parent node before the split.
    - left_child_values (numpy.ndarray): Values of the left child node after the split.
    - right_child_values (numpy.ndarray): Values of the right child node after the split.

    Returns:
    - float: Purity gain using RSS.
    """

    rss_parent = calculate_rss(parent_values)
    rss_left_child = calculate_rss(left_child_values)
    rss_right_child = calculate_rss(right_child_values)
    total_instances = len(parent_values)
    total_instances_left = len(left_child_values)
    total_instances_right = len(right_child_values)

    i_p = (1 / total_instances) * rss_parent
    i_l = (1 / total_instances_left) * rss_left_child
    i_r = (1 / total_instances_right) * rss_right_child

    print(f"Impurity PARENT      : {i_p}")
    print(f"Impurity LEFT CHILD  : {i_l}")
    print(f"Impurity RIGHT CHIKD : {i_r}")

    # Calculate the purity gain using RSS

    purity_gain = (
        i_p
        - (total_instances_left / total_instances) * i_l
        - (total_instances_right / total_instances) * i_r
    )

    return purity_gain

# # EXAMPLE OF USAGE
# parent_values = [12, 6, 8, 10, 4, 2]  # Values of the parent node
# left_child_values = [12, 6, 8, 10]  # Values of the left child node after the split
# right_child_values = [4, 2]  # Values of the right child node after the split

# purity_gain = purity_gain_rss(parent_values, left_child_values, right_child_values)
# print(f"Purity Gain using RSS: {purity_gain}")



def kmeans_1d(x, k, init=None):
    """
    assigns cluster to values in a 1d array
    -----------------------------------------
    parameters:
    -----------
    x = list of values to cluster
    k = number of clusters
    init = values to initialize clusters at
    """
    x = np.array(x).reshape(-1, 1)
    if init == None:
        kmeans = KMeans(n_clusters=k).fit(x)
    else:
        init = np.array(init).reshape(-1, 1)
        kmeans = KMeans(n_clusters=k, init=init).fit(x)

    clusters = kmeans.predict(x)

    centers = []

    for c in np.unique(clusters):
        centers.append(np.mean(x[clusters == c]))

    centers = np.round(centers, 4)
    print("The assigned clusters are: {}".format(clusters))
    print(
        "The cluster centers of the converged k-means algortihm is: {}".format(
            centers
        )
    )
    return None


def adaboost(delta, rounds, weigths = None):
    """
    delta : list of misclassified observations, 
    0 = correctly classified, 1 = misclassified

    rounds [int] : how many rounds to run
    
    !!!CALL FOR EACH ROUND!!!
    
    Example: 
    Given a classification problem with 25 observations in total, 
    with 5 of them being misclassified in round 1, the weights can be calculated as:
    miss = np.zeros(25) 
    miss[:5] = 1 
    te.adaboost(miss, 1)

    The weights are printed 
    """
    # Initial weights
    delta = np.array(delta)
    n = len(delta)
    if weights ==None:
        weights = np.ones(n) / n
        
    # Run all rounds
    for i in range(rounds):
        eps = np.mean(delta == 1)
        alpha = 0.5 * np.log((1 - eps) / eps)
        s = np.array([-1 if d == 0 else 1 for d in delta])
        print(alpha)
        
        # Calculate weight vector and normalize it
        weights = weights.T * np.exp(s * alpha)
        weights /= np.sum(weights)

        # Print resulting weights
    for i, w in enumerate(weights):
        print('w[%i]: %f' % (i, w))



def confusion_matrix_stats(TN, TP, FN, FP):
    accuracy = Fraction(TP + TN, TP + TN + FP + FN)
    recall = Fraction(TP, TP + FN)
    specificity = Fraction(TN, TN + FP)
    precision = Fraction(TP, TP + FP)
    f1_score = Fraction(2 * (precision * recall), precision + recall)
    FPR = Fraction(1 - specificity)
    TNR = Fraction(TN, TN + FP)
    balanced_accuracy = Fraction(recall + specificity, 2)

    metrics = {
        "Accuracy": accuracy,
        "Recall": recall,
        "Specificity": specificity,
        "Precision": precision,
        "F1-score": f1_score,
        "FPR": FPR,
        "TNR": TNR,
        "Balanced Accuracy": balanced_accuracy
    }

    results = {}
    for metric_name, value in metrics.items():
        decimal_value = float(value)
        fractional_value = f" {value.numerator}/{value.denominator} "
        results[metric_name] = f"{metric_name} = {fractional_value} = {decimal_value:.5f}"

    return results

# # Esempio di utilizzo
# TN = 14
# TP = 14
# FN = 18
# FP = 10

# metrics = ef.confusion_matrix_stats(TN, TP, FN, FP)
# for metric, value in metrics.items():
#     print(value)


def draw_dendrogram_df(df, method="single", metric="euclidean"):
    """
    Draw a dendrogram based on the input DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing the data for clustering.
    method : str, optional
        Clustering method: 'single', 'complete', 'average', 'centroid', 'median', 'ward', 'weighted'.
        'single' represents min and 'complete' represents max.
    metric : str, optional
        Distance metric to use for clustering.

    Returns:
    --------
    None (displays the dendrogram plot)
    """
    data = df.to_numpy()  # Convert DataFrame to numpy array
    z = linkage(squareform(data), method=method, metric=metric, optimal_ordering=True)
    plt.figure(figsize=(10, 4))
    dendrogram(z, count_sort="descendent", labels=list(df.index + 1))
    plt.show()
    

def draw_dendrogram_string(x, method='single', metric='euclidean'):
    """
    :param x:
    :param method: single complete average centroid median ward weighted
    single is min, complete is max
    :param metric:
    :return: prints result
    """
    data = np.array(choose_mode(x))
    data = np.array(data)
    z = linkage(squareform(data), method=method, metric=metric, optimal_ordering=True)
    figure(2, figsize=(10, 4))
    dendrogram(z, count_sort='descendent', labels=list(range(1, len(data[0]) + 1)))
    show()


# a = """O1 O2 O3 O4 O5 O6 O7 O8 O9 O10
# O1 0 8.55 0.43 1.25 1.14 3.73 2.72 1.63 1.68 1.28
# O2 8.55 0 8.23 8.13 8.49 6.84 8.23 8.28 8.13 7.66
# O3 0.43 8.23 0 1.09 1.10 3.55 2.68 1.50 1.52 1.05
# O4 1.25 8.13 1.09 0 1.23 3.21 2.17 1.29 1.33 0.56
# O5 1.14 8.49 1.10 1.23 0 3.20 2.68 1.56 1.50 1.28
# O6 3.73 6.84 3.55 3.21 3.20 0 2.98 2.66 2.50 3.00
# O7 2.72 8.23 2.68 2.17 2.68 2.98 0 2.28 2.30 2.31
# O8 1.63 8.28 1.50 1.29 1.56 2.66 2.28 0 0.25 1.46
# O9 1.68 8.13 1.52 1.33 1.50 2.50 2.30 0.25 0 1.44
# O10 1.28 7.66 1.05 0.56 1.28 3.00 2.31 1.46 1.44 0"""
# draw_dendrogram(a, method='average')


# a2 = '5.7 6.0 6.2 6.3 6.4 6.6 6.7 6.9 7.0 7.4'

# print(draw_dendrogram(a2, method='average'))


def jaccard_clusters(Z, Q): #use it only for clusters
    N = len(Z)
    
    # Inizializza le variabili per il conteggio di coppie nello stesso cluster (S) e coppie in cluster diversi (D)
    S = 0
    D = 0

    # Calcola il numero di coppie in S e D
    for i in range(N):
        for j in range(i + 1, N):
            if Z[i] == Z[j] and Q[i] == Q[j]:
                S += 1
            elif Z[i] != Z[j] and Q[i] != Q[j]:
                D += 1

    # Calcola J(Z, Q)
    denominator = 0.5 * N * (N - 1)
    J_ZQ = S / (denominator - D)

    return J_ZQ

# C1 = [1,1,3,1,1,1,1,3,3,2]
# C2 = [1,1,2,2,2,3,3,3,3,3]
# jaccard_clusters(C1, C2)

def rand_index_clusters(Z, Q): #use it only for clusters. NB: Rand index = SMC
    N = len(Z)
    
    # Inizializza le variabili per il conteggio di coppie nello stesso cluster (S) e coppie in cluster diversi (D)
    S = 0
    D = 0

    # Calcola il numero di coppie in S e D
    for i in range(N):
        for j in range(i + 1, N):
            if Z[i] == Z[j] and Q[i] == Q[j]:
                S += 1
            elif Z[i] != Z[j] and Q[i] != Q[j]:
                D += 1

    # Calcola R(Z, Q)
    numerator = S + D
    denominator = 0.5 * N * (N - 1)
    R_ZQ = numerator / denominator

    return R_ZQ

# # Esempio di utilizzo con i tuoi dati
# Z = [1, 2, 2, 3]
# Q = [1, 3, 3, 2]

# result = rand_index_clusters(Z, Q)
# print("R(Z, Q):", result)



def similarity_metrics(x, y): #use this only for vectors related exercises, not clusters!
    K = len(x)
    f00 = np.sum((x == 0) & (y == 0))
    f11 = np.sum((x == 1) & (y == 1))

    smc = (f00 + f11) / K
    jaccard = f11 / (K - f00)
    
    dot_product = np.dot(x, y)
    norm_x_squared = np.dot(x, x)
    norm_y_squared = np.dot(y, y)
    extended_jaccard = dot_product / (norm_x_squared + norm_y_squared - dot_product)
    
    cosine_similarity = dot_product / (np.linalg.norm(x) * np.linalg.norm(y))

    # Stampa i risultati
    print("Simple Matching Coefficient (SMC):", smc)
    print("Jaccard Coefficient (J):", jaccard)
    print("Extended Jaccard Coefficient (EJ):", extended_jaccard)
    print("Cosine Similarity:", cosine_similarity)

# x = np.array([0,0,0,1,0,0,0,0,0])
# y= np.array([0,1,1,1,1,1,0,0,0])
# similarity_metrics(x, y)


def ARD(df, obs, K):
    """
    Calculates average relative density
    ----------------------
    parameters:
    ----------------------
    df = symmetric matrix with distances
    obs = the observation to calculate ARD for  (0 index)
    K = the number of nearest neighbors to consider
    """

    O = df.loc[obs, :].values
    dist_sort = np.argsort(O)
    k_nearest_ind = dist_sort[1 : K + 1]

    densO = 1 / (1 / K * np.sum(O[k_nearest_ind]))

    densOn = []

    for n in k_nearest_ind:
        On = df.loc[n, :].values
        dist_sort_n = np.argsort(On)
        k_nearest_ind_n = dist_sort_n[1 : K + 1]
        densOn.append(1 / (1 / K * np.sum(On[k_nearest_ind_n])))

    ARD = densO / (1 / K * np.sum(densOn))

    print("The density for observation O{} is {}".format(obs + 1, densO))
    print(
        "The average relative density for observation O{} is {}".format(
            obs + 1, ARD
        )
    )

    return ARD

# data =[[0,8.55,0.43,1.25,1.14,3.73,2.72,1.63,1.68,1.28],
#         [8.55,0,8.23,8.13,8.49,6.84,8.23,8.28,8.13,7.66],
#         [0.43,8.23,0,1.09,1.10,3.55,2.68,1.50,1.52,1.05],
#         [1.25,8.13,1.09,0,1.23,3.21,2.17,1.29,1.33,0.56],
#         [1.14,8.49,1.10,1.23,0,3.20,2.68,1.56,1.50,1.28],
#         [3.73,6.84,3.55,3.21,3.20,0,2.98,2.66,2.50,3.00],
#         [2.72,8.23,2.68,2.17,2.68,2.98,0,2.28,2.30,2.31],
#         [1.63,8.28,1.50,1.29,1.56,2.66,2.28,0,0.25,1.46],
#         [1.68,8.13,1.52,1.33,1.50,2.50,2.30,0.25,0,1.44],
#         [1.28,7.66,1.05,0.56,1.28,3.00,2.31,1.46,1.44,0]]
# df = pd.DataFrame(data)

# ARD(df,1,2)

def plot_roc(true_val, pred_val):
    """
    calculates the fpr and tpr and plots a roc curve
    to compare the outputtet graph with the possible answers, look at where the plot has a elbow
    -----------------------------------------------
    parameters:
    -----------
    true_val = list of the correct labels, must be binarised so that 1 = positve class and 0 = negative class
    pred_val = list of the predicted labels, must be binarised so that 1 = positve class and 0 = negative class

    returns the area under the curve (AUC)
    """
    fpr, tpr, _ = metrics.roc_curve(true_val, pred_val)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title("Receiver Operating Characteristic")
    plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.show()




def normalized_mutual_info_score(y, clusterid):
    '''
    normalization of the Mutual Information (MI) score to scale the results between 0 (no mutual information) and 1 (perfect correlation)
    param:
    y = list of correct labels
    clusterid = list of labels assigned by the clustering

    returns the normalized mutual informationof the two clusterings (the correct clustering and the other one). average_method is set to default to geometric (exercises in the exam want
    geometric method (always?)). other average methods are min, max.
    '''

    NMI = cluster_metrics.normalized_mutual_info_score(y,clusterid, average_method='geometric')
    return NMI

# y =[0,0,0,1,1,1,1,1,2,2]
# clusterid =[0,2,0,1,0,1,0,0,0,0]
# print(normalized_mutual_info_score(y,clusterid))


def empirical_correlation_from_covariance(cov_matrix):
        """
        cov_matrix : 2d array of covariance matrix, eg: [[0.2639, 0.0803], [0.0803, 0.0615]]
        Calculates the correlation between the two x's in the covariance matrix
        The correlation coefficient is defined as: p=cov(x,y)/(sigma_x*sigma_y)
        nb: se l'esercizio chiede la correlazione tra x1 e x2 efornisce una matrice n x n, 
        bisogna creare una matrice 2x2 del tipo = [[covarianza x1-x1, covarianza x1-x2],[covarianza x1-x2 , covarianza x2-x2]
        """
        cov = np.array(cov_matrix)
        p = cov[1][0]/math.sqrt((cov[0][0]*cov[1][1]))
        print (p)
        return p


def rule_stats(df, X, Y, index=0):
    """
    calculates support and confidence of an association rule
    ----------------------
    parameters:
    -----------
    df = binary dataframe with rows as transactions and columns representing the attributes
    X = list with the X itemsets (0-index)
    Y = list with the Y itemsets (0-index)
    """
    X = np.array(X) - index
    Y = np.array(Y) - index

    X_item = df.iloc[:, X.flatten()]  # Use flatten to ensure 1-dimensional indexing
    Y_item = df.iloc[:, Y.flatten()]
    items = pd.concat([X_item, Y_item], axis=1)

    support = np.mean(np.mean(items, axis=1) == 1)
    confidence = np.sum(np.mean(items, axis=1) == 1) / np.sum(
        np.mean(X_item, axis=1) == 1
    )

    the_rule = "{{{}}} ---> {{{}}}".format(X+1, Y+1)

    print("The rule is {}".format(the_rule))
    print("The support for the rule is {}".format(support))
    print("The confidence for the rule is {}".format(confidence))

# example
# data = [[1, 1, 1, 0, 0],
#         [1, 1, 1, 0, 0],
#         [1, 1, 1, 0, 0],
#         [1, 1, 1, 0, 0],
#         [1, 1, 1, 0, 0],
#         [0, 1, 1, 0, 0],
#         [0, 1, 0, 1, 1],
#         [1, 1, 1, 0, 0],
#         [1, 0, 1, 0, 0],
#         [0, 0, 0, 1, 1],
#         [0, 1, 0, 1, 1]]

# df = pd.DataFrame(data)
# x = [0,1]
# X = pd.DataFrame(x)
# y = [2]
# Y = pd.DataFrame(y)

# rule_stats(df, X, Y)



def apriori_algorithm(df, support_min):
        """
        df: dataframe with each row being a basket, and each column being an item
        support_min: minimum support level
        Remember that the printed itemsets start from 0!
        return the frequent itemsets, that is to say the itemsets with support>support_min
        """
        itemsets = []
        n = len(df)
        for itsetSize in np.arange(1, len(df.columns) + 1): # Start with 1-itemsets, keep going till n_attributes-itemsets
            itemsets.append("L: ")
            itemsets.append(itsetSize)
            for combination in IT.combinations(df.columns, itsetSize):
                itemset = df[list(combination)]
                baskets = itemset.iloc[:,0].copy()
                for col in itemset.columns[1:]:
                    baskets = baskets & itemset[col]
                sup =  baskets.sum() / float(len(baskets))
                if sup > support_min:
                    itemsets.append(set(combination))
        print(itemsets)
        return itemsets

#esempio:
# data=[[1,0,1,0,1,0,1,0,1,0,1,0],
# [0,1,0,1,0,1,0,1,0,1,0,1],
# [1,0,0,1,1,0,1,0,1,0,1,0],
# [1,0,1,0,1,0,0,1,0,1,1,0],
# [0,1,1,0,1,0,1,0,1,0,1,0],
# [0,1,0,1,0,1,0,1,0,1,0,1],
# [0,1,1,0,1,0,0,1,0,1,0,1],
# [1,0,1,0,1,0,1,0,0,1,0,1],
# [0,1,0,1,1,0,1,0,0,1,0,1],
# [1,0,0,1,0,1,0,1,0,1,1,0]]

# df= pd.DataFrame(data)
# apriori_algorithm(df, 0.4)
#l'output è del tipo: ['L: ', 1, {0}, {1}, {2}, {3}, {4}, {6}, {7}, {9}, {10}, {11}, 'L: ', 2, {2, 4}, {4, 6}, {9, 7}, {9, 11}, 'L: ', 3, 'L: ', 4, 'L: ', 5 etc etc
# L:1 significa itemset con una sola feature, L:2 istemset con due features etc etc

def cum_var_explained(S, plot=True, show_df=True):
    """
    S = list of variances of components (can be read from the S/Sigma matrix)
    plot = to plot or not
    show_df = to show df with varaince explained or not
    """
    S = np.array(S)

    # df with cumulative variance explained
    df_var_exp = pd.DataFrame(columns=["k", "var_explained"])
    for i in range(len(S)):
        t = np.sum(S[0 : i + 1] ** 2) / np.sum(S ** 2)
        df_var_exp.loc[i] = [i + 1, t]
    if plot:
        # plot of cumulative variance explained
        plt.plot(df_var_exp["k"], df_var_exp["var_explained"])
        plt.scatter(df_var_exp["k"], df_var_exp["var_explained"])
        plt.title("Variance explained")
        plt.xlim(np.min(df_var_exp["k"]), np.max(df_var_exp["k"]))
        plt.ylim(0, 1)
        plt.show()
        


    if show_df:
        print(df_var_exp)
    return df_var_exp



def gmm_weighted_density(sigma, dim, vect_mean, weights):
    """
    Computes the weighted probability density in a Gaussian Mixture Model (GMM).

    Args:
    sigma (float): Standard deviation of the distribution
    dim (int): Dimensionality of the distribution (number of observations)
    vect_mean (list): List of mean values of observations (or distances from the matrix given)
    weights (list): List of weights corresponding to the observations (usually equally weigthed)

    Returns:
    float: Calculated probability density considering different weights for each observation.
    """
    # Convert the weights into a NumPy array
    weights_array = np.array(weights)
    
    # Calculate the prefactors for each observation
    prefactor = weights_array / ((2 * np.pi * sigma**2) ** (dim / 2))

    # Calculate exponentials of the observation mean values
    exp_tot = np.exp(-np.array(vect_mean) ** 2 / (2 * sigma**2))

    # Compute the weighted sum of prefactors and exponentials
    result = np.dot(prefactor, exp_tot)

    return result

## EXAMPLE OF USE
# weigths = [0.3333, 0.3333, 0.3333]
# vect_dist = np.array([2.11, 1.15, 1.09])
# sigma = 0.5
# dim = 10

# gmm_weighted_density(sigma, dim, vect_dist, weigths)



def prob_gmm(x, weights, means, standard_dev, target_class="all"):
        """Return the mixture prob that x belongs to a class
        A normal distribution is assumed
        Input:
            x: the variable of interest in prob calculations
            weights: a list of weights for all classes
            means: a list of means of the normal distribution for all classes
            standard_dev: a list of standard deviaitons (sigma) (NOT VARIANCES) for all classes
            target_class: desired class to calc the prob for. ZERO INDEX (y=0, or 1, 2 etc)
                        By default, keyword "all" calculates for every class (recommended)
        Example:
        x = 3.19
        weights = [0.19, 0.34, 0.48]
        means = [3.177, 3.181, 3.184]
        standard_dev = [0.0062, 0.0076, 0.0075]
        gm = gmm()
        gm.prob_gmm(x,weights,means,standard_dev,"all") #zero index for target class, we want 2 so y=1

        """
        
        #Error checking for bad user input:
        assert len(weights)==len(means)==len(standard_dev),\
        "The weights, means and std lists must have the same number of items. Check your lists."
        if type(target_class) == int:
            assert target_class < len(means), "Target Class is out of range. Make sure you are starting from 0"
        
        y = target_class
        res = []
        for (w,u,s) in zip(weights, means, standard_dev):
            p_i = w*st.norm.pdf(x=x,loc=u,scale=s)
            res.append(p_i)
        if y == "all":
            for i in range(len(res)):
                print(f"The prob that x={x} belongs to class {i} (0 index) is {res[i]/sum(res)}")
        else:
            print(f"The prob that x={x} belongs to class {y} (0 index) is {res[y]/sum(res)}")

#example
# x = 3.19
# weights = [0.19, 0.34, 0.48]
# means = [3.177, 3.181, 3.184]
# standard_dev = [0.0062, 0.0076, 0.0075]
# prob_gmm(x,weights,means,standard_dev,"all") #zero index for target class, we want 2 so y=1



def knn_dist_pred_2d(df, class1, class2, K, show=False):
        """
        calculates predictions given a matrix with euclidean distances, can only handle two classes: red and black
        -------------------------------------------------------
        df = panda dataframe con le distanze tra le osservazioni
        class1 = list with coloumn numbers of observations in the red class (starts at 1). indicare i numeri delle colonne (index 1) delle osservazioni che appartengono a class1!!
        class2 = list with coloumn numbers of observations in the black class (starts at 1)
        """
        classes = {"red": class1, "black": class2}
        # Get indexes of of red/black observations
        red_ind = [i - 1 for i in classes["red"]]
        black_ind = [i - 1 for i in classes["black"]]
        pred_label = []
        O = [i for i in range(1, df.shape[1] + 1)]
        for row in range(df.shape[0]):
            dist = df.loc[row, :].values
            # sort
            dist_sort = np.argsort(dist)
            k_nearest_ind = dist_sort[1 : K + 1]
            pred_red = 0
            pred_black = 0
            for i in range(K):
                if k_nearest_ind[i] in red_ind:
                    pred_red += 1
                elif k_nearest_ind[i] in black_ind:
                    pred_black += 1
            if pred_red > pred_black:
                pred_label.append("red")
            elif pred_black > pred_red:
                pred_label.append("black")
            elif pred_black == pred_red:
                if k_nearest_ind[0] in red_ind:
                    pred_label.append("red")
                else:
                    pred_label.append("black")
        true_label = []
        for obs in O:
            if obs - 1 in red_ind:
                true_label.append("red")
            elif obs - 1 in black_ind:
                true_label.append("black")
        predictions = pd.DataFrame(
            {"Obs": O, "True_label": true_label, "Predicted_label": pred_label}
        )
        if show:
            print("-" * 100)
            print("The predictions when using the {} nearest neighbors are: ".format(K))
            print(predictions)
        return predictions
#example
# b=[[0,58.5,51.6,18.1,38.0,52.5,71.7,50.7],
# [58.5,0,32.1,72.6,50.5,65.0,13.2,63.8],
# [51.6,32.1,0,60.5,28.4,32.9,45.3,56.3],
# [18.1,72.6,60.5,0,45.9,60.4,79.8,56.8],
# [38.0,50.5,28.4,45.9,0,17.5,63.7,50.7],
# [52.5,65.0,32.9,60.4,17.5,0,78.2,57.2],
# [71.7,13.2,45.3,79.8,63.7,78.2,0,71.0],
# [50.7,63.8,56.3,56.8,50.7,57.2,71.0,0]]
# df = pd.DataFrame(b)
# class1 = [1,2,3,4]
# class2 = [5,6,7,8]
# knn_dist_pred_2d(df, class1, class2, 3, show=True)


def knn_dist_pred_3d(df, class1, class2, class3, K, show=False):
    """
    calculates predictions given a matrix with euclidean distances, can handle tree classes: red, black, blue
    -------------------------------------------------------
    class1 = list with coloumn numbers of observations in the red class (starts at 1)
    class2 = list with coloumn numbers of observations in the black class (starts at 1)
    class3 = list with coloumn numbers of observations in the blue class (starts at 1)
    """
    classes = {"red": class1, "black": class2,"blue": class3}
    # Get indexes of of red/black observations
    red_ind = [i - 1 for i in classes["red"]]
    black_ind = [i - 1 for i in classes["black"]]
    blue_ind = [i - 1 for i in classes["blue"]]
    pred_label = []
    O = [i for i in range(1, df.shape[1] + 1)]
    for row in range(df.shape[0]):
        dist = df.loc[row, :].values
        # sort
        dist_sort = np.argsort(dist)
        k_nearest_ind = dist_sort[1 : K + 1]
        pred_red = 0
        pred_black = 0
        pred_blue = 0
        for i in range(K):
            if k_nearest_ind[i] in red_ind:
                pred_red += 1
            elif k_nearest_ind[i] in black_ind:
                pred_black += 1
            elif k_nearest_ind[i] in blue_ind:
                pred_blue += 1
        if pred_red > pred_black and pred_red > pred_blue:
            pred_label.append("red")
        elif pred_black > pred_red and pred_black > pred_blue:
            pred_label.append("black")
        elif pred_blue > pred_red and pred_blue > pred_black:
            pred_label.append("blue")
        elif pred_black == pred_red == pred_blue:
            if k_nearest_ind[0] in red_ind:
                pred_label.append("red")
            elif k_nearest_ind[0] in black_ind:
                pred_label.append("black")
            else:
                pred_label.append("blue")
    true_label = []
    for obs in O:
        if obs - 1 in red_ind:
            true_label.append("red")
        elif obs - 1 in black_ind:
            true_label.append("black")
        elif obs - 1 in blue_ind:
            true_label.append("blue")
    predictions = pd.DataFrame(
        {"Obs": O, "True_label": true_label, "Predicted_label": pred_label}
    )
    if show:
        print("-" * 100)
        print("The predictions when using the {} nearest neighbors are: ".format(K))
        print(predictions)
    return predictions



def softmax(x):
    """
    calcola funzione di attivazione softmax dato in input un'osservazione x di qualsiasi dimensione
    """
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def multinomial_regression(x, w):
    """
        calcola la probabilità che l'osservazione x appartenga a una classe usando multinomial regression classifier
        -------------------------------------------------------
        x = vettore di osservazioni di qualsiasi dimensione
        w = vettore dei pesi (un peso per ogni classe, i pesi devono avere la stessa dimensione delle osservazioni)
        """
    for i, x_i in enumerate(x):
        probabilities = softmax(np.dot(x_i, np.transpose(w)))
        for j, theta_j in enumerate(probabilities):
            print(f"Probabilità che x{i+1} ({x_i}) appartenga a classe {j+1} = {theta_j:.4f}")

# # Esempio di utilizzo
# x = np.array([[0.8, -1], [0.2, 1]]) #vettore x che contiene le osservazioni
# w = np.array([[-1, 1], [1, 1], [-1, -1]]) #un peso per ciascuna classe

# multinomial_regression(x, w)


def naive_bayes(y, df, cols, col_vals, pred_class):
        """
        probability of a naive bayes classifier, with more than 2 classes
        -------------------------------------
        parameters:
        ----------
        y = list of observation class labels (starting at 0)
        df = data frame with binary data
        cols = columns to condition the probability on (starts at 0)
        col_vals = the values the columns are condtioned on
        pred_class = the class you would like to predict the probability of (starts at 0) <- remember this if y starts on 1
        """
        y = np.array(y)

        probs = []
        for c in range(len(np.unique(y))):
            n = np.mean(y == c)
            suby = df.iloc[y == c, :]
            for i in range(len(cols)):
                p = np.mean(suby.loc[:, cols[i]] == col_vals[i])
                n *= p
            probs.append(n)

        prob = probs[pred_class] / np.sum(probs)

        print(
            "The probability that the given class is predicted by the Naïve Bayes classifier is {}".format(
                prob
            )
        )
        return None


# a = np.array([[1,1,1,0,0],
# [1,1,1,0,0],
# [1,1,1,0,0],
# [1,1,1,0,0],
# [1,1,1,0,0],
# [0,1,1,0,0],
# [0,1,0,1,1],
# [1,1,1,0,0],
# [1,0,1,0,0],
# [0,0,0,1,1],
# [0,1,0,1,1]])

# cols = [1,2]
# y = [0,0,0,0,0,0,0,0,1,1,1]
# col_vals = [1,0]
# pred_class = 1
# df = pd.DataFrame(a)
# naive_bayes(y, df, cols, col_vals, pred_class)


def distance_between_clusters(self,dist_matrix,cluster1,cluster2):
        """Returns distance between clusters (avg linkage function)

        Args:
            dist_matrix (matrix): Distance matrix between observations
            cluster1: Indexes of cluster 1 observations
            cluster2: Indexes of cluster 2 observations
        """
        distances = []
        for i in cluster1:
            for j in cluster2:
                distances.append(dist_matrix[i][j])
        distances = np.array(distances)
        sum = np.sum(distances)
        elements = len(cluster1) * len(cluster2) 
        print (sum/elements)

# data = [[0,725,800,150,1000,525,600,500,400,850],
# [725,0,75,575,275,1250,1325,226,325,125],
# [800,75,0,650,200,1325,1400,300,400,51],
# [150,575,650,0,850,675,750,350,250,700],
# [1000,275,200,850,0,1525,1600,500,600,150],
# [525,1250,1325,675,1525,0,75,1025,925,1375],
# [600,1325,1400,750,1600,75,0,1100,1000,1450],
# [500,226,300,350,500,1025,1100,0,100,350],
# [400,325,400,250,600,925,1000,100,0,450],
# [850,125,51,700,150,1375,1450,350,450,0]]

# distance_between_clusters(data, [5,6], [7,8,9])


# # MAHALONOBIS DISTANCE
# # Define the two points as arrays
# point1 = np.array([1, 2, 3])
# point2 = np.array([4, 5, 6])
# # Define the covariance matrix
# covariance = np.array([[1, 0, 0],
#                        [0, 1, 0],
#                        [0, 0, 1]])
# # Compute the Mahalanobis distance
# mahalanobis_distance = distance.mahalanobis(point1, point2, np.linalg.inv(covariance))
# print("Mahalanobis distance:", mahalanobis_distance)


# # EUCLIDEAN DISTANCE
# point1 = np.array((1, 2, 3))
# point2 = np.array((1, 1, 1))
 
# # calculating Euclidean distance
# # using linalg.norm()
# dist = np.linalg.norm(point1 - point2)
 
# # printing Euclidean distance
# print(dist)


def mcnemar_test(n1,n2):
        """
        Prints the p value from the McNemar test between 2 classification models 
        n1 : the total number of times that Model 1 is correct and Model 2 is incorrect. Remember to sum, if multiple folds
        n2 : the total number of times that Model 1 is incorrect and Model 2 is correct. Remember to sum, if multiple folds
        """
        N = n1+n2
        m = min(n1,n2)
        theta = 1/2 #always
        
        p_val = 2*st.binom.cdf(m,N,theta)
        
        print(f"p-val={p_val:.5f}")

def params_ANN(M, n_h, C):

    """
    calculates the amount of parameters needed in ANN
    ----------------------
    parameters [int]:
    -----------
    M = number of features taken into account
    n_h = units in the layer
    C = number of classes in the dataset
    """
    
    params_hidden_layer = (M + 1) * n_h

    params_softmax = (n_h + 1) * C

    total = params_hidden_layer + params_softmax

    print("Parameters in hidden layer:", params_hidden_layer)
    print("Parameters in softmatrix:", params_softmax)
    print("Total parameters in the neural network:", total)
    
    
