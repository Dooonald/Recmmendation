#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'LDD'

import sys
import numpy as np
import time
import random
import math
import os
from surprise.model_selection import KFold
from surprise import accuracy, KNNBasic, Reader
from collections import defaultdict
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise import NMF, KNNBaseline


def get_top_n(predictions, n=10, threshold = 3.5):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n_est = defaultdict(list)
    true_ratings = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n_est[uid].append((iid, est))
        true_ratings[uid].append((iid, true_r))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n_est.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        user_ratings = [x for x in user_ratings if x[1] > threshold]
        top_n_est[uid] = user_ratings[:n]       # top n
        # add 0 if less than n
        est_len = len(top_n_est[uid])
        if est_len < n:
            for i in range(est_len, n):
                top_n_est[uid].append(('0', 0)) # append 0 if not enough
    # Then sort the true ratings for each user and retrieve the k highest ones.
    for uid, user_ratings in true_ratings.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        true_ratings[uid] = [x for x in user_ratings if x[1] > threshold]          # len

    return top_n_est, true_ratings


def nDCG(ranked_list, ground_truth):
    dcg = 0
    idcg = IDCG(len(ground_truth))
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id not in ground_truth:
            continue
        rank = i + 1
        dcg += 1 / math.log(rank + 1, 2)
    return dcg / idcg if idcg != 0 else 0


def IDCG(n):
    idcg = 0
    for i in range(n):
        idcg += 1 / math.log(i + 2, 2)
    return idcg


def AP(ranked_list, ground_truth):
    hits, sum_precs = 0, 0.0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_truth:
            hits += 1
            sum_precs += hits / (i + 1.0)
    if hits > 0:
        return sum_precs / len(ground_truth)
    else:
        return 0.0


def RR(ranked_list, ground_list):
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            return 1 / (i + 1.0)
    return 0


def precision_and_racall(ranked_list, ground_list):
    hits = 0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            hits += 1
    pre = hits / (1.0 * len(ranked_list) if len(ground_list) != 0 else 1)
    rec = hits / (1.0 * len(ground_list) if len(ground_list) != 0 else 1)
    return pre, rec


def evaluate(top_n_est, true_ratings):
    precision_list = []
    recall_list = []
    ap_list = []
    ndcg_list = []
    rr_list = []

    for u, user_ratings in top_n_est.items():
        tmp_r = top_n_est.get(u)  # [('302', 4.2889227920390836), ('258', 3.9492992642799027)]
        tmp_t = true_ratings.get(u)
        tmp_r_list = []
        tmp_t_list = []
        for (item, rate) in tmp_r:
            tmp_r_list.append(item)

        for (item, rate) in tmp_t:
            tmp_t_list.append(item)
        print(tmp_r_list, "-->", tmp_t_list)

        pre, rec = precision_and_racall(tmp_r_list, tmp_t_list)
        ap = AP(tmp_r_list, tmp_t_list)
        rr = RR(tmp_r_list, tmp_t_list)
        ndcg = nDCG(tmp_r_list, tmp_t_list)
        precision_list.append(pre)
        recall_list.append(rec)
        ap_list.append(ap)
        rr_list.append(rr)
        ndcg_list.append(ndcg)
    precison = sum(precision_list) / len(precision_list)
    recall = sum(recall_list) / len(recall_list)
    f1 = 2 * precison * recall / (precison + recall)
    map = sum(ap_list) / len(ap_list)
    mrr = sum(rr_list) / len(rr_list)
    mndcg = sum(ndcg_list) / len(ndcg_list)
    return f1, map, mrr, mndcg


def main():
    rec = 'SVD'
    threshold = 4
    # First train an SVD algorithm on the movielens dataset.
    print("load data...")
    # data = Dataset.load_builtin('ml-1m')

    # path to dataset file
    file_path = os.path.expanduser('E:/Workspace/PyCharm/BiNE-master/data/1m/ratings.dat')

    # As we're loading a custom dataset, we need to define a reader. In the
    # movielens-100k dataset, each line has the following format:
    # 'user item rating timestamp', separated by '\t' characters.
    reader = Reader(line_format='user item rating timestamp', sep='::')
    data = Dataset.load_from_file(file_path, reader=reader)

    # test set is made of 40% of the ratings.
    test_size = 0.4
    trainset, testset = train_test_split(data, test_size=test_size)

    # data = Dataset.load_builtin('ml-1m')
    trainset = data.build_full_trainset()
    print("test size %.1f..." % test_size)
    print("training...")

    sim_options = {'name': 'cosine',

                   'user_based': False  # compute  similarities between items
                   }
    if rec == 'NMF':
        algo = NMF()
    elif rec == 'SVD':
        algo = SVD()
        name = ['SVD']
    else:
        algo = KNNBaseline(sim_options=sim_options)
        name = ['ItemKNN']

    train_start = time.time()
    algo.fit(trainset)
    train_end = time.time()
    print('train time:%.1f s' % (train_end - train_start))

    # Than predict ratings for all pairs (u, i) that are NOT in the training set.
    # testset = trainset.build_anti_testset()
    predictions = algo.test(testset)
    test_end = time.time()
    print('test time:%.1f s' % (test_end - train_end))

    top_n_est, true_ratings = get_top_n(predictions, n=10, threshold = threshold)

    f1, map, mrr, mndcg = evaluate(top_n_est, true_ratings)
    eval_end = time.time()
    print('evaluate time:%.1f s' % (eval_end - test_end))
    print("algorithm : %s" % rec)
    print('recommendation metrics: F1 : %0.4f, NDCG : %0.4f, MAP : %0.4f, MRR : %0.4f' % (f1, mndcg, map, mrr))

    '''
    # Print the recommended items for each user
    for uid, user_ratings in top_n_est.items():
        print(uid, [iid for (iid, _) in user_ratings])
    print("#" * 150)
    for uid, user_ratings in top_n_true.items():
        print(uid, [iid for (iid, _) in user_ratings])
    '''


if __name__ == "__main__":
    sys.exit(main())
