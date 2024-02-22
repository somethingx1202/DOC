# -*- coding: utf8 -*-

import csv
import numpy
import pandas as pd
import itertools
from math import comb

DATA_PATH = '../input/datasets/'
root = DATA_PATH


def permutation(lst):
        # If lst is empty then there are no permutations
        if len(lst) == 0:
            return []
        # If there is only one element in lst then, only
        # one permutation is possible
        if len(lst) == 1:
            return [lst]
        # Find the permutations for lst if there are
        # more than 1 characters
     
        l = [] # empty list that will store current permutation
     
        # Iterate the input(lst) and calculate the permutation
        for i in range(len(lst)):
           m = lst[i]
     
           # Extract lst[i] or m from the list.  remLst is
           # remaining list
           remLst = lst[:i] + lst[i+1:]
     
           # Generating all permutations where m is first
           # element
           for p in permutation(remLst):
               l.append([m] + p)
        return l


def pairwise_instance_construct(fnameIn, fnameOut):
    # This is for this DOC training dataset
    df = pd.read_csv(root + fnameIn)
    df['aspect_catetegory'] = df['aspect_catetegory'].apply(lambda x: x % 8)
    lst4combine = [i for i in range(df.shape[0])]
    # lst4combine = [1, 2, 3, 4]
    # print(lst4permu)
    lst_all_possible_combins = list(itertools.combinations(lst4combine, 2))
    # print(lst_all_possible_combins)
    # Or we can use rebalancing tool in the loss function. I would choose rebalancing tool in the loss function
    lst_aspectlabel2count = [0 for _ in range(8)]
    for index, row in df.iterrows():
        aspectcate = row['aspect_catetegory']
        lst_aspectlabel2count[aspectcate] += 1
    total_num = len(lst_all_possible_combins)
    lst_percentage = [0 for _ in range(8)]
    print(total_num)
    for i in range(8):
        num_tweets = lst_aspectlabel2count[i]
        num_combs = comb(num_tweets, 2)
        if i == 0:
            print(num_tweets, num_combs)
        lst_percentage[i] = num_combs / total_num
        # lst_same_aspectlabel[]
    
    percent_in_cluster = sum(lst_percentage)
    percent_outofcluster = 1 - percent_in_cluster
    print(f'percent_outofcluster = {percent_outofcluster}')
    # print(num_in_cluster / )
    # dvd = 1 / percent_outofcluster
    reweight_cluster = [1 / percent / percent_in_cluster for i, percent in enumerate(lst_percentage)]
    # so it's percent_in_cluster: percent_outofcluster, 
    print(f'in-clusrter-comb lst_percentage = {lst_percentage}')
    print(f'percent-in-cluster = {percent_in_cluster}')
    print(f'reweight-cluster that reweight each in-cluster {reweight_cluster}')

    fpointerout = open(root + fnameOut, 'wt', encoding='utf8')
    csv_writer = csv.writer(fpointerout, lineterminator='\n')
    csv_writer.writerow(["ID", "lefttext", "righttext", "notorsame", "stance", "aspect_span", "opinion_span", "left_aspect_category", "right_aspect_category"])
    for an_indexpair in lst_all_possible_combins:
        l, r = an_indexpair
        the_array = df.iloc[[l, r]].values
        l_array = the_array[0]
        r_array = the_array[1]
        notorsame = 0
        l_tid = l_array[0]
        if l_array[5] == r_array[5]:
            notorsame = 1
        else:
            notorsame = 0
        csv_writer.writerow([l_tid, l_array[1], r_array[1], notorsame, l_array[2], l_array[3], l_array[4], l_array[5], r_array[5]])
    set_nonsingular = set()
    for an_indexpair in lst_all_possible_combins:
        l, r = an_indexpair
        set_nonsingular.add(l)
        the_array = df.iloc[[l, r]].values
        l_array = the_array[0]
        r_array = the_array[1]
        notorsame = 0
        l_tid = l_array[0]
        # id same or not
        if l_array[0] == r_array[0]:
            notorsame = 1
        else:
            notorsame = 0
        csv_writer.writerow([l_tid, l_array[1], r_array[1], notorsame, l_array[2], l_array[3], l_array[4], l_array[5], r_array[5]])
    for idx in lst4combine:
        if idx not in set_nonsingular:
            the_array = df.iloc[idx].values
            csv_writer.writerow([the_array[0], the_array[1], the_array[1], 1, the_array[2], the_array[3], the_array[4], the_array[5], the_array[5]])
    fpointerout.close()

    # permutation should be rebalanced
    # [16.774555297757157, 24.437725928360173, 43379.000000000015, 11.528451203307252, 11.097921330348578, 2344.8108108108113, 2478.8, 547.9452631578948]

if __name__ == '__main__':
    pairwise_instance_construct("VAD_train.csv", "VAD_train_pairwise.csv")
    pairwise_instance_construct("VAD_test.csv", "VAD_test_pairwise.csv")

