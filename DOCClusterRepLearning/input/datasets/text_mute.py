# -*- coding: utf8 -*-

import csv
import numpy
import pandas as pd

def text_mute(fnameIn, fnameOut):
    fpIn = open(fnameIn, 'rt', encoding='utf8')
    fpOut = open(fnameOut, 'wt', encoding='utf8')
    csv_reader = csv.reader(fpIn, lineterminator='\n')
    csv_writer = csv.writer(fpOut, lineterminator='\n')
    row0 = next(csv_reader)
    csv_writer.writerow(row0)
    for arow in csv_reader:
        arow[1] = 'anonymised text'
        csv_writer.writerow(arow)
    fpIn.close()


if __name__ == '__main__':
    text_mute('VAD_train_wtext.csv', 'VAD_train_withouttext.csv')
    text_mute('VAD_test_wtext.csv', 'VAD_test_withouttext.csv')
    text_mute('CMF_train_r_wtext.csv', 'CMF_train_r_withouttext.csv')
    text_mute('CMF_test_r_wtext.csv', 'CMF_test_r_withouttext.csv')
