from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve

import os
import numpy as np
import pandas as pd
import scipy.spatial.distance as dis

IGNORE_CRITERION = 10
N_COLUMNS = 91
REGISTERED_SPEAKER = 'f0001'
ROOT_PATH = "C:/Users/ryo/Desktop/research/compats/result/"


def remove_verbose_files(input_dirs):
    for input_dir in input_dirs:
        input_dir = os.path.join(ROOT_PATH, input_dir)
        input_files = os.listdir(input_dir)

        for input_file in input_files:
            input_file = os.path.join(input_dir, input_file)

            try:
                df = pd.read_csv(input_file, delimiter='\t')
            except:
                print(f"can't open file: {input_file}")
                os.remove(input_file)
                continue

            if df['j'].max() - df['j'].min() < IGNORE_CRITERION:
                os.remove(input_file)


def remove_empty_directory(input_dirs):
    for input_dir in input_dirs:
        input_dir = os.path.join(ROOT_PATH, input_dir)
        input_files = os.listdir(input_dir)

        if len(input_files) == 0:
            os.rmdir(input_dir)

def overwrite_agg_files(input_dirs):
    for input_dir in input_dirs:
        input_dir = os.path.join(ROOT_PATH, input_dir)
        input_files = os.listdir(input_dir)

        for input_file in input_files:
            input_file = os.path.join(input_dir, input_file)

            df = pd.read_csv(input_file, delimiter='\t')

            sort_target = ['k', 'i']
            grouped = df.groupby(sort_target, as_index=False).sum()
            grouped = grouped.drop(columns='j', axis=1) \
                          .sort_values(sort_target, ascending=False) \
                          .loc[:, sort_target + ['s']]

            if len(grouped.loc[:, 's']) < N_COLUMNS:
                os.remove(input_file)

            print(f"{input_file}: nrows->{len(grouped.loc[:, 's'])}")
            grouped.to_csv(input_file, sep='\t')

def get_likelihoods(file):
    df = pd.read_csv(file, delimiter='\t')
    likelihood = df['s'].values
    return likelihood

def get_difference(std, comp):
    std = get_likelihoods(std)
    comp = get_likelihoods(comp)
    return np.abs(std - comp)

def main():
    std_dir = os.path.join(ROOT_PATH, REGISTERED_SPEAKER)
    std_file, comp_file = os.listdir(std_dir)[:2]
    std_file = os.path.join(std_dir, std_file)
    comp_file = os.path.join(std_dir, comp_file)
    std_diff = get_difference(std_file, comp_file)
    dists, labels = [], []
    input_dirs = os.listdir(ROOT_PATH)
    for input_dir in input_dirs:
        input_dir = os.path.join(ROOT_PATH, input_dir)
        input_files = os.listdir(input_dir)
        if REGISTERED_SPEAKER in input_dir:
            input_files = input_files[2:]

        for input_file in input_files:
            input_file = os.path.join(input_dir, input_file)
            diff = get_difference(std_file, input_file)

            distance = np.linalg.norm(std_diff - diff)
            # distance = dis.cosine(std_diff, diff)
            label = os.path.basename(input_dir) != REGISTERED_SPEAKER
            dists.append(distance)
            labels.append(int(label))
    fpr, tpr, threshold = roc_curve(labels, dists)
    i = np.arange(len(tpr))
    roc = pd.DataFrame(
        {'tf': pd.Series(tpr - (1 - fpr), index=i),
         'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]['threshold'].values[0]
    result = np.where(np.array(dists) > roc_t, 1, 0)
    report = classification_report(labels, result)
    cm = confusion_matrix(labels, result)
    print(report)
    print(cm)


if __name__ == '__main__':
    input_dirs = os.listdir(ROOT_PATH)
    remove_verbose_files(input_dirs)

    input_dirs = os.listdir(ROOT_PATH)
    remove_empty_directory(input_dirs)

    input_dirs = os.listdir(ROOT_PATH)
    overwrite_agg_files(input_dirs)

    # main()