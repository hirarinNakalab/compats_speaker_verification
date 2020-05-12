from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, f1_score

import os
import numpy as np
import pandas as pd
import scipy.spatial.distance as dis

IGNORE_CRITERION = 10
N_COLUMNS = 91
ROOT_PATH = "C:/Users/rin/Desktop/researchProjects/compats_speaker_verification/result"


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
                continue
            try:
                if df['j'].max() - df['j'].min() < IGNORE_CRITERION:
                    os.remove(input_file)
                if len(df.index) < IGNORE_CRITERION:
                    os.remove(input_file)
            except:
                import traceback
                traceback.print_exc()

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
                print(f"removed: {input_file}")
                continue

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
    speakers = os.listdir(ROOT_PATH)
    input_dirs = speakers

    for n in [3, 5, 10, 30, len(speakers)]:
        sub_dirs = input_dirs[:n]
        sub_speakers = speakers[:n]
        get_n_speakers_result(sub_dirs, sub_speakers)


def get_n_speakers_result(input_dirs, speakers):
    scores = []
    for registerd_speaker in speakers:
        std_dir = os.path.join(ROOT_PATH, registerd_speaker)
        if len(os.listdir(std_dir)) < 3:
            continue
        std_diff, std_file = get_std_diff(std_dir)
        dists, labels = [], []
        for input_dir in input_dirs:
            input_dir = os.path.join(ROOT_PATH, input_dir)
            input_files = os.listdir(input_dir)
            if registerd_speaker in input_dir:
                input_files = input_files[2:]

            for input_file in input_files:
                input_file = os.path.join(input_dir, input_file)
                diff = get_difference(std_file, input_file)
                # distance = np.linalg.norm(std_diff - diff)
                distance = dis.cosine(std_diff, diff)
                label = os.path.basename(input_dir) != registerd_speaker
                dists.append(distance)
                labels.append(int(label))

        scores.append(calc_fmeasure(dists, labels))
    print(np.mean(scores))


def get_std_diff(std_dir):
    std_file, comp_file = os.listdir(std_dir)[:2]
    std_file = os.path.join(std_dir, std_file)
    comp_file = os.path.join(std_dir, comp_file)
    std_diff = get_difference(std_file, comp_file)
    return std_diff, std_file


def calc_fmeasure(dists, labels):
    fpr, tpr, threshold = roc_curve(labels, dists)
    i = np.arange(len(tpr))
    roc = pd.DataFrame(
        {'tf': pd.Series(tpr - (1 - fpr), index=i),
         'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]['threshold'].values[0]
    result = np.where(np.array(dists) > roc_t, 1, 0)
    # report = classification_report(labels, result)
    # cm = confusion_matrix(labels, result)
    f1 = f1_score(labels, result, average="binary")
    return f1

if __name__ == '__main__':
    # input_dirs = os.listdir(ROOT_PATH)
    # remove_verbose_files(input_dirs)
    # remove_empty_directory(input_dirs)
    # overwrite_agg_files(input_dirs)

    main()