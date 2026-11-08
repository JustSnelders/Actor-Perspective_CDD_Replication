import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

colors = ['darkred']
# colors = ['darkorange', 'darkred']
# colors = ['gold', 'darkorange', 'red', 'darkred']


def plot_trends(np_feature_vectors, feature_names, window_size, output_directory, dict_change_points="",
                subgroup="", min_freq=5, highlight_ranges=None):
    np_feature_vectors = np.transpose(np_feature_vectors)
    intervals = range(0, np.shape(np_feature_vectors)[1])
    for index, feature in enumerate(feature_names):
        feature_vector = np_feature_vectors[index]
        sum_freq = np.sum(feature_vector)
        if sum_freq < min_freq:
            continue
        plt.figure(figsize=(8, 3))
        plt.plot(intervals, feature_vector, '-')
        y_max = max(10, np.amax(feature_vector))

        penalty_file_name = ""
        # check if dict_change_points was passed and if it contains change points
        if dict_change_points != "":
            if [item for sublist in list(dict_change_points.values()) for item in sublist]:
                idx = 0
                for penalty, change_points in dict_change_points.items():
                    penalty_file_name += f"{penalty}_"
                    for cp_index, cp in enumerate(change_points):
                        if cp_index + 1 == len(change_points):
                            plt.plot([cp, cp], [0, y_max], '--', label=penalty, color=colors[idx])
                        else:
                            plt.plot([cp, cp], [0, y_max], '--', color=colors[idx])
                    idx += 1
                plt.legend(loc="best")
        if highlight_ranges is not None:
            for r in highlight_ranges:
                plt.axvspan(r[0]-0.25, r[-1]+0.25, color='grey', alpha=0.2)
        plt.ylim(0, y_max)
        plt.xlim(intervals[0], intervals[-1])
        if window_size == 7:
            plt.xlabel("Week")
        elif window_size == 1:
            plt.xlabel("Day")
        plt.title(f"{subgroup}_{feature}")
        # plt.ylabel("Frequency")
        if subgroup == "":
            plt.savefig(f"{output_directory}\\{window_size}day_{penalty_file_name}{feature}", bbox_inches='tight')
        else:
            plt.savefig(f"{output_directory}\\{window_size}day_{penalty_file_name}{subgroup}_{feature}", bbox_inches='tight')
        plt.close()
