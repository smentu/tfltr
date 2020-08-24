import os
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
from scipy.stats import spearmanr
from scipy.stats.mstats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score as r2
from tensorflow.core.util import event_pb2


def csr_to_sparse_tensor(csr_matrix):
    coo = csr_matrix.tocoo()
    t = tf.SparseTensor(
        indices=np.array([coo.row, coo.col]).T,
        values=coo.data,
        dense_shape=coo.shape)
    return t


def performance_indicators(y, y_true, modelname, verbose=False, plot_scatter=False):
    # calculate different accuracy scores
    r2_score = r2(y, y_true)
    spearman_corr = spearmanr(y, y_true)[0]
    rms_error = np.sqrt(mean_squared_error(y, y_true))
    pearson_corr = pearsonr(y, y_true)[0]

    if verbose:
        print(f"prediction accuracy for {modelname}")
        print(f"R^2 score: \t {r2_score}")
        print(f"RMS error: \t {rms_error}")
        print(f"Pearson: \t {pearson_corr}")
        print(f"Spearman: \t {spearman_corr}")

    if plot_scatter:
        data = pd.DataFrame({'true_values': y_true.reshape(-1), 'predictions': y.reshape(-1)})

        joint_grid = sns.jointplot("true_values", "predictions", data=data,
                                   kind="scatter",
                                   xlim=(min(y_true), max(y_true)), ylim=(min(y_true), max(y_true)),
                                   height=7)
        joint_grid.ax_joint.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r')

    summary_dict = {"rmse": rms_error,
                    "r2": r2_score,
                    "pearson": pearson_corr,
                    "spearman": spearman_corr}

    return summary_dict


def my_summary_iterator(path):
    for r in tf.data.TFRecordDataset(path):
        yield event_pb2.Event.FromString(r)


def get_summaries(log_path):
    summaries = []
    for root, dirs, files in os.walk(log_path):
        for name in files:

            summary_dict = {"Pearson": [], "RMSE": [], "Spearman": [], "lr": []}
            summary_dict["step"] = list(range(500))

            path = os.path.join(root, name)

            serialized_examples = tf.data.TFRecordDataset(path)
            for serialized_example in serialized_examples:
                event = event_pb2.Event.FromString(serialized_example.numpy())
                for value in event.summary.value:
                    if value.tag == 'Test Pearson':
                        summary_dict["Pearson"].append(float(tf.make_ndarray(value.tensor)))
                    elif value.tag == 'Test RMSE':
                        summary_dict["RMSE"].append(float(tf.make_ndarray(value.tensor)))
                    elif value.tag == 'Test Spearman':
                        summary_dict["Spearman"].append(float(tf.make_ndarray(value.tensor)))
                    elif value.tag == 'learning rate':
                        summary_dict["lr"].append(float(tf.make_ndarray(value.tensor)))
            try:
                summaries.append(pd.DataFrame(summary_dict))
            except:
                raise OSError('Log files not found')
    return summaries
