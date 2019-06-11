
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing

import config
from metrics import gini_norm
from DataReader import FeatureDictionary, DataParser
sys.path.append("..")
from DCN import DCN

gini_scorer = make_scorer(gini_norm, greater_is_better=True, needs_proba=True)


def _load_data():
    train_df = pd.read_csv(config.TRAIN_FILE)
    test_df = pd.read_csv(config.TEST_FILE)
    return train_df, test_df

def _preprocess_data(train_df, test_df):
    def preprocess(df):
        cols = [c for c in df.columns if c not in ["id", "target"]]
        df["missing_feat"] = np.sum((df[cols] == -1).values, axis=1)
        df["ps_car_13_x_ps_reg_03"] = df["ps_car_13"] * df["ps_reg_03"]
        return df

    train_df = preprocess(train_df)
    test_df = preprocess(test_df)

    all_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

    cate_list = []
    cate_cnt = 0
    for i in [j for j in all_df.columns if (j not in config.IGNORE_COLS + ['id', 'target']) and (j not in config.NUMERIC_COLS)]:
        cate_list.append(i)
        le = preprocessing.LabelEncoder()
        all_df[i] = le.fit_transform(all_df[i])
        all_df[i] = all_df[i] + cate_cnt
        cate_cnt = all_df[i].max() + 1

    train_df = all_df[:train_df.shape[0]]
    test_df = all_df[train_df.shape[0]:].reset_index(drop=True)
    del all_df

    cols = [c for c in train_df.columns if c not in ["id", "target"]]
    cols = [c for c in cols if (not c in config.IGNORE_COLS)]

    X_train = train_df[cols]
    y_train = train_df["target"].values
    X_test = test_df[cols]
    ids_test = test_df["id"].values

    def preprocess2(df):
        df_copy = df.copy()
        Xi = df_copy[cate_list].values

        df_copy = df.copy()
        Xv = df_copy[config.NUMERIC_COLS].values

        return Xi, Xv

    # 转换成Xi_train, Xi_test以及Xv_train, Xv_test, Xi代表类别型特征，Xv代表数值型特征
    Xi_train, Xv_train = preprocess2(X_train)
    Xi_test, Xv_test = preprocess2(X_test)

    return train_df, test_df, Xi_train, Xv_train, y_train, Xi_test, Xv_test, ids_test, cate_cnt


def _run_base_model_dfm(Xi_train, Xv_train, y_train, Xi_test, Xv_test, ids_test, cate_cnt, folds, dfm_params):
    dfm_params["cate_feature_size"] = cate_cnt
    dfm_params["cate_field_size"] = len(Xi_train[0])
    dfm_params["num_field_size"] = len(Xv_train[0])

    y_train_meta = np.zeros((Xi_train.shape[0], 1), dtype=float)
    y_test_meta = np.zeros((Xi_test.shape[0], 1), dtype=float)
    _get = lambda x, l: [x[i] for i in l]
    gini_results_cv = np.zeros(len(folds), dtype=float)
    gini_results_epoch_train = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)
    gini_results_epoch_valid = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)
    for i, (train_idx, valid_idx) in enumerate(folds):
        Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train, train_idx)
        Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train, valid_idx)

        dfm = DCN(**dfm_params)
        dfm.fit(Xi_train_, Xv_train_, y_train_, Xi_valid_, Xv_valid_, y_valid_)

        y_train_meta[valid_idx,0] = dfm.predict(Xi_valid_, Xv_valid_)
        y_test_meta[:,0] += dfm.predict(Xi_test, Xv_test)

        gini_results_cv[i] = gini_norm(y_valid_, y_train_meta[valid_idx])
        gini_results_epoch_train[i] = dfm.train_result
        gini_results_epoch_valid[i] = dfm.valid_result

    y_test_meta /= float(len(folds))

    # save result
    if dfm_params["use_cross"] and dfm_params["use_deep"]:
        clf_str = "DeepAndCross"
    elif dfm_params["use_cross"]:
        clf_str = "CROSS"
    elif dfm_params["use_deep"]:
        clf_str = "DNN"
    print("%s: %.5f (%.5f)"%(clf_str, gini_results_cv.mean(), gini_results_cv.std()))
    filename = "%s_Mean%.5f_Std%.5f.csv"%(clf_str, gini_results_cv.mean(), gini_results_cv.std())
    _make_submission(ids_test, y_test_meta, filename)

    _plot_fig(gini_results_epoch_train, gini_results_epoch_valid, clf_str)

    return y_train_meta, y_test_meta


def _make_submission(ids, y_pred, filename="submission.csv"):
    pd.DataFrame({"id": ids, "target": y_pred.flatten()}).to_csv(
        os.path.join(config.SUB_DIR, filename), index=False, float_format="%.5f")


def _plot_fig(train_results, valid_results, model_name):
    colors = ["red", "blue", "green"]
    xs = np.arange(1, train_results.shape[1]+1)
    plt.figure()
    legends = []
    for i in range(train_results.shape[0]):
        plt.plot(xs, train_results[i], color=colors[i], linestyle="solid", marker="o")
        plt.plot(xs, valid_results[i], color=colors[i], linestyle="dashed", marker="o")
        legends.append("train-%d"%(i+1))
        legends.append("valid-%d"%(i+1))
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Gini")
    plt.title("%s"%model_name)
    plt.legend(legends)
    plt.savefig("./fig/%s.png"%model_name)
    plt.close()


# load data & preprocess
train_df, test_df = _load_data()
train_df, test_df, Xi_train, Xv_train, y_train, Xi_test, Xv_test, ids_test, cate_cnt = _preprocess_data(train_df, test_df)

# folds
folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,
                             random_state=config.RANDOM_SEED).split(Xi_train, y_train))


# ------------------ DeepAndCross Model ------------------
# params
dfm_params = {
    "use_cross": True,
    "use_deep": True,
    "embedding_size": 8,
    "deep_layers": [32, 32],
    "dropout_deep": [0.5, 0.5, 0.5],
    "deep_layers_activation": tf.nn.relu,
    "epoch": 30,
    "batch_size": 1024,
    "learning_rate": 0.001,
    "optimizer_type": "adam",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.01,
    "verbose": True,
    "eval_metric": gini_norm,
    "random_seed": config.RANDOM_SEED
}
y_train_dfm, y_test_dfm = _run_base_model_dfm(Xi_train, Xv_train, y_train, Xi_test, Xv_test, ids_test, cate_cnt, folds, dfm_params)

# # ------------------ CROSS Model ------------------
# fm_params = dfm_params.copy()
# fm_params["use_deep"] = False
# y_train_fm, y_test_fm = _run_base_model_dfm(dfTrain, dfTest, folds, fm_params)
#
#
# # ------------------ DNN Model ------------------
# dnn_params = dfm_params.copy()
# dnn_params["use_fm"] = False
# y_train_dnn, y_test_dnn = _run_base_model_dfm(dfTrain, dfTest, folds, dnn_params)



