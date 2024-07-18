#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gc
from os.path import isfile, join
import os
import io
import json
import pickle
import sys
import time

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd 

from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier        

TICKS_LABEL_SIZE = 20
FIGURES_LABEL_SIZE = 25
FIGURES_LEGEND_SIZE = 14

########################################
# Iterator
########################################
class EncryptedTrafficClassifierIterator():
    def __init__(self, results):
        self.current = -1
        self.high = len(results)
        self.results = results
        
    def __iter__(self):
        return self

    def __next__(self): 
        self.current += 1
        if self.current < self.high:
            _t = self.results[self.current]
            if len(_t) > 1:
                return tuple(_t)
            else:
                return _t[0]
        raise StopIteration

########################################
# EncryptedTrafficClassifier
########################################
class EncryptedTrafficClassifier:
    def __init__(
            self,
            nb_folds,
            nb_packets_per_flow,
            filename_prefix,
            processed_data_output_dir,
            data_dir
    ):
        self.filename_prefix = filename_prefix
        self.processed_data_output_dir = processed_data_output_dir
        self.data_dir = data_dir
        self.figure_output_dir = 'results/figures/'
        self.nb_folds = nb_folds
        self.nb_packets_per_flow = nb_packets_per_flow

        self.force_rf_classification = False
        self.features_used = []
        
        self.results_filename = "results/results_" + self.filename_prefix + "_" + str(int(time.time())) + ".csv"        
        if isfile(self.results_filename):
            print(f"results already present for in file {self.results_filename}")
            sys.exit(1)
        self.X_train_flows = {}
        self.y_train_flows = {}
        self.X_test_flows = {}
        self.y_test_flows = {}

        # self.X_test_isolated_flows = {}
        # self.y_test_isolated_flows = {}

        self.X_train_packets = {}
        self.y_train_packets = {}
        self.X_test_packets = {}
        self.y_test_packets = {}

        self.all_classes = {}
        self.classes = {}        

        self.random_seed = 42

        self.classification_results = pd.DataFrame()

    def __set_rf_pickle_filename(self):
        self.rf_output = {}
        feats = "none"
        if self.features_used != None:            
            feats = '_'.join(self.features_used)
        for pkt in self.nb_packets_per_flow:
            _p = str(pkt)
            if pkt == 600000:
                _p = "all"
            self.rf_output[pkt] = "results/rf_" + self.filename_prefix + "_p" + _p + "_f" + str(self.nb_folds) + "_feats" + str(len(feats)) + ".pickle"

    def __set_xg_pickle_filename(self):                   
        self.xg_output = {}
        feats = '_'.join(self.features_used)
        for pkt in self.nb_packets_per_flow:
            _p = str(pkt)
            if pkt == 600000:
                _p = "all"
            self.xg_output[pkt] = "results/xg_" + self.filename_prefix + "_p" + _p + "_f" + str(self.nb_folds) + "_feats_" + str(len(feats)) + ".pickle"

    def _pickle_dump(self, df, filename):
        with open(self.processed_data_output_dir + filename, "wb") as f:
            pickle.dump(df, f)
            
    def _load_pickle(self, filename):
        with open(self.processed_data_output_dir + filename, 'rb') as f:
            df = pickle.load(f)
            return df.fillna(0)
        
    # encoding of class features (our y)
    def _hotencode_class(self, df):
        print("_hotencode_class")
        print("df.head", df.head())
        print("df.head", df.columns)
        print("classes", self.classes)
        if 'class' not in df.columns:
            return
        for j in range(len(self.classes)):
            df.loc[df['class'] == self.classes[j], 'type'] = j

        df['type'] = df['type'].astype(int)
    
        # make sure that the 'type' feature is correctly filled
        assert len(df[df['type'].isna()]) == 0

    def _generate_data_folds(self, df, filename):
        print("_generate_data_folds")
        start_time = time.time()
        skf = StratifiedKFold(n_splits = self.nb_folds, shuffle = True, random_state = self.random_seed)
        
        df.reset_index()
        X = df.drop('type', axis = 1)
        y = df['type']
        print(X.shape)
        print(y.shape)
        # X_train_isolated, X_test_isolated, y_train_isolated, y_test_isolated = train_test_split(X, y,
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            stratify=y, 
                                                            test_size=0.2)
        # for _i, (train_index, test_index) in enumerate(skf.split(X_train_isolated, y_train_isolated)):
        #     print("  Generating fold #", _i, "after: ", time.time() - start_time, "s")
        #     X_train = X_train_isolated.iloc[train_index]
        #     self._pickle_dump(X_train, str(_i) + "_X_train_" + filename)
        #     y_train = y_train_isolated.iloc[train_index]
        #     self._pickle_dump(y_train, str(_i) + "_y_train_" + filename)
        #     X_test = X_train_isolated.iloc[test_index]
        #     self._pickle_dump(X_test, str(_i) + "_X_test_" + filename)
        #     y_test = y_train_isolated.iloc[test_index]
        #     self._pickle_dump(y_test, str(_i) + "_y_test_" + filename)
        #     del X_train
        #     del X_test
        #     del y_train
        #     del y_test
        #     gc.collect()
        # self._pickle_dump(X_test_isolated, str(_i) + "_X_test_isolated_" + filename)
        # self._pickle_dump(y_test_isolated, str(_i) + "_y_test_isolated_" + filename)
        for _i, (train_index, test_index) in enumerate(skf.split(X, y)):
            print("  Generating fold #", _i, "after: ", time.time() - start_time, "s")
            X_train = X.iloc[train_index]
            self._pickle_dump(X_train, str(_i) + "_X_train_" + filename)
            y_train = y.iloc[train_index]
            self._pickle_dump(y_train, str(_i) + "_y_train_" + filename)
            X_test = X.iloc[test_index]
            self._pickle_dump(X_test, str(_i) + "_X_test_" + filename)
            y_test = y.iloc[test_index]
            self._pickle_dump(y_test, str(_i) + "_y_test_" + filename)
            del X_train
            del X_test
            del y_train
            del y_test
            gc.collect()
            
    def _test_data_prepared(self, test):
        pkt, fold = test
        for prefix in ["X_train_", "y_train_", "X_test_", "y_test_"]:
            filename = self.processed_data_output_dir + str(fold) + "_" + prefix + self.filename_prefix + "_" + str(pkt) + ".pickle"
            if not isfile(filename):
                print(filename, "not found")
                return False
        return True
            
    def data_prepared(self):
        print("data_prepared")
        for pkt in self.nb_packets_per_flow:
            for fold in range(self.nb_folds):
                if not self._test_data_prepared((pkt, fold)):
                    print((pkt, fold), "not found")
                    return False
        return True       

    def data_preparation(self):
        return

    def load_data(self, suffix):
        return
    
    def load_flows(self):
        print("load_flows")
        start_time = time.time()
        features_set = False
        for i in EncryptedTrafficClassifierIterator(self.flow_ids):
            pkt, fold = i
            name = str(fold) + "_X_train_" + self.filename_prefix + "_" + str(pkt) + ".pickle"
            self.X_train_flows[i] = self._load_pickle(name)
            if not features_set :
                self.features_used =  list(self.X_train_flows[i].columns)
                features_set = True
            
            name = str(fold) + "_y_train_"+ self.filename_prefix + "_" + str(pkt) + ".pickle"
            self.y_train_flows[i] = self._load_pickle(name)
            
            name = str(fold) + "_X_test_"+ self.filename_prefix + "_" + str(pkt) + ".pickle"
            self.X_test_flows[i] = self._load_pickle(name)
            
            name = str(fold) + "_y_test_"+ self.filename_prefix + "_" + str(pkt) + ".pickle"
            self.y_test_flows[i] = self._load_pickle(name)
        
            # print("=============")
            # print("X_train =", self.X_train_flows[i].describe(), self.X_train_flows[i].dtypes)
            # print("y_train =", self.y_train_flows[i], self.y_train_flows[i].unique(), self.y_train_flows[i].dtypes)
            # print("X_test =", self.X_test_flows[i].describe(), self.X_test_flows[i].dtypes)
            # print("y_test =", self.y_test_flows[i], self.y_test_flows[i].unique(), self.y_test_flows[i].dtypes)
            # print("=============")
        # name = str(fold) + "_X_test_isolated_"+ self.filename_prefix + "_" + str(pkt) + ".pickle"
        # self.X_test_isolated_flows = self._load_pickle(name)
            
        # name = str(fold) + "_y_test_isolated_"+ self.filename_prefix + "_" + str(pkt) + ".pickle"
        # self.y_test_isolated_flows = self._load_pickle(name)
        
        print(f"  flows data loaded from {name} in {time.time() - start_time} seconds")
        
    def cleanup_data(self, X_train, y_train, X_test, y_test, results, non_needed_features):
        print("cleanup_data")
        for i in EncryptedTrafficClassifierIterator(results):
            for _f in non_needed_features:
                if self.features_used != None:
                    if _f in self.features_used:
                        self.features_used.remove(_f)
                if _f in X_train[i].columns:
                    X_train[i] = X_train[i].drop(_f, axis = 1)
                if _f in X_test[i].columns:
                    X_test[i] = X_test[i].drop(_f, axis = 1)
        # for _f in non_needed_features:
        #     if _f in self.X_test_isolated_flows.columns:
        #         self.X_test_isolated_flows = self.X_test_isolated_flows.drop(_f, axis = 1)

            # print(X_train[fold].dtypes)
        
    def _correlation(self):
        corr = df.corr(method='pearson').sort_values(by='type',
                                                     axis = 0,
                                                     ascending = False
                                                     )
        print(corr['type'])
        print(df.shape)

    # Confusion Matrix
    def confusion_matrix(self, rf_regr, y_test, y_test_pred, results, prefix):
        print("confusion_matrix")
    
        cm_dict = {}
        cm_dict_normalized = {}
        output = ""
        for i in EncryptedTrafficClassifierIterator(results):
            output += ("== %s ==\n" % str(i))
            cm_dict[i] = confusion_matrix(y_test[i], y_test_pred[i].astype(int))
            output += str(cm_dict[i]) + '\n'
            # disp = ConfusionMatrixDisplay(confusion_matrix = cm_dict[i], 
            #                               display_labels = self.classes)
            # disp.plot(xticks_rotation = 45)
            # plt.tight_layout()
            # plt.savefig(str(i) + '_confusion_matrix.pdf', dpi = 300, format = "pdf")

            # plt.clf()
            cm_dict_normalized[i] = confusion_matrix(y_test[i], y_test_pred[i].astype(int), normalize = 'true')
            output += str(cm_dict_normalized[i]) + '\n'
            # disp_normalized = ConfusionMatrixDisplay(confusion_matrix = cm_dict_normalized[i], 
            #                               display_labels = self.classes)
            # disp_normalized.plot(xticks_rotation = 45, values_format='.2f')
            # plt.tight_layout()
            # plt.savefig(str(i) + '_confusion_matrix_normalized.pdf', dpi = 300, format = "pdf")

        for i in EncryptedTrafficClassifierIterator(results):
            pkt, fold = i
            memfile = io.BytesIO()
            np.save(memfile, cm_dict[i])
            serialized = memfile.getvalue()
            _s = json.dumps(serialized.decode('latin-1'))
            self.classification_results.loc[(self.classification_results['nb_packets'] == pkt) & (self.classification_results['fold_id'] == fold), prefix + '_confusion_matrix'] = _s
            
        return cm_dict, output

    def get_F1_score(self, cm_dict, y, y_pred, results, prefix):
        # print("get_F1_score")
        rf_F1 = {}
        skl_F1 = {}
        output = ""
        for i in EncryptedTrafficClassifierIterator(results):
            output += ("== %s ==\n" % str(i))
            pkt, fold = i
            cm = cm_dict[i]
            FP = cm.sum(axis=0) - np.diag(cm)  
            FN = cm.sum(axis=1) - np.diag(cm)
            TP = np.diag(cm)
            TN = cm.sum() - (FP + FN + TP)
            rf_F1[i] = 2 * (TP) / (2 * TP + FP + FN) * 100
            output += ("FP = %s\n" % str(FP))
            output += ("FN = %s\n" % str(FN))
            output += ("TP = %s\n" % str(TP))
            output += ("TN = %s\n" % str(TN))
            if len(y) > 0:
                skl_F1[i] = f1_score(y[i], y_pred[i], average = 'micro')
                output += ("skl_F1 = %s\n" % str(skl_F1[i]))
            output += "\n"
            for j in range(len(self.classes)):
                # print("for type %s, TP/(TP+FP) = %f" % (t, 100*(TP[j] / (TP[j] + FP[j])))
                t = self.classes[j]
                try:
                    output += ("for type %s \t\t F1 = %.2f\n" % (t, rf_F1[i][j]))
                    # self.classification_results[prefix + '_f1_' + t] = rf_F1[i][j]
                    self.classification_results.loc[(self.classification_results['nb_packets'] == pkt) & (self.classification_results['fold_id'] == fold), prefix + '_f1_' + t] = rf_F1[i][j]
                except IndexError as e:
                    # print("Index", i, ",", j, "not found", e)
                    # continue
                    pass
                except KeyError as e:
                    pass
                    # print("KeyError", i, ",", j, "not found", e)
                    # continue
            output += "\n"
        return rf_F1, output

    def avg_f1_scores(self, f1_scores, results):
        # print("avg_f1_scores")
        f1 = {}
        for i in EncryptedTrafficClassifierIterator(results):
            pkt, _ = i
            f1[pkt] = [0 for _ in range(len(self.classes))]
        for i in EncryptedTrafficClassifierIterator(results):
            pkt, _ = i
            for j in range(len(self.classes)):
                try:
                    # f1[pkt][j] += f1_scores[i][j] # TODO
                    # print("f1_scores[", i, ",", j,"]", f1_scores[i][j])
                    # print("f1[", pkt, "]", f1[pkt])
                    # print("f1[", pkt, ",", j,"]", f1[pkt][j])
                    # print(len(f1[pkt]), len(f1_scores[i]),i,j,pkt)
                    f1[pkt][j] += f1_scores[i][j]
                except KeyError as e:
                    # print("KeyError i =", i, ", j =", j, "exception", e)
                    continue
                except IndexError as e:
                    # print("IndexError i =", i, ", j =", j, "exception", e)
                    continue
        
        # print(f1)
        avg_scores = {}
        output = ""
        for pkt in self.nb_packets_per_flow:
            # print(pkt)
            output += f"for {pkt} packets\n"
            # for j, t in self.classes.items(): # TODO
            for j in range(len(self.classes)):
                t = self.classes[j]
                avg_scores[(pkt, t)] = f1[pkt][j] / self.nb_folds
                output += "average for type %s [%d] \t\t F1 = %.2f\n" % (t, j, avg_scores[(pkt, t)])
        output += "\n"
        
        return avg_scores, output
    
    def _get_scores_from_models(self, models, y, y_pred, feats):
        for i in EncryptedTrafficClassifierIterator(self.flow_ids):
            class_report = classification_report(y[i], 
                                                 y_pred[i], 
                                                 target_names = self.classes, 
                                                 output_dict = True)
        accurac = models[i].score(X_test[i][feats], y_test[i])
        macro_score = class_report['macro avg']['f1-score']
        weighted_score = class_report['weighted avg']['f1-score']
        print(i, accurac, macro_score, weighted_score, class_report)
        return class_report
    
    ########################################
    # Visualization
    ########################################
    def _getFigureMaskFromInt(self, i):
        xylin = False
        xlog = False
        ylog = False
        xylog = False
        
        if i == 0:
            xylin = True
        elif i == 1:
            ylog = True
        elif i == 10:
            xlog = True
        elif i == 11:
            xylog = True
        else:
            raise Exception("Wrong parameter %d" % i)
        
        return xylin, xlog, ylog, xylog


    # def __get_viz_class_name(self, c):
    #     __class_names = { 
    #         "youtube": "YouTube",
    #         "Google_Play_Music": "Music",
    #         "GoogleHangout_VoIP": "VoIP",
    #         "GoogleHangout_Chat": "Chat",
    #         "FileTransfer": "FileTransfer",    
    #     }
    #     if c not in  __class_names.keys():
    #         print(c,"=>", c)
    #         return c
        
    #     print(c,"=>", __class_names[c])
    #     return __class_names[c]

    def _viz(self, distribution = -1, class_distribution = -1, nb_packets = -1, min_iat = -1, max_iat = -1):
        # Aggregate train and test sets in a single DataFrame        
        pkt = self.nb_packets_per_flow[0]
        fold = 0
        _i = pkt, fold
        _df1 = self.X_train_flows[_i].copy()
        _df1['type'] = self.y_train_flows[_i]#.map(self.__get_viz_class_name)
            
        _df2 = self.X_test_flows[_i].copy()
        _df2['type'] = self.y_test_flows[_i]#.map(self.__get_viz_class_name)
        _df = pd.concat([_df1, _df2])
        _df.reset_index()
        print(_df.shape)
        print(_df['type'].value_counts())

        if distribution >= 0:
            xylin, xlog, ylog, xylog = self._getFigureMaskFromInt(distribution)
            self._distribution(_df, self.filename_prefix + "_flows_class_split_" + str(pkt) + '_pkt',
                               xlog = xlog, ylog = ylog, xylog = xylog, xylin = xylin)
        if class_distribution >= 0:
            xylin, xlog, ylog, xylog = self._getFigureMaskFromInt(class_distribution)
            self._class_distribution(_df, self.filename_prefix + '_flows_distribution_' + str(pkt) + '_pkt',
                                     xlog = xlog, ylog = ylog, xylog = xylog, xylin = xylin)

        if nb_packets >= 0:
            xylin, xlog, ylog, xylog = self._getFigureMaskFromInt(nb_packets)
            self._nb_packets_distribution(_df, self.filename_prefix + "_flows_nb_packets_distribution_" + str(pkt) + '_pkt',
                                          xlog = xlog, ylog = ylog, xylog = xylog, xylin = xylin)

        if min_iat >= 0:
            xylin, xlog, ylog, xylog = self._getFigureMaskFromInt(min_iat)
            self._min_iat_distribution(_df, self.filename_prefix + "_flows_min_iat_distribution_" + str(pkt) + '_pkt',
                                       xlog = xlog, ylog = ylog, xylog = xylog, xylin = xylin)
        if max_iat >= 0:                
            xylin, xlog, ylog, xylog = self._getFigureMaskFromInt(max_iat)
            self._max_iat_distribution(_df, self.filename_prefix + "_flows_max_iat_distribution_" + str(pkt) + '_pkt',
                                       xlog = xlog, ylog = ylog, xylog = xylog, xylin = xylin)

        
    def _distribution(self, df, filename_prefix, xticks = True, xlog = False, ylog = False, xylog = False, xylin = True):
        print("_distribution")

        plt.clf()
        # sns.set(rc={'figure.figsize':(11.7, 16.54)})
        _df = pd.DataFrame(df['type'])
        # print("columns:", _df.columns)
        # print("value_counts = ", df['type'].value_counts())
        cmap = sns.color_palette("viridis", as_cmap=True)
        norm = matplotlib.colors.BoundaryNorm([0, 1, 2, 3, 4, 5, 6, 7], cmap.N)
        g = sns.histplot(
            # data = df['type'],
            data = _df,
            x = 'type',
            hue = 'type',
            hue_norm = norm,
            palette='viridis',
            discrete = True,
            #element = "bars",
            #kde = True,
            multiple = "stack",   
            legend = False
        )
        #g.set_rasterization_zorder(0)
        if xticks and len(self.classes) < 100:
            g.set_xticks(range(len(self.classes)))
            g.set_xticklabels(self.classes, size = TICKS_LABEL_SIZE)
            g.tick_params(axis = 'x', labelrotation = 45)
            print(g.get_yticks())
            if g.get_yticks()[1] > 1000:
                ylabels = ['{:,.0f}'.format(x) + 'k' for x in g.get_yticks()/1000]
                if g.get_yticks()[0] == 0:
                    ylabels[0] = "0"
                        
                    g.set_yticklabels(ylabels, size = TICKS_LABEL_SIZE)
            else:
                g.set_yticklabels(list(map(int, g.get_yticks())), size = TICKS_LABEL_SIZE)
        else:
            g.set_xticklabels([])
        g.set_xlabel("class", fontsize = FIGURES_LABEL_SIZE)
        g.set_ylabel("count", fontsize = FIGURES_LABEL_SIZE)
        plt.tight_layout()
        if xylin:
            plt.savefig(join(self.figure_output_dir, filename_prefix) + "_" + str(int(time.time())) + ".pdf",
                        format = "pdf")
        plt.xscale('log')
        if xlog:
            plt.savefig(join(self.figure_output_dir, filename_prefix) + "_xlog_" + str(int(time.time())) + ".pdf",
                        format = "pdf")
        plt.yscale('log')
        if xylog:
            plt.savefig(join(self.figure_output_dir, filename_prefix) + "_xylog_" + str(int(time.time())) + ".pdf",
                        format = "pdf")
            
        plt.xscale('linear')
        if ylog:
            plt.savefig(join(self.figure_output_dir, filename_prefix) + "_ylog_" + str(int(time.time())) + ".pdf",
                        format = "pdf")
        
    def _nb_packets_distribution(self, df, filename_prefix, xticks = True, xlog = False, ylog = False, xylog = False, xylin = True):
        print("_nb_packets_distribution")
        from  matplotlib.ticker import FuncFormatter
        
        matplotlib.use('Agg')
        plt.clf()
        # _df = df.copy()
        # _df['nb_packets'] = _df['nb_packets'].astype(int)
        g = sns.boxplot(
            data = df,
            x = 'type',
            y = 'nb_packets',
        )
        g.set_rasterization_zorder(0)
        if xlog or xylog:
            plt.xscale('log')
        if ylog or xylog:
            plt.yscale('log')
        if xticks and len(self.classes) < 100:
            print(self.classes)
            g.set_xticks(ticks = range(len(self.classes)), labels = self.classes, size = TICKS_LABEL_SIZE)
            # g.set_xticklabels(self.classes, size = TICKS_LABEL_SIZE)
            g.tick_params(axis = 'x', labelrotation = 45)
            #g.set_xticklabels(self.classes, size = TICKS_LABEL_SIZE)
            g.set_yticklabels(g.get_yticks(), size = TICKS_LABEL_SIZE)
            g.yaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))

            # g.set_xticks(ticks = range(len(self.classes)), labels = self.classes, size = TICKS_LABEL_SIZE)
            # # g.set_xticklabels(self.classes, size = TICKS_LABEL_SIZE)
            # g.tick_params(axis = 'x', labelrotation = 45)
            # g.set_xticks(range(len(self.classes)))
            # g.set_xticklabels(self.classes)
            # g.tick_params(axis = 'x', labelrotation = 45)
        else:
            g.set_xticklabels([])
        # g.set_xticklabels(list(map(int, g.get_xticks())),size = TICKS_LABEL_SIZE)
        # g.set_yticklabels(list(map(int, g.get_yticks())),size = TICKS_LABEL_SIZE)
        g.set_xlabel("class", fontsize = FIGURES_LABEL_SIZE)
        g.set_ylabel("nb_packets", fontsize = FIGURES_LABEL_SIZE)
        # g.set(xlabel = 'class', ylabel = 'nb_packets')
        # plt.legend(title = 'Class', labels = self.classes)
        plt.tight_layout()
        if xylin:
            plt.savefig(join(self.figure_output_dir, filename_prefix) + "_" + str(int(time.time())) + ".pdf",
                        format = "pdf")
        if xlog:
            plt.savefig(join(self.figure_output_dir, filename_prefix) + "_xlog_" + str(int(time.time())) + ".pdf",
                        format = "pdf")
        if xylog:
            plt.savefig(join(self.figure_output_dir, filename_prefix) + "_xylog_" + str(int(time.time())) + ".pdf",
                        format = "pdf")
        if ylog:
            plt.savefig(join(self.figure_output_dir, filename_prefix) + "_ylog_" + str(int(time.time())) + ".pdf",
                        format = "pdf")
        
    def _min_iat_distribution(self, df, filename_prefix, xticks = True, xlog = False, ylog = False, xylog = False, xylin = True):
        print("_min_iat_distribution")

        matplotlib.use('Agg')
        plt.clf()
        g = sns.boxplot(
            data = df,
            x = 'type',
            y = 'min_iat',
            flierprops={"marker": "x"},
        )
        g.set_rasterization_zorder(0)
        if xlog or xylog:
            plt.xscale('log')
        if ylog or xylog:
            plt.yscale('log')
        if xticks and len(self.classes) < 100:
            print(self.classes)
            g.set_xticks(ticks = range(len(self.classes)), labels = self.classes, size = TICKS_LABEL_SIZE)
            # g.set_xticklabels(self.classes, size = TICKS_LABEL_SIZE)
            g.tick_params(axis = 'x', labelrotation = 45)
            #g.set_xticklabels(self.classes, size = TICKS_LABEL_SIZE)
            g.set_yticklabels(g.get_yticks(), size = TICKS_LABEL_SIZE)
        else:
            g.set_xticklabels([])

        # g.set(xlabel = 'class', ylabel = 'min_iat')
        g.set_xlabel("class", fontsize = FIGURES_LABEL_SIZE)
        g.set_ylabel("Minimum IAT", fontsize = FIGURES_LABEL_SIZE)
        # plt.legend(title = 'Class', labels = self.classes)
        plt.tight_layout()
        if xylin:
            plt.savefig(join(self.figure_output_dir, filename_prefix) + "_" + str(int(time.time())) + ".pdf",
                        format = "pdf")
        if ylog:
            plt.yscale('log')
            plt.savefig(join(self.figure_output_dir, filename_prefix) + "_ylog_" + str(int(time.time())) + ".pdf",
                        format = "pdf")
        if xylog:
            plt.xscale('log')
            plt.savefig(join(self.figure_output_dir, filename_prefix) + "_xylog_" + str(int(time.time())) + ".pdf",
                        format = "pdf")
        if xlog:
            plt.yscale('linear')
            plt.savefig(join(self.figure_output_dir, filename_prefix) + "_xlog_" + str(int(time.time())) + ".pdf",
                        format = "pdf")
        
    def _max_size_distribution(self, df, filename_prefix, xticks = True, xlog = False, ylog = False, xylog = False, xylin = True):
        print("_max_size_distribution")
        from  matplotlib.ticker import FuncFormatter

        matplotlib.use('Agg')
        plt.clf()
        print(df.columns);
        g = sns.boxplot(
            data = df,
            x = 'type',
            y = 'max_length',
            flierprops={"marker": "x"},
        )
        g.set_rasterization_zorder(0)
        if xlog or xylog:
            plt.xscale('log')
        if ylog or xylog:
            plt.yscale('log')
        if xticks and len(self.classes) < 100:
            print(self.classes)
            g.set_xticks(ticks = range(len(self.classes)), labels = self.classes, size = TICKS_LABEL_SIZE)
            # g.set_xticklabels(self.classes, size = TICKS_LABEL_SIZE)
            g.tick_params(axis = 'x', labelrotation = 45)
            #g.set_xticklabels(self.classes, size = TICKS_LABEL_SIZE)
            g.set_yticklabels(g.get_yticks(), size = TICKS_LABEL_SIZE)
            g.yaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
        else:
            g.set_xticklabels([])

        g.set_xlabel("class", fontsize = FIGURES_LABEL_SIZE)
        g.set_ylabel("Maximum size", fontsize = FIGURES_LABEL_SIZE)
        plt.tight_layout()
        if xylin:
            plt.savefig(join(self.figure_output_dir, filename_prefix) + "_" + str(int(time.time())) + ".pdf",
                        format = "pdf")
        if ylog:
            plt.yscale('log')
            plt.savefig(join(self.figure_output_dir, filename_prefix) + "_ylog_" + str(int(time.time())) + ".pdf",
                        format = "pdf")
        if xylog:
            plt.xscale('log')
            plt.savefig(join(self.figure_output_dir, filename_prefix) + "_xylog_" + str(int(time.time())) + ".pdf",
                        format = "pdf")
        if xlog:
            plt.yscale('linear')
            plt.savefig(join(self.figure_output_dir, filename_prefix) + "_xlog_" + str(int(time.time())) + ".pdf",
                        format = "pdf")
        

    def _max_iat_distribution(self, df, filename_prefix, xticks = True, xlog = False, ylog = False, xylog = False, xylin = True):
        print("_max_iat_distribution")

        matplotlib.use('Agg')
        plt.clf()
        g = sns.boxplot(
            data = df,
            x = 'type',
            y = 'max_iat',
        )
        g.set_rasterization_zorder(0)
        if xticks and len(self.classes) < 100:
            print("classes=", self.classes)
            g.tick_params(axis = 'x', labelrotation = 45)
            g.set_xticks(range(len(self.classes)))
            g.set_xticklabels(self.classes, size = TICKS_LABEL_SIZE)
            g.set_yticklabels(g.get_yticks(),size = TICKS_LABEL_SIZE)
        # g.set(xlabel = 'class', ylabel = 'max_iat')
        g.set_xlabel("class", fontsize = FIGURES_LABEL_SIZE)
        g.set_ylabel("max_iat", fontsize = FIGURES_LABEL_SIZE)
        # plt.legend(title = 'Class', labels = self.classes)
        plt.tight_layout()
        if xylin:
            plt.savefig(join(self.figure_output_dir, filename_prefix) + "_" + str(int(time.time())) + ".pdf",
                        format = "pdf")
        plt.xscale('log')
        if xlog:
            plt.savefig(join(self.figure_output_dir, filename_prefix) + "_xlog_" + str(int(time.time())) + ".pdf",
                        format = "pdf")
        plt.yscale('log')
        if xylog:
            plt.savefig(join(self.figure_output_dir, filename_prefix) + "_xylog_" + str(int(time.time())) + ".pdf",
                        format = "pdf")
        plt.xscale('linear')
        if ylog:
            plt.savefig(join(self.figure_output_dir, filename_prefix) + "_ylog_" + str(int(time.time())) + ".pdf",
                        format = "pdf")
        
    def _class_distribution(self, df, filename_prefix, xticks = True, xlog = False, ylog = False, xylog = False, xylin = True):
        print("_class_distribution")
        matplotlib.use('Agg')
        length = 'sum_length'
        iat = 'sum_iat'
        if 'sum_iat' not in df.columns:
            iat = 'iat'
        if 'sum_length' not in df.columns:
            length = 'length'
        if 'type' not in df.columns:
            print("unexpected format", df.columns)
            
        plt.clf()
        #fig, axes = plt.subplots(figsize=(12, 9))
        ax = sns.lmplot(
            x = iat,
            y = length, 
            data = df,
            hue = 'type',
            fit_reg = False,
            legend = False,
            palette='viridis',
            aspect = 1.5,
            scatter_kws = {'rasterized': True},
            #ax=axes,
            #rug = True,
            #rug_kws=dict(rasterized=True)
        )
        #for tick in ax.xaxis.get_major_ticks():
        #    tick.label.set_fontsize(TICKS_LABEL_SIZE)
        #for tick in ax.yaxis.get_major_ticks():
        #    tick.label.set_fontsize(TICKS_LABEL_SIZE)
        ax.set_xticklabels(list(map(int, ax.ax.get_xticks())),size = TICKS_LABEL_SIZE, rotation = 45)
        ax.set_yticklabels(list(map(int, ax.ax.get_yticks())),size = TICKS_LABEL_SIZE)
        ax.set_xlabels("Duration", fontsize = FIGURES_LABEL_SIZE)
        # ax.set_xlabels("Minimum IAT", fontsize = FIGURES_LABEL_SIZE)
        ax.set_ylabels("Size", fontsize = FIGURES_LABEL_SIZE)
        #ax = sns.scatterplot(
        #    x = iat,
        #    y = length, 
        #    data = df,
        #    hue = 'type',
        #    legend = False,
        #    palette='viridis',
        #    #rug = True,
        #    #rug_kws=dict(rasterized=True)
        #)
        #plt.xlabel("length", fontsize = FIGURES_LABEL_SIZE)
        #plt.ylabel("sum_min_iat", fontsize = FIGURES_LABEL_SIZE)
        # g.set(xticks=np.arange(0,1050,50))        
        # ax.set_xticklabels(ax.ax.get_xticks(), size = TICKS_LABEL_SIZE)   
        # ax.set_yticklabels(ax.ax.get_yticks(), size = TICKS_LABEL_SIZE)   
        # ax.set(xlabel = 'duration', ylabel = 'sum_packet_size')
        if len(self.classes) < 100:
            plt.legend(title = 'class', labels = self.classes, fontsize = FIGURES_LEGEND_SIZE)
        #ax.fig.set_rasterization_zorder(0)
        #fig = ax.fig.draw()
        #points = fig.axes[0].collections[0]
        #points.set_rasterized(True)
        plt.tight_layout()
        if xylin:
            plt.savefig(join(self.figure_output_dir, filename_prefix) + "_" + str(int(time.time())) + '.pdf',
                        format = 'pdf')
        plt.xscale('log')
        if xlog:
            plt.savefig(join(self.figure_output_dir, filename_prefix) + "_xlog_" + str(int(time.time())) + ".pdf",
                        format = "pdf")
        plt.yscale('log')
        if xylog:
            plt.savefig(join(self.figure_output_dir, filename_prefix) + "_xylog_" + str(int(time.time())) + ".pdf",
                        format = "pdf")
        plt.xscale('linear')
        if ylog:
            plt.savefig(join(self.figure_output_dir, filename_prefix) + "_ylog_" + str(int(time.time())) + ".pdf",
                        format = "pdf")

    def _size_std_max_class_distribution(self, df, filename_prefix, xticks = True, xlog = False, ylog = False, xylog = False, xylin = True):
        print("_class_distribution")
        matplotlib.use('Agg')
        x = 'max_length'
        y = 'std_length'
            
        plt.clf()
        #fig, axes = plt.subplots(figsize=(12, 9))
        ax = sns.lmplot(
            x = x,
            y = y, 
            data = df,
            hue = 'type',
            fit_reg = False,
            legend = False,
            palette='viridis',
            aspect = 1.5,
            scatter_kws = {'rasterized': True},
            #ax=axes,
            #rug = True,
            #rug_kws=dict(rasterized=True)
        )
        #for tick in ax.xaxis.get_major_ticks():
        #    tick.label.set_fontsize(TICKS_LABEL_SIZE)
        #for tick in ax.yaxis.get_major_ticks():
        #    tick.label.set_fontsize(TICKS_LABEL_SIZE)
        ax.set_xticklabels(list(map(int, ax.ax.get_xticks())),size = TICKS_LABEL_SIZE, rotation = 45)
        ax.set_yticklabels(list(map(int, ax.ax.get_yticks())),size = TICKS_LABEL_SIZE)
        ax.set_xlabels("max_size", fontsize = FIGURES_LABEL_SIZE)
        # ax.set_xlabels("Minimum IAT", fontsize = FIGURES_LABEL_SIZE)
        ax.set_ylabels("std_size", fontsize = FIGURES_LABEL_SIZE)
        #ax = sns.scatterplot(
        #    x = iat,
        #    y = length, 
        #    data = df,
        #    hue = 'type',
        #    legend = False,
        #    palette='viridis',
        #    #rug = True,
        #    #rug_kws=dict(rasterized=True)
        #)
        #plt.xlabel("length", fontsize = FIGURES_LABEL_SIZE)
        #plt.ylabel("sum_min_iat", fontsize = FIGURES_LABEL_SIZE)
        # g.set(xticks=np.arange(0,1050,50))        
        # ax.set_xticklabels(ax.ax.get_xticks(), size = TICKS_LABEL_SIZE)   
        # ax.set_yticklabels(ax.ax.get_yticks(), size = TICKS_LABEL_SIZE)   
        # ax.set(xlabel = 'duration', ylabel = 'sum_packet_size')
        if len(self.classes) < 100:
            plt.legend(title = 'class', labels = self.classes, fontsize = FIGURES_LEGEND_SIZE)
        #ax.fig.set_rasterization_zorder(0)
        #fig = ax.fig.draw()
        #points = fig.axes[0].collections[0]
        #points.set_rasterized(True)
        plt.tight_layout()
        if xylin:
            plt.savefig(join(self.figure_output_dir, filename_prefix) + "_" + str(int(time.time())) + '.pdf',
                        format = 'pdf')
        plt.xscale('log')
        if xlog:
            plt.savefig(join(self.figure_output_dir, filename_prefix) + "_xlog_" + str(int(time.time())) + ".pdf",
                        format = "pdf")
        plt.yscale('log')
        if xylog:
            plt.savefig(join(self.figure_output_dir, filename_prefix) + "_xylog_" + str(int(time.time())) + ".pdf",
                        format = "pdf")
        plt.xscale('linear')
        if ylog:
            plt.savefig(join(self.figure_output_dir, filename_prefix) + "_ylog_" + str(int(time.time())) + ".pdf",
                        format = "pdf")

    def save_results(self):
        for i in EncryptedTrafficClassifierIterator(self.flow_ids):
            print("features: ", self.X_train_flows[i].columns)
            break
        self.classification_results.to_csv(self.results_filename, sep = ",", header = True, index = True)

    
    ########################################
    # Preprocessing
    ########################################
    def preprocessing(self, X_train, y_train, X_test, y_test, results, feats):
        print("preprocessing")
        from sklearn import set_config
        set_config(display = "diagram")

        r = tuple(results[0])
        df = X_train[r]#[feats]
        numeric_features = df.select_dtypes(exclude = "object").columns
        
        numeric_transformer = Pipeline(
            steps = [
                #('imputer', SimpleImputer(missing_values = np.nan,
                #                          strategy = "constant")), #"mean"
                # ('imputer', IterativeImputer()),
                ("scaler", StandardScaler()), 
                # ("pca", PCA()),
                # ("poly_feat", PolynomialFeatures())
            ]
        )
    
        #categorical_transformer = Pipeline(
        #        steps = [
        #            ('imputer', SimpleImputer(strategy = "constant")), #"mean"
        #            ("encoder", OneHotEncoder(drop = 'first',
        #                                      handle_unknown = 'ignore')),
        #            # ("pca", PCA()),
        #        ]
        #)
    
        preprocessor = ColumnTransformer(
            transformers = [
                ("num", numeric_transformer, numeric_features),
                #("cat", categorical_transformer, categorical_features),
            ]
        )
        
        print(preprocessor)
        
        X_train_fitted = {}
        X_test_fitted = {}
        for i in EncryptedTrafficClassifierIterator(results):
            X_train_fitted[i] = preprocessor.fit_transform(X_train[i])
            X_test_fitted[i] = preprocessor.fit_transform(X_test[i])
            
        return X_train_fitted, X_test_fitted
    
    ########################################
    # RandomForest
    ########################################
    def RF_predict(self, X_train, y_train, X_test, y_test):
        print("RF_predict")
        
        # rf_test_isolated_score = {}
        # rf_y_test_isolated_predicted = {}

        rf_train_score = {}
        rf_y_train_predicted = {}
        rf_test_score = {}
        rf_y_test_predicted = {}
        rf_best_params = {}
        rf_features_importance = {}
        should_save = {}
        self.__set_rf_pickle_filename()
        
        for pkt in self.nb_packets_per_flow:
            should_save[pkt] = True
            if isfile(self.rf_output[pkt]):
                print("Loading previously saved results for", pkt, "packets in", self.rf_output[pkt])
                should_save[pkt] = False
                rf_regr = None
                with open(self.rf_output[pkt], "rb") as f:
                    _X_train = pickle.load(f)
                    _y_train = pickle.load(f)
                    _rf_train_score = pickle.load(f)
                    _rf_y_train_predicted = pickle.load(f)
                    _X_test = pickle.load(f)
                    _y_test = pickle.load(f)
                    _rf_test_score = pickle.load(f)
                    _rf_y_test_predicted = pickle.load(f)
                    # _rf_test_isolated_score = pickle.load(f)
                    # _rf_y_test_isolated_predicted = pickle.load(f)
                    _rf_best_params = pickle.load(f)
                    _rf_features_importance = pickle.load(f)
                for fold in range(self.nb_folds):
                    i = pkt, fold
                    X_train[i] = _X_train[i]
                    y_train[i] = _y_train[i]
                    rf_train_score[i] = _rf_train_score[i]
                    rf_y_train_predicted[i] = _rf_y_train_predicted[i]
                    X_test[i] = _X_test[i]
                    y_test[i] = _y_test[i]
                    rf_test_score[i] = _rf_test_score[i]
                    rf_y_test_predicted[i] = _rf_y_test_predicted[i]
                    # rf_test_isolated_score[i] = _rf_test_isolated_score[i]
                    # rf_y_test_isolated_predicted[i] = _rf_y_test_isolated_predicted[i]
                    rf_best_params[i] = _rf_best_params[i]
                    rf_features_importance[i] = _rf_features_importance[i]
                    
        rf_grid_search = {}
        nb_cores_to_use = max(1, os.cpu_count())
        rf_pipeline_logistic = Pipeline(
            steps = [                
                ("rf", RandomForestClassifier(n_jobs = nb_cores_to_use))
            ]
        )
        
        rf_param_grid = {
            "rf__n_estimators": range(150, 400,50) #range(10)
            # "rf__n_estimators": [10]
        }

        for i in EncryptedTrafficClassifierIterator(self.flow_ids):
            if self.force_rf_classification == False and i in rf_train_score.keys():
                # print("skipping", i)
                continue
            rf_grid_search[i] = GridSearchCV(rf_pipeline_logistic,
                                             param_grid = rf_param_grid,
                                             cv = 2,
                                             verbose = 3)
            
            rf_regr = {}
            print("==" +  str(i) + "==")
            X = X_train[i]  
            y = y_train[i]
            rf_regr[i] = rf_grid_search[i].fit(X, y)
            print(i, rf_regr[i].best_params_)
            
            rf_train_score[i] = rf_regr[i].score(X_train[i], y_train[i])
            rf_y_train_predicted[i] = rf_regr[i].predict(X_train[i])
            rf_test_score[i] = rf_regr[i].score(X_test[i], y_test[i])
            rf_y_test_predicted[i] = rf_regr[i].predict(X_test[i])
            
            # rf_test_isolated_score[i] = rf_regr[i].score(self.X_test_isolated_flows, self.y_test_isolated_flows)
            # rf_y_test_isolated_predicted[i] = rf_regr[i].predict(self.X_test_isolated_flows)
            # print("rf_y_test_isolated_predicted[i] =", rf_y_test_isolated_predicted[i])
            # print("rf_y_test_isolated_predicted test score for (%s, %d) = %f" % (i[1], i[0], rf_test_isolated_score[i]))
            rf_best_params[i] = rf_regr[i].best_params_
            rf_features_importance[i] = [] #rf_regr[i].best_estimator_.named_steps["rf"].feature_importances_
            print("test score for (%s, %d) = %f" % (i[1], i[0], rf_test_score[i]))
            
            print("Feature ranking:")
            importances = rf_regr[i].best_estimator_.named_steps["rf"].feature_importances_
            std = np.std([tree.feature_importances_ for tree in rf_regr[i].best_estimator_.named_steps["rf"].estimators_],
                         axis=0)
            indices = np.argsort(importances)[::-1]
            _features = {}
            for f in range(self.X_train_flows[i].shape[1]):
                print("%d. feature %s (%f)" % (f + 1, self.X_train_flows[i].columns[indices[f]], importances[indices[f]]))
                rf_features_importance[i].append((self.X_train_flows[i].columns[indices[f]], importances[indices[f]]))
                _features[f] = (self.X_train_flows[i].columns[indices[f]], importances[indices[f]])
            # print(_features)
            # self.classification_results.loc[(self.classification_results['nb_packets'] == pkt) & (self.classification_results['fold_id'] == fold), 'feature_ranking'] = [_features]

        for pkt in self.nb_packets_per_flow:
            if should_save[pkt]:
                _X_train = {}
                _y_train = {}
                _rf_train_score = {}
                _rf_y_train_predicted = {}
                _X_test = {}
                _y_test = {}
                _rf_test_score = {}
                _rf_y_test_predicted = {}
                # _rf_test_isolated_score = {}
                # _rf_y_test_isolated_predicted = {}
                _rf_best_params = {}
                _rf_features_importance = {}
                for fold in range(self.nb_folds):
                    i = pkt, fold 
                    _X_train[i] = X_train[i]
                    _y_train[i] = y_train[i]
                    _rf_train_score[i] = rf_train_score[i]
                    _rf_y_train_predicted[i] = rf_y_train_predicted[i]
                    _X_test[i] = X_test[i]
                    _y_test[i] = y_test[i]
                    _rf_test_score[i] = rf_test_score[i]
                    _rf_y_test_predicted[i] = rf_y_test_predicted[i]
                    # _rf_test_isolated_score[i] = rf_test_isolated_score[i]
                    # _rf_y_test_isolated_predicted[i] = rf_y_test_isolated_predicted[i]
                    _rf_best_params[i] = rf_best_params[i]
                    _rf_features_importance[i] = rf_features_importance[i]
                try:
                    print("Saving results for", pkt, "packets in", self.rf_output[pkt])
                    with open(self.rf_output[pkt], "wb") as f:
                        pickle.dump(_X_train, f)
                        pickle.dump(_y_train, f)
                        pickle.dump(_rf_train_score, f)
                        pickle.dump(_rf_y_train_predicted, f)
                        pickle.dump(_X_test, f)
                        pickle.dump(_y_test, f)
                        pickle.dump(_rf_test_score, f)
                        pickle.dump(_rf_y_test_predicted, f)
                        # pickle.dump(_rf_test_isolated_score, f)
                        # pickle.dump(_rf_y_test_isolated_predicted, f)
                        pickle.dump(_rf_best_params, f)
                        pickle.dump(_rf_features_importance, f)
                except Exception as e:
                    print("Exception", e)
                    pass

        for i in EncryptedTrafficClassifierIterator(self.flow_ids):
            pkt, fold = i
            memfile = io.BytesIO()
            np.save(memfile, rf_features_importance[i])
            serialized = memfile.getvalue()
            _s = json.dumps(serialized.decode('latin-1'))
            _r = pd.DataFrame(
                {
                    'nb_packets': [pkt],
                    'fold_id': [fold],
                    'rf_train_score': [rf_train_score[i]],
                    'rf_test_score': [rf_test_score[i]],
                    # 'rf_test_isolated_score': [rf_test_isolated_score],
                    'rf_best_params': [rf_best_params[i]],
                    'rf_features_importance': [_s],
                    # 'gb_train_score': [np.nan],
                    # 'gb_test_score': [np.nan],
                    # 'xg_train_score': [np.nan],
                    # 'xg_test_score': [np.nan],
                }
            )
            self.classification_results = pd.concat([_r, self.classification_results])
        
        # return rf_regr, rf_y_train_predicted, rf_y_test_predicted, rf_y_test_isolated_predicted
        return rf_regr, rf_y_train_predicted, rf_y_test_predicted
    
    ########################################
    # XGBoost
    ########################################
    def XGBoost_predict(self, X_train, y_train, X_test, y_test):
        print("XGBoost_predict")

        xg_train_score = {}
        xg_y_train_predicted = {}
        xg_test_score = {}
        xg_y_test_predicted = {}
        should_save = {}
        self.__set_xg_pickle_filename()
        
        for pkt in self.nb_packets_per_flow:
            should_save[pkt] = True
            if isfile(self.xg_output[pkt]):
                print("Loading previously saved results for", pkt, "packets in", self.xg_output[pkt])
                should_save[pkt] = False
                xg_regr = None
                with open(self.xg_output[pkt], "rb") as f:
                    _X_train = pickle.load(f)
                    _y_train = pickle.load(f)
                    _xg_train_score = pickle.load(f)
                    _xg_y_train_predicted = pickle.load(f)
                    _X_test = pickle.load(f)
                    _y_test = pickle.load(f)
                    _xg_test_score = pickle.load(f)
                    _xg_y_test_predicted = pickle.load(f)
                for fold in range(self.nb_folds):
                    i = pkt, fold
                    X_train[i] = _X_train[i]
                    y_train[i] = _y_train[i]
                    xg_train_score[i] = _xg_train_score[i]
                    xg_y_train_predicted[i] = _xg_y_train_predicted[i]
                    X_test[i] = _X_test[i]
                    y_test[i] = _y_test[i]
                    xg_test_score[i] = _xg_test_score[i]
                    xg_y_test_predicted[i] = _xg_y_test_predicted[i]
        
        xg_model = {}
        
        for i in EncryptedTrafficClassifierIterator(self.flow_ids):
            nb_features = len(X_train[i].columns)
            if i in xg_train_score.keys():
                # print("skipping", i)
                continue
            xg_model[i] = XGBClassifier()

            print("==",i,"==")
            try:
                start_time = time.time()
                xg_model[i].fit(X_train[i], y_train[i])
                print(f"  finished after {time.time() - start_time} seconds")
            except ValueError as e:
                print(e)
                pass
        
            xg_y_train_predicted[i] = xg_model[i].predict(X_train[i])
            xg_y_test_predicted[i] = xg_model[i].predict(X_test[i])
            xg_train_score[i] = xg_model[i].score(X_train[i], y_train[i])
            xg_test_score[i] = xg_model[i].score(X_test[i], y_test[i])
        
        for pkt in self.nb_packets_per_flow:
            if should_save[pkt]:
                _X_train = {}
                _y_train = {}
                _xg_train_score = {}
                _xg_y_train_predicted = {}
                _X_test = {}
                _y_test = {}
                _xg_test_score = {}
                _xg_y_test_predicted = {}
                for fold in range(self.nb_folds):
                    i = pkt, fold 
                    _X_train[i] = X_train[i]
                    _y_train[i] = y_train[i]
                    _xg_train_score[i] = xg_train_score[i]
                    _xg_y_train_predicted[i] = xg_y_train_predicted[i]
                    _X_test[i] = X_test[i]
                    _y_test[i] = y_test[i]
                    _xg_test_score[i] = xg_test_score[i]
                    _xg_y_test_predicted[i] = xg_y_test_predicted[i]
                try:
                    print("Saving results for", pkt, "packets in", self.xg_output[pkt])
                    with open(self.xg_output[pkt], "wb") as f:
                        pickle.dump(_X_train, f)
                        pickle.dump(_y_train, f)
                        pickle.dump(_xg_train_score, f)
                        pickle.dump(_xg_y_train_predicted, f)
                        pickle.dump(_X_test, f)
                        pickle.dump(_y_test, f)
                        pickle.dump(_xg_test_score, f)
                        pickle.dump(_xg_y_test_predicted, f)
                except Exception as e:
                    print("Exception", e)
                    pass
                
        for i in EncryptedTrafficClassifierIterator(self.flow_ids):
            pkt, fold = i
            _r = pd.DataFrame(
                {
                    'nb_packets': [pkt],
                    'fold_id': [fold],
                    'xg_train_score': [xg_train_score[i]],
                    'xg_test_score': [xg_test_score[i]],
                    'xg_nb_features': [nb_features],
                }
            )
            self.classification_results = pd.concat([_r, self.classification_results])

        return xg_model, xg_y_train_predicted, xg_y_test_predicted
    
if __name__ == "__main__":
    sys.exit(1)