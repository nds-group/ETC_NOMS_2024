#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import os
from os.path import isfile, join
import sys
import time

import numpy as np

import pandas as pd 

from scipy.stats import kurtosis, skew

from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
    
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer

from encrypted_traffic_classification import EncryptedTrafficClassifier, EncryptedTrafficClassifierIterator

REGENERATE_FLOWS_DATA = False

TEST_FLOWS = True
TEST_PACKETS = False

########################################
# Data preparation: convert RAW data
########################################
class UCDavisQuicClassifier(EncryptedTrafficClassifier):
    def __init__(self, nb_folds, nb_packets_per_flow):
        super().__init__(
            nb_folds= nb_folds,
            nb_packets_per_flow = nb_packets_per_flow,
            filename_prefix = "ucdavis_quic",
            # processed_data_output_dir = "ucdavis_quic_output/",
            processed_data_output_dir = "ucdavis_quic_output_addendum/",
            data_dir = "data/ucdavis_quic_pretraining/"            
        )

        pools = [tuple(pool) for pool in [self.nb_packets_per_flow, range(self.nb_folds)]]
        result = [[]]
        for pool in pools:
            result = [x + [y] for x in result for y in pool]
        self.flow_ids = result

        pools = [tuple(pool) for pool in [range(self.nb_folds)]]
        result = [[]]
        for pool in pools:
            result = [x + [y] for x in result for y in pool]
        self.packet_ids = result

    def data_preparation(self):
        print("data_prepation")
        # limit = 100000

        start_time = time.time()
        traffic_type = 0
        subdirs = os.listdir(self.data_dir)
        dfs = []
        dfs_memory_usage = 0
        df = pd.DataFrame()
        for d in subdirs:
            self.classes[traffic_type] = d
            # print(d)
            files = os.listdir(self.data_dir + d)
            # i = 0
            for filename in files:
                f = self.data_dir + d + "/" + filename
                # print(filename)
                file_df = pd.read_csv(f, 
                                      delimiter = '\t',
                                      names = ['timestamp', 'time_delta', 'packet_size', 'direction']
                                      )
                file_df['type'] = traffic_type
                file_df['src'] = filename
                dfs.append(file_df)
                # short hack: there is a trade-off between memory usage and speed
                # as much as possible DataFrames are insert in the dfs numpy array
                # which is much faster than pandas.concat, but if the dfs array grows
                # too big the process will be killed by oom_killer on Linux, so
                # once memory is above 2GB concat what we have in dfs
                dfs_memory_usage += file_df.memory_usage(deep = True).sum()
                if dfs_memory_usage > 2 * 1024 * 1024 * 1024:
                    df = pd.concat([df, *dfs])
                    del dfs
                    dfs = []
                    dfs_memory_usage = 0
                    
                # i += 1
                # if i >= limit:
                #     break
    
                traffic_type += 1
        
        df = pd.concat([df, *dfs])
        df = df.fillna(0)

        del dfs
        print(f"  flows data loaded in {time.time() - start_time} seconds")
        filename = self.filename_prefix + ".pickle"
        self._generate_data_folds(df, filename)
        
        print(df.columns)
        print(df.describe())
        print(df.info)
        print(df.shape)

    def __get_flow_df(self, flow_df, traffic_type):
        # filter by direction
        #file_df = file_df[file_df['direction'] == 1]
        _df = flow_df['packet_size']
        packet_size = sum(_df)
        min_packet_size = np.min(_df)
        max_packet_size = np.max(_df)
        mean_packet_size = np.mean(_df)
        median_packet_size = np.median(_df)
        std_packet_size = np.std(_df)
        Q1_packet_size = np.quantile(_df, 0.25)
        Q3_packet_size = np.quantile(_df, 0.75)
        _a = list(_df)
        skew_packet_size = skew(_a)
        kurt_packet_size = kurtosis(_a)
                
        min_time_delta = np.min(flow_df[flow_df['time_delta'] > 0]['time_delta'])
        _df = flow_df['time_delta']
        time_delta = sum(_df)
        max_time_delta = np.max(_df)
        mean_time_delta = np.mean(_df)
        median_time_delta = np.median(_df)
        std_time_delta = np.std(_df)
        Q1_iat = np.quantile(_df, 0.25)
        Q3_iat = np.quantile(_df, 0.75)
        _a = list(_df)
        skew_iat = skew(_a)
        kurt_iat = kurtosis(_a)
        data = {
            'sum_iat': [time_delta],
            'sum_length': [packet_size],
            'min_length': [min_packet_size],
            'max_length': [max_packet_size],
            'mean_length': [mean_packet_size],
            'median_length': [median_packet_size],
            'std_length': [std_packet_size],
            '1stQ_length': [Q1_packet_size],
            '3stQ_length': [Q3_packet_size],
            'skew_length': [skew_packet_size],
            'kurt_length': [kurt_packet_size],
            'min_iat': [min_time_delta],
            'max_iat': [max_time_delta],
            'mean_iat': [mean_time_delta],
            'median_iat': [median_time_delta],
            'std_iat': [std_time_delta],
            '1stQ_iat': [Q1_iat],
            '3stQ_iat': [Q3_iat],
            'skew_iat': [skew_iat],
            'kurt_iat': [kurt_iat],
            'nb_packets': [len(flow_df)],
            'type': [traffic_type],
            #'direction': [flow_df['direction']]
        }
        _df = pd.DataFrame(data = data)
        _df.fillna(0)
        return _df

    def packets2flows_nofold(self):
        print("packets2flows_nofold")
        traffic_type = 0
        subdirs = os.listdir(self.data_dir)
        dfs = {}
        flow_id = 0
        for n in self.nb_packets_per_flow:
            dfs[n] = []
        for d in subdirs:
            self.classes[traffic_type] = d
            files = os.listdir(self.data_dir + d)
            for filename in files:
                f = self.data_dir + d + "/" + filename
                # print(filename)
                file_df = pd.read_csv(f, 
                                      delimiter = '\t',
                                      names = ['timestamp', 'time_delta', 'packet_size', 'direction']
                                      )
                file_df['type'] = traffic_type
                file_df['src'] = filename
                file_df['flow_id'] = flow_id
                flow_id += 1
                
                dfs[n].append(file_df.head(n = n))
            
            traffic_type += 1
        for n in self.nb_packets_per_flow:
            df = pd.concat(dfs[n])        
            self._pickle_dump(df, "for_signatures_" + str(n) + "_flows_" + self.pickle_filename_suffix)
            
    def _get_flows_with_all_packets(self):
        print("_get_flows_with_all_packets")
        traffic_type = 0
        subdirs = os.listdir(self.data_dir)
        _df = []
        start_time = time.time()
        for d in subdirs:
            self.classes[traffic_type] = d
            files = os.listdir(self.data_dir + d)
            for filename in files:
                f = self.data_dir + d + "/" + filename
                file_df = pd.read_csv(
                    f, 
                    delimiter = '\t',
                    names = ['timestamp', 'time_delta', 'packet_size', 'direction']
                )
                file_df['type'] = traffic_type
                file_df['src'] = filename
                _flow_df = self.__get_flow_df(file_df, traffic_type)
                
                _df.append(_flow_df)
                # uncomment for debugging behavior with a single file
                # break
            traffic_type += 1
        print("  processing took ", time.time() - start_time, "seconds.")
        _df = pd.concat(_df)
        _df =_df.fillna(0)
        return _df
        
    def packets2flows(self):
        print("packets2flows")
        traffic_type = 0
        subdirs = sorted(os.listdir(self.data_dir))
        dfs = {}
        for n in self.nb_packets_per_flow:
            dfs[n] = []
        idx_d = 0
        for d in subdirs:
            start_time = time.time()
            print("Processing directory #%d/%d: %s" % (idx_d, len(subdirs), d))
            idx_d += 1
            self.classes[traffic_type] = d
            files = sorted(os.listdir(self.data_dir + d))
            for filename in files:
                f = self.data_dir + d + "/" + filename
                # print(filename)
                file_df = pd.read_csv(f, 
                                      delimiter = '\t',
                                      names = ['timestamp', 'time_delta', 'packet_size', 'direction']
                                      )
                file_df['type'] = traffic_type
                file_df['src'] = filename
                for n in self.nb_packets_per_flow:
                    _flow_df = self.__get_flow_df(file_df.head(n = n), traffic_type)
                
                    dfs[n].append(_flow_df)
            print("  ", d, "processed in ", time.time() - start_time, "seconds.")            
            
            traffic_type += 1
            
        for n in self.nb_packets_per_flow:
            df = pd.concat(dfs[n])
            df = df.fillna(0)
            seed = 42
            filename = self.filename_prefix + "_" + str(n) + ".pickle"
            # filename = self.filename_prefix + "_flows_" + str(n) + ".pickle"
            self._generate_data_folds(df, filename)

    def load_flows_nofold(self):
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits = self.nb_folds, shuffle = True, random_state = self.random_seed)
        for n in [4]:
            df = self._load_pickle("for_signatures_" + str(n) + "_flows_" + self.pickle_filename_suffix)
            X = df.drop('type', axis = 1)
            y = df['type']
        
            for _i, (train_index, test_index) in enumerate(skf.split(X, y)):
                i = _i, n
                self.X_train_flows[i] = X.iloc[train_index].fillna(0)
                self.y_train_flows[i] = y.iloc[train_index].fillna(0)
                self.X_test_flows[i] = X.iloc[test_index].fillna(0)
                self.y_test_flows[i] = y.iloc[test_index].fillna(0)
        
    def load_packets(self, suffix):
        print("load_packets", suffix)
        start_time = time.time()
        
        for fold in EncryptedTrafficClassifierIterator(self.packet_ids):
            name = str(fold) + "_X_train_" + suffix
            self.X_train_packets[fold] = self._load_pickle(name)
            
            name = str(fold) + "_y_train_" + suffix
            self.y_train_packets[fold] = self._load_pickle(name)
            
            name = str(fold) + "_X_test_" + suffix
            self.X_test_packets[fold] = self._load_pickle(name)
            
            name = str(fold) + "_y_test_" + suffix
            self.y_test_packets[fold] = self._load_pickle(name)
        print(f"  packets data loaded in {time.time() - start_time} seconds")

    def LogReg_predict(X_train, y_train, X_test, y_test, ):    
        lr = LogisticRegression(penalty='none', solver='newton-cg')
        lr.fit(X, y)
        metrics.plot_roc_curve(lr, X, y)
        plt.plot([0, 1], [0, 1], "-")
        plt.show()
        
        
        display(metrics.roc_auc_score(y, lr.predict_proba(X)[:, 1]))
        display(metrics.confusion_matrix(y, lr.predict_proba(X)[:, 1]>0.5))
        
        print("train score = %f" % (lr.score(X_train, y_train)))
        print("test score = %f" % (lr.score(X_test, y_test)))
        
########################################
# Entry point
########################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='ucdavis_quic_traffic_classifier',
        description='Classify packets or flows from UCDavis QUIC dataset',
        epilog=''
    )
    parser.add_argument('-p', '--nb_packets', action = 'append', type = int, required = True)
    parser.add_argument('-c', '--classifier', action = 'append', type = str)
    parser.add_argument('-f', '--nb_folds', action = 'store', default = 12, type = int)
    parser.add_argument('-v', '--visualization', action = 'store_true', required = False, default = False)
    parser.add_argument('-r', '--report', action = 'store_true', required = False, default = False)
    parser.add_argument('-F', '--force_rf_classification', action = 'store_true', required = False, default = False)
    args = parser.parse_args(sys.argv[1:])

    VISUALIZATION_ENABLED = False
    if args.visualization == True:
        VISUALIZATION_ENABLED = True

    RF_ENABLED = False
    GB_ENABLED = False
    XG_ENABLED = False

    for c in args.classifier:
        print(c)
        if c == "rf":
            RF_ENABLED = True
        elif c == "gb":
            GB_ENABLED = True
        elif c == "xg":
            XG_ENABLED = True
        else:
            print("Unknown classifier", c)
            
    classifier = UCDavisQuicClassifier(
        nb_folds = args.nb_folds,
        nb_packets_per_flow = args.nb_packets
    )
    FORCE_RF_CLASSIFICATION = False
    if args.force_rf_classification == True:
        classifier.force_rf_classification = True

    classifier.all_classes = [
        "Google Doc",
        "Google Drive",
        "Google Music",
        "Google Search",
        "Youtube",
    ]
    if REGENERATE_DATA_FOR_SIGNATURES:
        classifier.packets2flows_nofold()
        sys.exit(1)

    if REGENERATE_FLOWS_DATA:
        classifier.load_packets(classifier.pickle_filename_suffix)
        classifier.packets2flows()
        sys.exit(1)
        
    # data preparation, convert raw data to pickle file, split in StratifiedKFold X_train, y_train, X_test, y_test
    if not classifier.data_prepared():        
        # classifier.data_preparation()
        classifier.packets2flows()
    else:
        subdirs = os.listdir(classifier.data_dir)
        traffic_type = 0
        for d in subdirs:
            classifier.classes[traffic_type] = d
            traffic_type += 1
            
    non_needed_features = [
        'timestamp',
        'direction',
        # "nb_packets"
    ]
    # non_needed_features += [
    #     'min_length', 'max_length',
    #     'mean_length', 'std_length',
    #     '1stQ_length',
    #     '3stQ_length',
    #     'skew_length',
    #     'kurt_length',
    #     'min_iat',
    #     'max_iat', 'mean_iat', 'std_iat',
    #     '1stQ_iat',
    #     '3stQ_iat',
    #     'skew_iat',
    #     'kurt_iat',
    # ]
    
    if TEST_PACKETS:
        classifier.load_packets(classifier.pickle_filename_suffix)
        classifier.cleanup_data(classifier.X_train_packets,
                                classifier.y_train_packets,
                                classifier.X_test_packets,
                                classifier.y_test_packets,
                                classifier.packet_ids,
                                non_needed_features)

    if TEST_FLOWS:
        classifier.load_flows()
        _c = classifier.y_train_flows[(classifier.nb_packets_per_flow[0], 0)].unique()
        classifier.classes = []
        for i in _c:
            classifier.classes.append(classifier.all_classes[i])
        classifier.cleanup_data(classifier.X_train_flows,
                                classifier.y_train_flows,
                                classifier.X_test_flows,
                                classifier.y_test_flows,
                                classifier.flow_ids,
                                non_needed_features)
    
    # __correlation()
    if args.report == True:
        classifier._viz(distribution = 0, class_distribution = -1, nb_packets = -1, min_iat = -1, max_iat = -1)
        for n in classifier.nb_packets_per_flow:
            if n == 4:
                classifier._viz(distribution = -1, class_distribution = 11, nb_packets = 0, min_iat = -1, max_iat = -1)
            elif n == 8:
                classifier._viz(distribution = -1, class_distribution = 11, nb_packets = 0, min_iat = -1, max_iat = -1)
            elif n == 600000:
                classifier._viz(distribution = -1, class_distribution = 0, nb_packets = 0, min_iat = 1, max_iat = -1)
        sys.exit(1)
    if VISUALIZATION_ENABLED:       
        pkt = classifier.nb_packets_per_flow[0]
        fold = 0
        _i = pkt, fold
        _df1 = classifier.X_train_flows[_i].copy()
        _df1['type'] = classifier.y_train_flows[_i]
        _df2 = classifier.X_test_flows[_i].copy()
        _df2['type'] = classifier.y_test_flows[_i]
        _df = pd.concat([_df1, _df2])
        _df.reset_index()
        print(_df.shape)
        print(_df['type'].value_counts())
        classifier._distribution(_df, classifier.filename_prefix + "_flows_class_split_" + str(pkt) + '_pkt')
        # classifier._class_distribution(_df, classifier.filename_prefix + '_flows_distribution_' + str(pkt) + '_pkt')
        # classifier._nb_packets_distribution(_df, classifier.filename_prefix + "_flows_nb_packets_distribution_" + str(pkt) + '_pkt')
        classifier._min_iat_distribution(_df, classifier.filename_prefix + "_flows_min_iat_distribution_" + str(pkt) + '_pkt')


    all_features_flows = ['sum_iat', 'sum_length', 'min_length', 'max_length',
                          'mean_length', 'median_length', 'std_length',
                          '1stQ_length',
                          '3stQ_length',
                          'skew_length',
                          'kurt_length',
                          'min_iat',
                          'max_iat', 'mean_iat', 'median_iat', 'std_iat',
                          '1stQ_iat',
                          '3stQ_iat',
                          'skew_iat',
                          'kurt_iat',
                          'nb_packets'
                          ]

    basic_features_flows = ['sum_iat', 'sum_length', 'min_length', 'max_length',
                          'mean_length', 'std_length',
                          'min_iat',
                          'max_iat', 'mean_iat', 'std_iat', 'nb_packets'
                          ]

    all_features_packets = ['sum_iat', 'sum_length']
    # feats_flows = all_features_flows
    feats_flows = basic_features_flows
    feats_packets = all_features_packets
    
    # classification based on flows
    if TEST_FLOWS:
        # classifier.X_train_flows_fitted, classifier.X_test_flows_fitted = classifier.preprocessing(
        #     classifier.X_train_flows,
        #     classifier.y_train_flows,
        #     classifier.X_test_flows,
        #     classifier.y_test_flows,
        #     classifier.flow_ids,
        #     feats_flows
        # )
        classifier.X_train_flows_fitted = classifier.X_train_flows
        classifier.X_test_flows_fitted = classifier.X_test_flows
        if RF_ENABLED:
            rf_regr_flows, rf_y_train_flows_predicted, rf_y_test_flows_predicted, rf_y_test_flows_isolated_predicted = classifier.RF_predict(
                classifier.X_train_flows_fitted,
                classifier.y_train_flows,
                classifier.X_test_flows_fitted,
                classifier.y_test_flows
            )
            rf_cm_dict_flows, output = classifier.confusion_matrix(rf_regr_flows,
                                                                   classifier.y_test_flows,
                                                                   rf_y_test_flows_predicted,
                                                                   classifier.flow_ids,
                                                                   "rf"
                                                                   )
            print(output)
            rf_f1_scores_flows, output = classifier.get_F1_score(rf_cm_dict_flows,
                                                                 classifier.y_test_flows,
                                                                 rf_y_test_flows_predicted,
                                                                 classifier.flow_ids,
                                                                 "rf_flows")
            print(output)
            avg_scores, output = classifier.avg_f1_scores(rf_f1_scores_flows, classifier.flow_ids)
            print(output)
            
            ######
    
            cm_dict = {}
            cm_dict_normalized = {}
            rf_F1 = {}
            skl_F1 = {}
            
            print("== isolated ==\n")
            from sklearn.metrics import f1_score, confusion_matrix
            print("classifier.y_test_isolated_flows =", classifier.y_test_isolated_flows.shape)
            print("rf_y_test_flows_isolated_predicted =", rf_y_test_flows_isolated_predicted)
            for i in EncryptedTrafficClassifierIterator(classifier.flow_ids):
                output = ("== %s ==\n" % str(i))
                cm_dict = confusion_matrix(classifier.y_test_isolated_flows, rf_y_test_flows_isolated_predicted[i])
                output += str(cm_dict) + '\n'
                cm_dict_normalized = confusion_matrix(classifier.y_test_isolated_flows, rf_y_test_flows_isolated_predicted[i], normalize = 'true')
                output += str(cm_dict_normalized) + '\n'
                print(output)

                output = ""
                
                cm = cm_dict
                FP = cm.sum(axis=0) - np.diag(cm)  
                FN = cm.sum(axis=1) - np.diag(cm)
                TP = np.diag(cm)
                TN = cm.sum() - (FP + FN + TP)
                rf_F1[i] = 2 * (TP) / (2 * TP + FP + FN) * 100
                output += ("FP = %s\n" % str(FP))
                output += ("FN = %s\n" % str(FN))
                output += ("TP = %s\n" % str(TP))
                output += ("TN = %s\n" % str(TN))
                if len(classifier.y_test_isolated_flows) > 0:
                    skl_F1 = f1_score(classifier.y_test_isolated_flows, rf_y_test_flows_isolated_predicted[i], average = 'micro')
                    output += ("skl_F1 = %s\n" % str(skl_F1))
                output += "\n"
                for j in range(len(classifier.classes)):
                    t = classifier.classes[j]
                    try:
                        output += ("for type %s \t\t F1 = %.2f\n" % (t, rf_F1[j]))
                    except IndexError as e:
                        pass
                    except KeyError as e:
                        pass
                output += "\n"
                print(output)
                
            output =""
            f1 = {}
            for i in EncryptedTrafficClassifierIterator(classifier.flow_ids):
                pkt, _ = i
                f1[pkt] = [0 for _ in range(len(classifier.classes))]
            for i in EncryptedTrafficClassifierIterator(classifier.flow_ids):
                pkt, _ = i
                for j in range(len(classifier.classes)):
                    try:
                        f1[pkt][j] += rf_F1[i][j]
                    except KeyError as e:
                        continue
                    except IndexError as e:
                        continue
        
            avg_scores = {}
            output = ""
            for pkt in classifier.nb_packets_per_flow:
                output += f"for {pkt} packets\n"
                for j in range(len(classifier.classes)):
                    t = classifier.classes[j]
                    avg_scores[(pkt, t)] = f1[pkt][j] / classifier.nb_folds
                    output += "average for type %s [%d] \t\t F1 = %.2f\n" % (t, j, avg_scores[(pkt, t)])
            output += "\n"
            print(output)
            ####

        if XG_ENABLED:
            print("==== XGBoost =====")
            xg_regr, xg_y_train_predicted, xg_y_test_flows_predicted = classifier.XGBoost_predict(
                classifier.X_train_flows_fitted,
                classifier.y_train_flows,
                classifier.X_test_flows_fitted,
                classifier.y_test_flows
            )
            
            # feats_flows, classification_results)
            # xg_cm_dict = classifier.confusion_matrix(xg_regr, xg_y_test_predicted, False)
            xg_cm_dict_flows, output = classifier.confusion_matrix(xg_regr,
                                                                   classifier.y_test_flows,
                                                                   xg_y_test_flows_predicted,
                                                                   classifier.flow_ids,
                                                                   "xg"
                                                                )
            print(output)
            
            xg_f1_scores_flows, output = classifier.get_F1_score(
                xg_cm_dict_flows,
                classifier.y_test_flows,
                xg_y_test_flows_predicted,
                classifier.flow_ids,
                "xg_flows")
            print(output)
            avg_scores, output = classifier.avg_f1_scores(xg_f1_scores_flows, classifier.flow_ids)
            print(output)
    # classification based on Packets
    if TEST_PACKETS:
        classifier.X_train_packets_fitted, classifier.X_test_packets_fitted = classifier.preprocessing(
            classifier.X_train_packets,
            classifier.y_train_packets,
            classifier.X_test_packets,
            classifier.y_test_packets,
            classifier.packet_ids,
            feats_packets
        )
        if RF_ENABLED:
            rf_regr_packets, rf_y_train_packets_predicted, rf_y_test_packets_predicted = classifier.RF_predict(
                classifier.X_train_packets_fitted,
                classifier.y_train_packets,
                classifier.X_test_packets_fitted,
                classifier.y_test_packets,
                classifier.packet_ids,
            )
            for i in EncryptedTrafficClassifierIterator(classifier.packet_ids):
                print("Feature ranking:")
                importances = rf_regr_flows[i].best_estimator_.named_steps["rf"].feature_importances_
                std = np.std([tree.feature_importances_ for tree in rf_regr_flows[i].best_estimator_.named_steps["rf"].estimators_],
                             axis=0)
                indices = np.argsort(importances)[::-1]       
                for f in range(classifier.X_train_flows[i].shape[1]):
                    print("%d. feature %s (%f)" % (f + 1, classifier.X_train_flows[i].columns[indices[f]], importances[indices[f]]))
            # __show_actual_and_predicted(X_test, y_test, rf_y_test_predicted, 1)
            # __analyze_CHAT(X_train, y_train, rf_y_train_predicted)
            # __analyze_CHAT(X_test, y_test, rf_y_test_predicted)
            rf_cm_dict, output = classifier.confusion_matrix(rf_regr_packets,
                                                             classifier.y_test_packets,
                                                             rf_y_test_packets_predicted,
                                                             classifier.packet_ids)
            print(output)
            rf_f1_scores, output = classifier.get_F1_score(rf_cm_dict,
                                                           classifier.y_test_packets,
                                                           rf_y_test_packets_predicted,
                                                           classifier.packet_ids,
                                                           "rf_packets")
            print(output)
            classifier.avg_f1_scores(rf_f1_scores, classifier.packet_ids)
    else:
        print("CLASSIFICATION BASED ON PACKETS NOT ENABLED")

    print(classifier.classification_results)
    if RF_ENABLED or GB_ENABLED or XG_ENABLED:
        classifier.save_results()