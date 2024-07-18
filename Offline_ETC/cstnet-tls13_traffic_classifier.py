#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from os import listdir
from os.path import isfile, join
import sys
import time

import pandas as pd

import numpy as np

from scipy.stats import kurtosis, skew

from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay

import seaborn as sns
import matplotlib.pyplot as plt 

from encrypted_traffic_classification import EncryptedTrafficClassifier, EncryptedTrafficClassifierIterator

########################################
# Data preparation: convert RAW data
########################################
class CstNetTls13Classifier(EncryptedTrafficClassifier):
    def __init__(self, nb_folds, nb_packets_per_flow):        
        super().__init__(
            nb_folds= nb_folds,
            nb_packets_per_flow = nb_packets_per_flow,
            filename_prefix = "cstnet_tls13",
            processed_data_output_dir = "cstnet_tls13_output/",
            data_dir = "data/cstnet_tls13/"            
        )
        
        pools = [tuple(pool) for pool in [self.nb_packets_per_flow, range(self.nb_folds)]]
        result = [[]]
        for pool in pools:
            result = [x+[y] for x in result for y in pool]
        self.flow_ids = result

        pools = [tuple(pool) for pool in [self.nb_packets_per_flow]]
        result = [[]]
        for pool in pools:
            result = [x+[y] for x in result for y in pool]
        self.flow_ids_without_folds = result

        pools = [tuple(pool) for pool in [range(self.nb_folds)]]
        result = [[]]
        for pool in pools:
            result = [x+[y] for x in result for y in pool]
        self.packet_ids = result
        
    ########################################
    # Preprocessing
    ########################################
    def _get_flows_with_all_packets(self):
        print("_get_flows_with_all_packets")
        start_time = time.time()
        subdirs = sorted([f for f in listdir(self.data_dir)])
        nb_flows = 0
        df_flows = pd.DataFrame()
        self.classes = set()
        for subdir in subdirs:
            # print("subdir", self.data_dir+subdir)
            _files = sorted([f for f in listdir(self.data_dir + subdir) if isfile(join(self.data_dir + subdir, f))])
            # print("  files", _files)
            for _i in range(len(_files)):
                f = self.data_dir + subdir + "/" + _files[_i]

                df_new = pd.read_csv(f, 
                                     names = [
                                         'flow_id',
                                         'timestamp', 
                                         'iat',                                                         
                                         'source',
                                         'sport',
                                         'dest', 
                                         'dport',
                                         'protocol', 
                                         'length'
                                     ],
                                     header = 0
                                     )   
                print(f, df_new.shape)
            
                # drop DNS traffic
                df_new = df_new.drop(df_new[df_new['sport'] == 53].index)
                df_new = df_new.drop(df_new[df_new['dport'] == 53].index)
                
                found = False
                for _c in self.all_classes:
                    if _c in f:
                        found = True
                        df_new['class'] = _c
                        self.classes.add(_c)
                        break
                if found == False:
                    print("class not identified for", f)
            
                # extract flow and add statistical features
                dfs = []
                for flow_id in df_new['flow_id'].unique():                    
                    nb_flows += 1
                    d = df_new[df_new['flow_id'] == flow_id].head(n = 1)
                    d['nb_packets'] = len(df_new[df_new['flow_id'] == flow_id])
                    c = d['class'].tolist()
                    dport = d.dport.tolist()
                    sport = d.sport.tolist()
                    #print(d)
                    _df = df_new.loc[df_new['flow_id'] == flow_id, 'iat']
                    d['sum_iat'] = np.sum(_df)
                
                    _df = df_new.loc[df_new['flow_id'] == flow_id, 'length']
                    d['sum_length'] = np.sum(_df)
                    d['src'] = f
                    dfs.append(d)
            _d = pd.concat(dfs)
            df_flows = pd.concat([_d, df_flows])
            # For debugging
            # break
                
        print("  processing took ", time.time() - start_time, "seconds.")
        print("%d flows processed" % nb_flows)            
        # Finish processing the data, create the train/tests split and save as pickle files
        df_flows = df_flows.fillna(0)
        
        self.classes = list(self.classes)
        self._hotencode_class(df_flows)
        return df_flows
        
    def data_preparation(self):
        print("data_preparation")
        import warnings
        warnings.filterwarnings("ignore")

        df_flows = {}
        files = []
        subdirs = [f for f in listdir(self.data_dir)]
        for subdir in subdirs:
            # print("subdir", self.data_dir+subdir)
            _files = [f for f in listdir(self.data_dir + subdir) if isfile(join(self.data_dir + subdir, f))]
            # print("  files", _files)
            for _i in range(len(_files)):
                _files[_i] = self.data_dir + subdir + "/" + _files[_i]
            files += _files

        # print(files)
        for i in self.nb_packets_per_flow:
            self.__generate_pickle_for_n_packets(i, files)

    def __generate_pickle_for_n_packets(self, n, files):
        print("__generate_pickle_for_n_packets n =", n)
        nb_flows = 0
        df_flows = pd.DataFrame()
        # dfs = []
        self.classes = set()
        for f in files:
            # print("f=", f)
            df_new = pd.read_csv(f, 
                                 names = [
                                     'flow_id',
                                     'timestamp', 
                                     'iat',                                                         
                                     'source',
                                     'sport',
                                     'dest', 
                                     'dport',
                                     'protocol', 
                                     'length'
                                 ],
                                 header = 0
                                 )   
            print(n, f, df_new.shape)
            
            # drop DNS traffic
            df_new = df_new.drop(df_new[df_new['sport'] == 53].index)
            df_new = df_new.drop(df_new[df_new['dport'] == 53].index)
            
            found = False
            for _c in self.all_classes:
                if _c in f:
                    found = True
                    df_new['class'] = _c
                    self.classes.add(_c)
                    break
            if found == False:
                print("class not identified for", f)
            
            # extract flow and add statistical features
            for flow_id in df_new['flow_id'].unique():                    
                nb_flows += 1
                _df_new = df_new[df_new['flow_id'] == flow_id].head(n = n)
                d = _df_new.head(n = 1)
                d['nb_packets'] = len(_df_new) #df_new[df_new['flow_id'] == flow_id])
                c = d['class'].tolist()
                dport = d.dport.tolist()
                sport = d.sport.tolist()
                #print(d)
                _df = _df_new['iat']
                d['min_iat'] = np.min(df_new[df_new['iat'] > 0]['iat']) # probably useless as most probably always 0 for the first packet
                d['max_iat'] = np.max(_df)
                d['sum_iat'] = np.sum(_df)
                d['mean_iat'] = np.mean(_df)
                d['median_iat'] = np.median(_df)
                d['std_iat'] = np.std(_df)
                d['1stQ_iat'] = np.quantile(_df, 0.25)
                d['3rdQ_iat'] = np.quantile(_df, 0.75)
                _a = list(_df)
                d['skew_iat'] = skew(_a)
                d['kurt_iat'] = kurtosis(_a)
                
                _df = _df_new['length']
                d['min_length'] = np.min(_df)
                d['max_length'] = np.max(_df)
                d['sum_length'] = np.sum(_df)
                d['median_length'] = np.median(_df)
                d['mean_length'] = np.mean(_df)
                d['std_length'] = np.std(_df)
                d['1stQ_length'] = np.quantile(_df, 0.25)
                d['3rdQ_length'] = np.quantile(_df, 0.75)
                _a = list(_df)
                d['skew_length'] = skew(_a)
                # d['skew_length'] = skew(np.array(df_new.loc[df_new['flow_id'] == flow_id, 'length']))
                d['kurt_length'] = kurtosis(_a)
                d['src'] = f
                # dfs.append(d)
                df_flows = pd.concat([d, df_flows])
            # if nb_flows > 20:
            #     break
                
        print("%d flows processed" % nb_flows)            
        # Finish processing the data, create the train/tests split and save as pickle files
        df_flows = df_flows.fillna(0)
        
        self.classes = list(self.classes)
        self._hotencode_class(df_flows)
        
        filename = self.filename_prefix + "_" + str(n) + ".pickle"
        # filename = "cstnet_tls13_" + str(n) + ".pickle"
        self._generate_data_folds(df_flows, filename)
        
    ########################################
    # Data Analysis
    ########################################
    """
    def __show_actual_and_predicted(self, X, y, y_pred, _class):
        print(self.classes)
        for _i in itertools.product(NB_PACKETS, self.filenames):
            i = (_i[0], _i[1], 0)
            print(i)
            df = X[i].copy()
            df['type'] = y[i]
            df['type_pred'] = y_pred[i]
            print(df.columns)
            a4_dims = (23.4, 16.54)
            fig, ax = plt.subplots(figsize = a4_dims)
            sns.lmplot(
                x = 'sum_iat', 
                y = 'sum_length', 
                data = df[df['type'] == _class],
                hue = 'type', 
                fit_reg = False,
                height = 4, aspect = 5,
                # color = 'green',
                # scatter_kws = {'alpha': 0.3},
                # ax = ax,
                legend = False,
                palette = 'viridis'
            )
            #ax.set(xlabel='time_delta', ylabel='packet_size')
            ax.set(xlabel = 'duration', ylabel = 'sum_packet_size')
            plt.legend(title = 'Class', labels =self.classes)
            plt.savefig("cstnet_tls13_" + self.classes[_class] + "_"+ str(i[0]) + "_" + i[1]+".png", format = 'png')    
            fig, ax2 = plt.subplots(figsize = a4_dims)
            sns.lmplot(
                x = 'sum_iat', 
                y = 'sum_length', 
                data = df[df['type_pred'] == _class],
                hue = 'type', 
                fit_reg = False,
                height = 4, aspect = 5,
                # color = 'orange',
                # scatter_kws = {'alpha': 0.3},
                legend = False,
                palette = 'viridis',
                # ax = ax2
            )
            ax2.set(xlabel = 'duration', ylabel = 'sum_packet_size')
            plt.legend(title = 'Class', labels =self.classes)
            plt.savefig(self.filename_prefix + "_" + self.classes[_class] + "_pred_"+ str(i[0]) + "_" + i[1]+".png", format = 'png')
    """
########################################
# Entry point
########################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='cstnet_tls13_instant_messaging_traffic_classifier',
        description='Classify packets or flows from CTSNET TTLS1.3 dataset',
        epilog=''
    )
    parser.add_argument('-p', '--nb_packets', action = 'append', type = int, required = True) #, default = [4, 8])
    parser.add_argument('-c', '--classifier', action = 'append', type = str) #, default = ['rf'])
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
        c = c.lower()
        if c == "rf":
            RF_ENABLED = True
        elif c == "gb":
            GB_ENABLED = True
        elif c == "xg":
            XG_ENABLED = True
        else:
            print("Unknown classifier", c)

    classifier = CstNetTls13Classifier(
        nb_folds = args.nb_folds,
        nb_packets_per_flow = args.nb_packets
    )

    if args.force_rf_classification == True:
        classifier.force_rf_classification = True

    classifier.all_classes = [
        "163.com",
        "chia.net",
        "github.com",
        "leetcode-cn.com",
        "qcloud.com",
        "toutiao.com",
        "51cto.com",
        "chinatax.gov.cn",
        "gitlab.com",
        "media.net",
        "qq.com",
        "twimg.com",
        "51.la",
        "cisco.com",
        "gmail.com",
        "mi.com",
        "researchgate.net",
        "twitter.com",
        "acm.org",
        "cloudflare.com",
        "goat.com",
        "microsoft.com",
        "runoob.com",
        "unity3d.com",
        "adobe.com",
        "cloudfront.net",
        "google.com",
        "mozilla.org",
        "sciencedirect.com",
        "v2ex.com",
        "alibaba.com",
        "cnblogs.com",
        "grammarly.com",
        "msn.com",
        "semanticscholar.org",
        "vivo.com.cn",
        "alicdn.com",
        "codepen.io",
        "gravatar.com",
        "naver.com",
        "sina.com.cn",
        "vk.com",
        "alipay.com",
        "crazyegg.com",
        "guancha.cn",
        "netflix.com",
        "smzdm.com",
        "vmware.com",
        "amap.com",
        "criteo.com",
        "huanqiu.com",
        "nike.com",
        "snapchat.com",
        "walmart.com",
        "amazonaws.com",
        "ctrip.com",
        "huawei.com",
        "notion.so",
        "sohu.com",
        "weibo.com",
        "ampproject.org",
        "dailymotion.com",
        "hubspot.com",
        "nvidia.com",
        "springer.com",
        "wikimedia.org",
        "apple.com",
        "deepl.com",
        "huya.com",
        "office.net",
        "spring.io",
        "wikipedia.org",
        "arxiv.org",
        "digitaloceanspaces.com",
        "ibm.com",
        "onlinedown.net",
        "squarespace.com",
        "wp.com",
        "asus.com",
        "duckduckgo.com",
        "icloud.com",
        "opera.com",
        "statcounter.com",
        "xiaomi.com",
        "atlassian.net",
        "eastday.com",
        "ieee.org",
        "oracle.com",
        "steampowered.com",
        "ximalaya.com",
        "azureedge.net",
        "eastmoney.com",
        "instagram.com",
        "outbrain.com",
        "taboola.com",
        "yahoo.com",
        "baidu.com",
        "elsevier.com",
        "iqiyi.com",
        "overleaf.com",
        "t.co",
        "yandex.ru",
        "bilibili.com",
        "facebook.com",
        "jb51.net",
        "paypal.com",
        "teads.tv",
        "youtube.com",
        "biligame.com",
        "feishu.cn",
        "jd.com",
        "pinduoduo.com",
        "thepaper.cn",
        "yy.com",
        "booking.com",
        "ggpht.com",
        "kugou.com",
        "python.org",
        "tiktok.com",
        "zhihu.com"
    ]
    
    non_needed_features = [
        'flow_id', 
        'class', 
        'source', 
        'dest', 
        'sport',
        'dport', 
        'protocol', 
        'timestamp', 
        # 'nb_packets',
        'src',
        'iat',
        'direction',
        'length'
    ]

    all_features_flows = [
        'min_iat',
        'max_iat',
        'sum_iat',
        'mean_iat',
        'median_iat',
        'std_iat',
        '1stQ_iat',
        '3rdQ_iat', 
        'skew_iat',
        'kurt_iat',
        'min_length',
        'max_length',
        'sum_length',
        'median_length',
        'mean_length', 
        'std_length',
        '1stQ_length',
        '3rdQ_length',
        'skew_length',
        'kurt_length',
        'nb_packets',
        # 'sport',
        # 'dport',
        # 'protocol',
        # 'direction'
    ]
    # best_features = [
    #     'max_iat',
    #     'sum_iat',
    #     'mean_iat',
    #     'median_iat',
    #     'std_iat',
    #     '1stQ_iat',
    #     '3rdQ_iat', 
    #     'skew_iat',
    #     'kurt_iat',
    #     'min_length',
    #     'max_length',
    #     'sum_length',
    #     'median_length',
    #     'mean_length', 
    #     'std_length',
    #     '1stQ_length',
    #     '3rdQ_length',
    #     'skew_length',
    #     'kurt_length'
    # ]
    best_features = ['3rdQ_iat', 'std_iat', 'std_length', 'skew_iat', 'max_iat', 'sum_iat', 'mean_length', '1stQ_length', 'max_length', 'mean_iat', 'min_length', 'sum_length', 'median_length', '1stQ_iat', 'median_iat', '3rdQ_length', 'kurt_iat', 'kurt_length', 'nb_packets']
    online_features=[
        'sum_iat',
        'sum_length',
        'max_length',
        'mean_iat',
        'max_iat',
        'mean_length',
        'min_length',
        'min_iat'
    ]
    feats_flows = all_features_flows
    
    # Preprocessing
    if not classifier.data_prepared():
        classifier.data_preparation()
        classifier.load_flows()
    else:
        classifier.load_flows()
        classifier.classes = classifier.all_classes
    # if not classifier.data_prepared():
    #     classifier.data_preparation()
    # else:
    #     classifier.classes = classifier.all_classes
        
    # classifier.load_flows()
    classifier.cleanup_data(classifier.X_train_flows,
                            classifier.y_train_flows,
                            classifier.X_test_flows,
                            classifier.y_test_flows,
                            classifier.flow_ids,
                            non_needed_features)
    # classifier._cleanup_data(non_needed_features)
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
    # __correlation()
    # feats = all_features
    # analyze_models_for_npkts(10, all_features, "all_feats")

    if args.report == True:
        classifier._viz(distribution = 0, class_distribution = -1, nb_packets = -1, min_iat = -1, max_iat = -1)
        for n in classifier.nb_packets_per_flow:
            if n == 4:
                classifier._viz(distribution = -1, class_distribution = 0, nb_packets = 0, min_iat = -1, max_iat = -1)
            elif n == 8:
                classifier._viz(distribution = -1, class_distribution = 10, nb_packets = 0, min_iat = -1, max_iat = -1)
            elif n == 600000:
                classifier._viz(distribution = -1, class_distribution = 11, nb_packets = 0, min_iat = 1, max_iat = -1)
        sys.exit(1)
    if VISUALIZATION_ENABLED:
        # f = classifier.filename_prefix + '_datasetflows_distribution.pickle'
        # if isfile(classifier.processed_data_output_dir + f):
        #     print("Loading dataset from pickle file", f)
        #     _df = classifier._load_pickle(f)
        # else:
        #     print("Creating dataset")
        #     _df = classifier._get_flows_with_all_packets()
        #     classifier._pickle_dump(_df, f)
        #     print("Dataset saved in file", f)
        # classifier._class_distribution(_df, classifier.filename_prefix + '_flows_distribution')
        # # classifier._nb_packets_distribution(_df, classifier.filename_prefix + "_flows_nb_packets_distribution" )
        # classifier._distribution(_df, classifier.filename_prefix + "_flows_class_split" )
        pkt = classifier.nb_packets_per_flow[0]
        fold = 0
        _i = pkt, fold
        _df1 = classifier.X_train_flows[_i].copy()
        # print("_df1", _df1.columns)
        # print("y_train", classifier.y_train_flows[_i])
        # print(classifier.y_train_flows[_i][classifier.y_train_flows[_i].index.duplicated()])
        _df1['type'] = classifier.y_train_flows[_i].values
        # print("_df1 type", _df1.columns)
        _df2 = classifier.X_test_flows[_i].copy()
        # print("_df2", _df1.columns)
        _df2['type'] = classifier.y_test_flows[_i].values
        _df = pd.concat([_df1, _df2])
        _df.reset_index()
        print(_df.shape)
        print(_df['type'].value_counts().to_string())
        # classifier._distribution(_df, classifier.filename_prefix + "_flows_class_split_" + str(pkt) + '_pkt', xticks = False)
        # classifier._class_distribution(_df, classifier.filename_prefix + '_flows_distribution_' + str(pkt) + '_pkt', xticks = False)
        classifier._nb_packets_distribution(_df, classifier.filename_prefix + "_flows_nb_packets_distribution_" + str(pkt) + '_pkt', xticks = False)
        # classifier._min_iat_distribution(_df, classifier.filename_prefix + "_flows_min_iat_distribution_" + str(pkt) + '_pkt', xticks = False)
        
    if RF_ENABLED:
        print("==== RandomForest =====")
        """
        classifier.X_train_flows_fitted, classifier.X_test_flows_fitted = classifier.preprocessing(
            classifier.X_train_flows,
            classifier.y_train_flows,
            classifier.X_test_flows,
            classifier.y_test_flows,
            classifier.flow_ids,
            feats_flows
        )
        classifier.X_train_flows_fitted = classifier.X_train_flows
        classifier.X_test_flows_fitted = classifier.X_test_flows
        """
        rf_regr_flows, rf_y_train_flows_predicted, rf_y_test_flows_predicted = classifier.RF_predict(
                classifier.X_train_flows_fitted,
                classifier.y_train_flows,
                classifier.X_test_flows_fitted,
                classifier.y_test_flows,
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

    if GB_ENABLED:
        gb_regr, gb_y_train_predicted, gb_y_test_predicted  = classifier.GBoost_predict(feats_flows, df_score)
        gb_cm_dict = classifier.confusion_matrix(gb_regr, gb_y_test_predicted, False)
        gb_f1_scores = classifier.get_F1_score(df_score, gb_cm_dict,  y_test, gb_y_test_predicted, "gb", False)
        classifier.avg_f1_scores(gb_f1_scores_flows, classifier.flow_ids_without_folds)
        classifier.avg_f1_scores(gb_f1_scores)

    if XG_ENABLED:
        print("==== XGBoost =====")
        xg_regr, xg_y_train_predicted, xg_y_test_flows_predicted = classifier.XGBoost_predict(
            classifier.X_train_flows_fitted,
            classifier.y_train_flows,
            classifier.X_test_flows_fitted,
            classifier.y_test_flows
        )
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

    print(classifier.classification_results)
    if RF_ENABLED or GB_ENABLED or XG_ENABLED:
        classifier.save_results()