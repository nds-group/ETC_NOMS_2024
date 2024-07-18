#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import itertools
from os import listdir
from os.path import isfile, join
import time
import sys

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

filename_patterns = { 
    "_aim_chat": "CHAT", 
    "_bittorrent": "P2P", 
    "_email": "MAIL",
    "_facebook_audio" : "VOIP",
    "_facebook_chat": "CHAT",
    "_ftps_": "FT",
    "_sftp_": "FT",
    "_hangouts_audio": "VOIP",
    "_hangouts_chat" : "CHAT",
    "_icq_chat" : "CHAT",
    "_netflix_" : "STREAMING",
    "_skype_audio" : "VOIP",
    "_skype_chat": "CHAT",
    "_skype_files" : "FT",
    "_spotify_" : "STREAMING",
    "_vimeo_" : "STREAMING",
    "_youtube_" : "STREAMING"
}

########################################
# Data preparation: convert RAW data
########################################
class ISCXVPN2016Classifier(EncryptedTrafficClassifier):
    def __init__(self, nb_folds, nb_packets_per_flow):
        super().__init__(
            nb_folds= nb_folds,
            nb_packets_per_flow = nb_packets_per_flow,
            filename_prefix = "iscxvpn2016",
            # processed_data_output_dir = "iscxvpn2016_output_addendum/",
            processed_data_output_dir = "iscxvpn2016_output/",
            data_dir = "data/ISCXVPN2016-20230713/"            
        )
        
        pools = [tuple(pool) for pool in [self.nb_packets_per_flow, range(self.nb_folds)]]#, self.filenames, 
        result = [[]]
        for pool in pools:
            result = [x+[y] for x in result for y in pool]
        self.flow_ids = result

        pools = [tuple(pool) for pool in [self.nb_packets_per_flow]] #, self.filenames]]
        result = [[]]
        for pool in pools:
            result = [x+[y] for x in result for y in pool]
        self.flow_ids_without_folds = result

        # pools = [tuple(pool) for pool in [self.filenames, range(self.nb_folds)]]
        pools = [tuple(pool) for pool in [range(self.nb_folds)]]
        result = [[]]
        for pool in pools:
            result = [x+[y] for x in result for y in pool]
        self.packet_ids = result

    def _get_flows_with_all_packets(self):
        print("_get_flows_with_all_packets")

        self.classes = set()
        nb_flows = 0
        df_flows = pd.DataFrame()
        start_time = time.time()
        dfs = []
        
        files = [f for f in listdir(self.data_dir) if isfile(join(self.data_dir, f))]
        for _i in range(len(files)):
            f = self.data_dir + files[_i]
            if 'voipbuster' in f:            
                continue
            else:
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
            
                found = False
                for k, v in filename_patterns.items():
                    if k in f:
                        df_new['class'] = v
                        self.classes.add(v)
                        found = True
                        break
                if found == False:
                    print("Type for file", f, "not found")
                    sys.exit(1)
                    
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
                    if dport[0] in [80, 443] or sport[0] in [80, 443]:                    
                        if c[0] != 'STREAMING': #'netflix' not in f:
                            d['class'] = 'BROWSING'                        
                            self.classes.add('BROWSING')
                    dfs.append(d)

            _d = pd.concat(dfs)
            df_flows = pd.concat([_d, df_flows])
            dfs = []
        df_flows = df_flows.fillna(0)
        print("  processing took ", time.time() - start_time, "seconds.")
        self.classes = list(self.classes)
        self._hotencode_class(df_flows)
        return df_flows
        
    def data_preparation(self):
        print("data_preparation")
        import warnings
        warnings.filterwarnings("ignore")

        # df_flows_dict = {}
                
        files = [f for f in listdir(self.data_dir) if isfile(join(self.data_dir, f))]
        for _i in range(len(files)):
            files[_i] = self.data_dir + files[_i]

        # TODO
        # PROCESSED_PATH = self.data_dir + "/processed/"
        # processed_files = [f for f in listdir(PROCESSED_PATH) if isfile(join(PROCESSED_PATH, f))]    
        # for _i in range(len(processed_files)):
        #     files.append(PROCESSED_PATH + processed_files[_i])

        for i in self.nb_packets_per_flow:
            self.__generate_pickle_for_n_packets(i, files)

    def __generate_pickle_for_n_packets(self, n, files):
        print("__generate_pickle_for_n_packets n =", n)
        nb_flows = 0
        df_flows = pd.DataFrame()
        # PROCESSED_PATH = "data/ISCXVPN2016-20230713/processed/"
        self.classes = set()
        
        for f in files:
            # print(f)
            start_time = time.time()
            if 'voipbuster' in f:            
                continue
            else:
            # elif str(n) + "_" in f or (n == 600000 and PROCESSED_PATH not in f): 
                #elif 'spotify' in f and (str(i) + "_" in f or (n == 600000 and PROCESSED_PATH not in f)): 
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
                for k, v in filename_patterns.items():
                    if k in f:
                        df_new['class'] = v
                        self.classes.add(v)
                        found = True
                        break
                if found == False:
                    print("Type for file", f, "not found")
                    sys.exit(1)
            
                # extract flow and add statistical features
                for flow_id in df_new['flow_id'].unique():                    
                    nb_flows += 1
                    _df_new = df_new[df_new['flow_id'] == flow_id].head(n = n)
                    d = _df_new.head(n = 1)
                    d['nb_packets'] = len(_df_new) #[df_new['flow_id'] == flow_id])
                    c = d['class'].tolist()
                    dport = d.dport.tolist()
                    sport = d.sport.tolist()
                    #print(d)
                    _df = _df_new.loc[_df_new['flow_id'] == flow_id, 'iat']
                    
                    d['min_iat'] = np.min(df_new[df_new['iat'] > 0]['iat'])
                    # previous code was just using np.min which was always returning 0 as iat of first packet of flow is 0
                    # code kept commented here to allow comparison with previous results
                    # d['min_iat'] = np.min(_df) # probably useless as most probably always 0 for the first packet
                    
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
                    
                    _df = _df_new.loc[_df_new['flow_id'] == flow_id, 'length']
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
                    # There is no file with BROWSING content: consider all traffic on port 80 or 443 to be BROWSING
                    # d_netflix_as_browsing = d
                    # d_no_browsing = d.copy()
                    if dport[0] in [80, 443] or sport[0] in [80, 443]:                    
                        if c[0] != 'STREAMING': #'netflix' not in f:
                            d['class'] = 'BROWSING'                        
                            self.classes.add('BROWSING')
                        # else:
                        #     d_netflix_as_browsing = d.copy()
                        #     d_netflix_as_browsing['class'] = "BROWSING"  
                                
                    # df_flows_netflix_as_browsing = pd.concat([d_netflix_as_browsing, df_flows_netflix_as_browsing])
                    df_flows = pd.concat([d, df_flows])                            
                    # df_flows_no_browsing = pd.concat([d_no_browsing, df_flows_no_browsing])
                
            # print("%d flows processed" % nb_flows)            
            print("  %d flows processed in " % (nb_flows), time.time() - start_time, "seconds.")            
        # Finish processing the data, create the train/tests split and save as pickle files
        df_flows = df_flows.fillna(0)
        # df_flows_netflix_as_browsing = df_flows_netflix_as_browsing.fillna(0)
        # df_flows_no_browsing = df_flows_no_browsing.fillna(0)        
        
        # self.__hotencode_class(df_flows_netflix_as_browsing)
        self.classes = list(self.classes)
        self._hotencode_class(df_flows)
        # self.__hotencode_class(df_flows_no_browsing)
        
        # seed = 42
        # # first dataset
        # filename = "iscxvpn2016_netflix_as_browsing_" + str(n) + ".pickle"
        # # df_flows_netflix_as_browsing.reset_index(inplace = True)
        # self._generate_data_folds(df_flows_netflix_as_browsing, filename)
        
        # second dataset
        # filename = "iscxvpn2016_netflix_as_streaming_" + str(n) + ".pickle"
        
        #self._nb_packets_distribution(df_flows, self.filename_prefix + "_flows_nb_packets_distribution" )
        
        filename = self.filename_prefix + "_" + str(n) + ".pickle"
        # df_flows_netflix_as_streaming.reset_index(inplace = True)
        self._generate_data_folds(df_flows, filename)
        
        # # third dataset
        # filename = "iscxvpn2016_no_browsing_" + str(n) + ".pickle"
        # # df_flows_no_browsing.reset_index(inplace = True)
        # self._generate_data_folds(df_flows_no_browsing, filename)
        
        
    ########################################
    # Data Analysis
    ########################################
    def __analyze_CHAT(self, X, y, y_pred):
        i = (NB_PACKETS[-1], self.filenames[0], 0)
        """
        df = X[i].copy()
        df['type'] = y[i]
        df_chat = df[df['type'] == 1]
        a4_dims = (11.7, 8.27)
        fig, ax = plt.subplots(figsize = a4_dims)
        df_chat = df_chat.reset_index()
        sns.scatterplot(y = 'sum_iat', x = 'index', data = df_chat, ax = ax)
        ax.set(xlim = (0, 50_000))
        plt.savefig("iscxvpn_CHAT_sum_iat_distribution.png", format="png")
        
        df = X[i].copy()
        df['type'] = y_pred[i]
        df_chat = df[df['type'] == 1]
        df_chat = df_chat.reset_index()
        fig2, ax2 = plt.subplots(figsize = a4_dims)
        sns.scatterplot(y = 'sum_iat', x = 'index', data = df_chat, ax = ax2)
        ax2.set(xlim = (0, 50_000))
        plt.savefig("iscxvpn_CHAT_predicted_sum_iat_distribution.png", format="png")
    
        a4_dims = (11.7, 8.27)
        fig, ax3 = plt.subplots(figsize = a4_dims)
        df_chat = df_chat.reset_index()
        sns.scatterplot(y = 'sum_length', x = 'index', data = df_chat, ax = ax3)
        ax3.set(xlim = (0, 50_000))
        plt.savefig("iscxvpn_CHAT_sum_length_distribution.png", format="png")

        df = X[i].copy()
        df['type'] = y_pred[i]
        df_chat = df[df['type'] == 1]
        df_chat = df_chat.reset_index()
        fig2, ax4 = plt.subplots(figsize = a4_dims)
        sns.scatterplot(y = 'sum_length', x = 'index', data = df_chat, ax = ax4)
        ax4.set(xlim = (0, 50_000))
        plt.savefig("iscxvpn_CHAT_predicted_sum_length_distribution.png", format="png")
        """
        df = X[i].copy()
        df['type_real'] = y[i]
        df['type_pred'] = y_pred[i]
        df_chat = df[df['type_real'] == 1]
        df_pred_correct = df_chat[df_chat['type_real'] == df_chat['type_pred']]
        df_pred_notcorrect = df_chat[df_chat['type_real'] != df_chat['type_pred']]
        print("df_chat.shape =", df_chat.shape)
        print("df_pred_correct.shape =", df_pred_correct.shape)
        print("df_pred_notcorrect.shape =", df_pred_notcorrect.shape)
        print("== df_pred_correct ==")
        print(df_pred_correct.describe())
        print("== df_pred_notcorrect ==")
        print(df_pred_notcorrect.describe())
        print("== df_pred_correct ==")
        print(df_pred_correct)
        print("== df_pred_notcorrect ==")
        print(df_pred_notcorrect)

        
    # def __show_actual_and_predicted(self, X, y, y_pred, _class):
    #     print(self.classes)
    #     for _i in itertools.product(NB_PACKETS, self.filenames):
    #         i = (_i[0], _i[1], 0)
    #         print(i)
    #         df = X[i].copy()
    #         df['type'] = y[i]
    #         df['type_pred'] = y_pred[i]
    #         print(df.columns)
    #         a4_dims = (23.4, 16.54)
    #         fig, ax = plt.subplots(figsize = a4_dims)
    #         sns.lmplot(
    #             x = 'sum_iat', 
    #             y = 'sum_length', 
    #             data = df[df['type'] == _class],
    #             hue = 'type', 
    #             fit_reg = False,
    #             height = 4, aspect = 5,
    #             # color = 'green',
    #             # scatter_kws = {'alpha': 0.3},
    #             # ax = ax,
    #             legend = False,
    #             palette = 'viridis'
    #         )
    #         #ax.set(xlabel='time_delta', ylabel='packet_size')
    #         ax.set(xlabel = 'duration', ylabel = 'sum_packet_size')
    #         plt.legend(title = 'Class', labels =self.classes)
    #         plt.savefig("iscxvpn_" + self.classes[_class] + "_"+ str(i[0]) + "_" + i[1]+".png", format = 'png')    
    #         fig, ax2 = plt.subplots(figsize = a4_dims)
    #         sns.lmplot(
    #             x = 'sum_iat', 
    #             y = 'sum_length', 
    #             data = df[df['type_pred'] == _class],
    #             hue = 'type', 
    #             fit_reg = False,
    #             height = 4, aspect = 5,
    #             # color = 'orange',
    #             # scatter_kws = {'alpha': 0.3},
    #             legend = False,
    #             palette = 'viridis',
    #             # ax = ax2
    #         )
    #         ax2.set(xlabel = 'duration', ylabel = 'sum_packet_size')
    #         plt.legend(title = 'Class', labels =self.classes)
    #         plt.savefig("iscxvpn_" + self.classes[_class] + "_pred_"+ str(i[0]) + "_" + i[1]+".png", format = 'png')    

    ########################################
    # Akem's methods
    ########################################
    # Feature Importance
    """
    Function to Fit model based on optimal values of depth and number of estimators and use it
    to compute feature importance for all the features.
    """
    def get_feature_importance(depth, n_tree, max_leaf, X_train, y_train):
        from sklearn.ensemble import RandomForestClassifier
        
        # rf_opt = RandomForestClassifier(max_depth = depth, n_estimators = n_tree, random_state=42, bootstrap=False)
        rf_opt = RandomForestClassifier(max_depth = depth, n_estimators = n_tree, max_leaf_nodes=max_leaf, random_state=42, bootstrap=False)
        rf_opt.fit(X_train, y_train)
        feature_importance = pd.DataFrame(rf_opt.feature_importances_)
        feature_importance.index = X_train.columns
        feature_importance = feature_importance.sort_values(by=list(feature_importance.columns),axis=0,ascending=False)
        
        return feature_importance


    """
    Function to Fit model based on optimal values of depth and number of estimators and feature importance
    to find the fewest possible features to exceed the previously attained score with all selected features
    """
    def get_fewest_features(depth, n_tree, max_leaf, importance):    
        sorted_feature_names = importance.index
        # print('sorted_feature_names: ', sorted_feature_names)
        features = []
        for f in range(1,len(sorted_feature_names)+1):
            features.append(sorted_feature_names[0:f])
            # print('features:', features)
        return features


    def get_result_scores(classes, cl_report):
        precision=[]
        recall=[]
        f1_score=[]
        supports=[]
        for a_class in classes:
            precision.append(cl_report[a_class]['precision'])
            recall.append(cl_report[a_class]['recall'])
            f1_score.append(cl_report[a_class]['f1-score'])
            supports.append(cl_report[a_class]['support'])
        return precision, recall, f1_score, supports


    def get_scores(classes, depth, n_tree, feats, max_leaf, X_train, y_train, X_test, y_test):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(max_depth=depth, n_estimators = n_tree, max_leaf_nodes=max_leaf, n_jobs=4,
                                       random_state=42, bootstrap=False)                              

        model.fit(X_train[feats], y_train)
        y_pred = model.predict(X_test[feats])
        
        class_report = classification_report(y_test, y_pred, target_names=classes, output_dict = True)
        
        accurac = model.score(X_test[feats], y_test)
        macro_score = class_report['macro avg']['f1-score']
        weighted_score = class_report['weighted avg']['f1-score']

        return model, class_report, macro_score, weighted_score, y_pred, accurac

    def get_x_y(Dataset, classes, feats):
        Dataset = Dataset[Dataset["Label"].isin(classes)]    
        X = Dataset[feats]
        y = Dataset['Label'].replace(classes, range(len(classes)))
        #     y = Dataset.columns[-1].replace(classes, range(len(classes)))
        
        return X, y

    def analyze_models(classes, model_type, depths, n_trees, X_train, y_train, X_test, y_test, max_leaf, filename_out):
        
        with open(filename_out, "w") as res_file:
            print('depth;tree;n_feat;Macro_F1;Weighted_F1;Accuracy;feats;c_report', file=res_file)
            if model_type == 'RF':
                # FOR EACH (depth, n_tree, feat)
                for depth in depths:
                    for n_tree in n_trees:
                        # get feature orders to use
                        importance = get_feature_importance(depth, n_tree, max_leaf, X_train, y_train)

                        m_feats = get_fewest_features(depth, n_tree, max_leaf, importance) 
                        for feats in m_feats:
                            # Get the scores with the given (depth, n_tree, feat)
                            model, c_report, macro_f1, weight_f1, y_pred, accuracs = get_scores(classes, depth, n_tree, feats, max_leaf, X_train, y_train, X_test, y_test)
                            
                            print(str(depth)+';'+str(n_tree)+';'+str(len(feats))+';'+str(macro_f1)+';'+str(weight_f1)+';'+str(accuracs)+';'+str(list(feats))+';'+str(c_report), file=res_file)
        print("Analysis Complete. Check output file.")
        return []

    # N = number of packets in flows, feats = array of feature names to use, feat_name = string to add to output file name
    def analyze_models_for_npkts(self, N, feats, feat_name):
        #for i in itertools.product(NB_PACKETS, self.filenames, range(NB_FOLDS)):
        i = (N, self.filenames[0], 0)
        print("Number of packets per flow: ", N)
        
        X_trains, y_trains = X_train[i][feats], y_train[i]
        X_tests,  y_tests  = X_test[i][feats], y_test[i]
        
        results_file = "Models_" + feat_name + "_" + str(N) + "_pkts_.csv"
        analyze_models(self.classes, "RF", range(7, 20, 1), range(1, 8, 2), X_trains, y_trains, X_tests, y_tests, 500, results_file)
        
        results = pd.read_csv(results_file, sep=';')
        results = results.sort_values(by=['Weighted_F1','Macro_F1'],ascending=False)
        print(results.head(10))
        print("******")
        print(results.head(1)['c_report'].values)

    ########################################
    # GBoost
    ########################################
    def GBoost_predict(self, feats, df_score):
        print("GBoost_predict")
        from sklearn.ensemble import GradientBoostingClassifier
        gb_model = {}
        
        for i in EncryptedTrafficClassifierIterator(classifier.flow_ids): # for i in itertools.product(NB_PACKETS, self.filenames, range(NB_FOLDS)):
            gb_model[i] = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state = 42)

        for i in EncryptedTrafficClassifierIterator(classifier.flow_ids): # for i in itertools.product(NB_PACKETS, self.filenames, range(NB_FOLDS)):
            print("==",i,"==")
            try:
                gb_model[i].fit(X_train[i][feats], y_train[i])
            except ValueError as e:
                print(e)
                pass
        
        gb_y_train_predicted = {}
        gb_y_test_predicted = {}
        gb_train_score = {}
        gb_test_score = {}
        for i in EncryptedTrafficClassifierIterator(classifier.flow_ids): # for i in itertools.product(NB_PACKETS, self.filenames, range(NB_FOLDS)):
            print("==",i,"==")
            gb_y_train_predicted[i] = gb_model[i].predict(X_train[i][feats])
            gb_y_test_predicted[i] = gb_model[i].predict(X_test[i][feats])
            gb_train_score[i] = gb_model[i].score(X_train[i][feats], y_train[i])
            gb_test_score[i] = gb_model[i].score(X_test[i][feats], y_test[i])

        self._get_scores_from_models(gb_model, y_test, gb_y_test_predicted, feats)

        gb_cm_dict = {}
        for i in EncryptedTrafficClassifierIterator(classifier.flow_ids): # for i in itertools.product(NB_PACKETS, self.filenames, range(NB_FOLDS)):
            print("==",i,"==")
            gb_cm_dict[i] = confusion_matrix(y_test[i], gb_y_test_predicted[i].astype(int))
            print(gb_cm_dict[i])

        for i in EncryptedTrafficClassifierIterator(classifier.flow_ids): # for i in itertools.product(NB_PACKETS, self.filenames, range(NB_FOLDS)):
            n = i[0]
            df_score[filename].loc[df_score[filename]['nb_packets'] == n, 'gb_train_score'] = gb_train_score[i]
            df_score[filename].loc[df_score[filename]['nb_packets'] == n, 'gb_test_score'] = gb_test_score[i]
        
        return gb_model, gb_y_train_predicted, gb_y_test_predicted

########################################
# Entry point
########################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='iscxvpn2016-vpn-classifier',
        description='Classify packets or flows from ISCXVPN2016',
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
            
    classifier = ISCXVPN2016Classifier(
        nb_folds = args.nb_folds,
        nb_packets_per_flow = args.nb_packets
    )
    
    if args.force_rf_classification == True:
        classifier.force_rf_classification = True

    classifier.all_classes = {
        0: 'BROWSING',
        1: 'CHAT',
        2: 'FT',
        3: 'P2P',
        4: 'STREAMING',
        5: 'VOIP',
        6: 'MAIL'
    }

    # 600000 means we consider all packets in a flow
    NB_PACKETS = [4]


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
        # re-order class names
        _c = classifier.y_train_flows[(classifier.nb_packets_per_flow[0], 0)].unique()        
        classifier.classes = [-1 for _ in range(len(classifier.all_classes))]
        _Xy = classifier.X_train_flows[(classifier.nb_packets_per_flow[0], 0)].copy()
        for index, row in _Xy.iterrows():
            for _i in range(len(classifier.all_classes)):
                if classifier.all_classes[_i] in row['class']:
                    classifier.classes[_i] = row['class']
                    break
            if -1 not in classifier.classes:
                break
        # print("classes =",classifier.classes)
        
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
                classifier._viz(distribution = -1, class_distribution = 0, nb_packets = 0, min_iat = -1, max_iat = -1)
            elif n == 600000:
                classifier._viz(distribution = -1, class_distribution = 11, nb_packets = 1, min_iat = 1, max_iat = -1)
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
        # classifier._nb_packets_distribution(_df, classifier.filename_prefix + "_flows_nb_packets_distribution" )
        # classifier._distribution(_df, classifier.filename_prefix + "_flows_class_split" )
        # classifier._viz(distribution = 0, class_distribution = 11, nb_packets = -1, min_iat = -1, max_iat = -1)
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
        classifier._distribution(_df, classifier.filename_prefix + "_flows_class_split_" + str(pkt) + '_pkt')
        # classifier._class_distribution(_df, classifier.filename_prefix + '_flows_distribution_' + str(pkt) + '_pkt', xylog = True)
        # # classifier._nb_packets_distribution(_df, classifier.filename_prefix + "_flows_nb_packets_distribution_" + str(pkt) + '_pkt')
        # # classifier._min_iat_distribution(_df, classifier.filename_prefix + "_flows_min_iat_distribution_" + str(pkt) + '_pkt')
        sys.exit(1)
        
    if RF_ENABLED:
        print("==== RandomForest =====")
        
        rf_regr_flows, rf_y_train_flows_predicted, rf_y_test_flows_predicted, rf_y_test_flows_isolated_predicted = classifier.RF_predict(
                classifier.X_train_flows_fitted,
                classifier.y_train_flows,
                classifier.X_test_flows_fitted,
                classifier.y_test_flows
        )
                
        # __show_actual_and_predicted(X_test, y_test, rf_y_test_predicted, 1)
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
            
        print("== isolated ==\n")
        from sklearn.metrics import f1_score, confusion_matrix
        print("classifier.y_test_isolated_flows =", classifier.y_test_isolated_flows.shape)
        print("rf_y_test_flows_isolated_predicted =", rf_y_test_flows_isolated_predicted)
        rf_F1 = {}
        skl_F1 = {}
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
                    output += ("for type %s \t\t F1 = %.2f\n" % (t, rf_F1[i][j]))
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


    if GB_ENABLED:
        gb_regr, gb_y_train_predicted, gb_y_test_predicted  = classifier.GBoost_predict(feats_flows, df_score)
        gb_cm_dict = classifier.confusion_matrix(gb_regr, gb_y_test_predicted, False)
        gb_f1_scores = classifier.get_F1_score(df_score, gb_cm_dict,  y_test, gb_y_test_predicted, "gb", False)
        classifier.avg_f1_scores(gb_f1_scores_flows, classifier.flow_ids_without_folds)
        # classifier.avg_f1_scores(gb_f1_scores)

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