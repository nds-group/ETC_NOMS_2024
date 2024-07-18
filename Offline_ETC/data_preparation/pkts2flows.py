#!/usr/bin/env python
# coding: utf-8

from os import listdir
from os.path import join

import pandas as pd
import numpy as np

def __pkts2flow(path, outpath, filename, first_flow_id):
    print("opening", join(path, filename))
    df = pd.read_csv(join(path, filename), names = ["packet_id", "timestamp", "iat", "src", "psrc", "dst", "pdst", "protocol", "length"], dtype={'packet_id': 'int', 'timestamp': 'float', 'iat': 'float', 'src': 'str', "psrc": 'int', 'dst':'str', 'pdst': 'int', 'protocol': 'int', 'length': 'int'},header = 0)
    df['protocol'].replace('', np.nan, inplace = True) 
    df = df.dropna(axis = 1)
    df['flow_id'] = df.groupby(['src', 'psrc', 'dst', 'pdst', 'protocol']).ngroup()
    df['flow_id'] = df['flow_id'].astype('int')
    
    print(df.shape)

    # update flow_id to consecutive values
    df = df.sort_values(by = ['flow_id'])
    df.flow_id = df.flow_id.ne(df.flow_id.shift()).cumsum().add(first_flow_id).astype('int')
    
    r = df['flow_id'].max()
    df.to_csv(join(outpath, filename), index = False)
    return r

def main():    
    # change inputdir to the full name of the directory where the dataset CSV files are stored.
    paths = ["inputdir"]
    first_flow_id = 0
    # change outputdir to the full name of the directory where you want to store the new CSV files
    output_path = "outputdir"
    
    for path in paths:
        for f in listdir(path):
            if "csv" in f:
                first_flow_id = __pkts2flow(path, output_path, f, first_flow_id)                
 
if __name__ == "__main__":
    main()