# Copyright 2020-2021 eBay Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gc
import os
import sys
path = "/".join([x for x in os.path.realpath(__file__).split('/')[:-2]])
sys.path.insert(0, path)
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import glob
import fire
from config import parse_args
from Utils.fstore import FeatureStore


def main(args, path_db='data/feat_store.db'):
    data_path = './'
    store = FeatureStore(path_db)
    node_type = ['bank', 'account', 'address']
    uid_cols = ['MessageId', 'Timestamp', 'UETR', 'Sender', 'Receiver', 'OrderingAccount', 'BeneficiaryAccount', 'Label', 'OrderingOriginAdd', 'BeneficiaryOriginAdd']

    # Train
    df = pd.read_csv(data_path+args.train_feat_path)
    feature_cols = list(df.columns)
    for i in uid_cols:
        feature_cols.remove(i)
    temp_df = df[['MessageId']+feature_cols]
    x = temp_df.values
    with store.db.write_batch() as wb:
        for i in tqdm(range(x.shape[0])):
            key = int(x[i, 0])
            value = x[i, 1:]
            store.put(key, value, wb=wb, dtype=np.float32)
    del(temp_df)
    gc.collect()
    print("Done for train dataset")
    # Test
    df = pd.read_csv(data_path+args.test_feat_path)
    feature_cols = list(df.columns)
    for i in uid_cols:
        feature_cols.remove(i)
    temp_df = df[['MessageId'] + feature_cols]
    x = temp_df.values
    with store.db.write_batch() as wb:
        for i in tqdm(range(x.shape[0])):
            key = int(x[i, 0])
            value = x[i, 1:]
            store.put(key, value, wb=wb, dtype=np.float32)
    del (temp_df)
    gc.collect()
    print("Done for test dataset")

    # Bank
    df = pd.read_csv(data_path+args.bank_path)
    feature_cols = list(df.columns)
    feature_cols.remove('id')
    temp_df = df[['id'] + feature_cols].copy()
    print(temp_df.head())
    temp_df.drop('Sender',axis=1, inplace=True)
    x = temp_df.values
    with store.db.write_batch() as wb:
        for i in tqdm(range(x.shape[0])):
            key = int(x[i, 0])
            value = x[i, 1:]
            store.put(key, value, wb=wb, dtype=np.float32)
    del (temp_df)
    gc.collect()
    print("Done for bank dataset")

    # Acc
    df = pd.read_csv(data_path+args.acc_path)
    feature_cols = list(df.columns)
    feature_cols.remove('id')
    temp_df = df[['id'] + feature_cols].copy()
    temp_df.drop('OrderingAccount', axis=1, inplace=True)
    x = temp_df.values
    with store.db.write_batch() as wb:
        for i in tqdm(range(x.shape[0])):
            key = int(x[i, 0])
            value = x[i, 1:]
            store.put(key, value, wb=wb, dtype=np.float32)
    del (temp_df)
    gc.collect()
    print("Done for Acc Dataset")

    # Add
    df = pd.read_csv(data_path+args.add_path)
    feature_cols = list(df.columns)
    feature_cols.remove('id')
    temp_df = df[['id'] + feature_cols].copy()
    temp_df.drop('OrderingOriginAdd', axis=1, inplace=True)
    x = temp_df.values
    with store.db.write_batch() as wb:
        for i in tqdm(range(x.shape[0])):
            key = int(x[i, 0])
            value = x[i, 1:]
            store.put(key, value, wb=wb, dtype=np.float32)
    del (temp_df)
    gc.collect()
    print("Done for Add dataset")




if __name__=="__main__":
    args = parse_args()
    main(args=args)