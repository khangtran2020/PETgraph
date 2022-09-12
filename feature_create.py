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
import pandas as pd
import numpy as np
import networkx as nx
import tqdm
import glob
import fire

from config import parse_args
from utils.fstore import FeatureStore


def main(args, path_db='./data/feat_store.db'):

    store = FeatureStore(path_db)
    node_type = ['bank', 'account', 'address']
    uid_cols = ['MessageId', 'Timestamp', 'UETR', 'Sender', 'Receiver', 'OrderingAccount', 'BeneficiaryAccount', 'Label', 'OrderingOriginAdd', 'BeneficiaryOriginAdd']
    if (args.mode == 'train'):
        df = pd.read_parquet(args.data_path + args.train_feat_path)
    else:
        df = pd.read_parquet(args.data_path + args.test_feat_path)
    feature_cols = list(df.columns)
    for i in uid_cols:
        feature_cols.remove(i)
    temp_df = df[['MessageId']+feature_cols]
    x = temp_df.values
    with store.db.write_batch() as wb:
        for i in range(x.shape[0]):
            key = x[i, 0]
            value = x[i, 1:]
            store.put(key, value, wb=wb, dtype=np.float32)
    del(temp_df)
    gc.collect()


    node_src = df['src'].drop_duplicates().tolist()
    node_dst = df['dst'].drop_duplicates().tolist()
    graph = nx.from_pandas_edgelist(df, source='src', target='dst')

    for node_id in tqdm.tqdm(node_dst):
        sum_feat = None
        cnt_neighbor = 0
        for neighbor in graph.neighbors(node_id):
            if neighbor in node_src:
                neighbor_feat = store.get(key=neighbor, default_value=None)
                if sum_feat is None:
                    sum_feat = neighbor_feat.copy()
                else:
                    sum_feat += neighbor_feat
                cnt_neighbor += 1
        store.put(key=node_id, value=sum_feat/cnt_neighbor)


if __name__=="__main__":
    args = parse_args()
    fire.Fire(main(args=args))