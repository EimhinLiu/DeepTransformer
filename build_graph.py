import pandas as pd
from torch_geometric.data import Data
import torch
import numpy as np

edge_index = torch.from_numpy(pd.read_csv(r'.\data\graph\raw\edge.csv', header = None).values.T.astype(np.int64))
edge_feature = torch.from_numpy(pd.read_csv(r'.\data\graph\raw\edge_feature.csv', header = None).values.astype(np.float32))
node_label = torch.from_numpy(pd.read_csv(r'.\data\graph\raw\node_label.csv', header = None).values.astype(np.float32))
node_feature = torch.from_numpy(pd.read_csv(r'.\data\graph\raw\node_feature.csv', header = None).values.astype(np.float32))

data = Data(edge_index = edge_index, edge_attr = edge_feature, y = node_label, x = node_feature)

train_idx = torch.from_numpy(pd.read_csv(r'.\data\graph\split\train.csv', header = None).values.T[0]).to(torch.long)
valid_idx = torch.from_numpy(pd.read_csv(r'.\data\graph\split\valid.csv', header = None).values.T[0]).to(torch.long)
test_idx = torch.from_numpy(pd.read_csv(r'.\data\graph\split\test.csv', header = None).values.T[0]).to(torch.long)
evaluate_idx = torch.from_numpy(pd.read_csv(r'.\data\graph\split\evaluate.csv', header = None).values.T[0]).to(torch.long) 
splitted_idx = {'train': train_idx, 'valid': valid_idx, 'test': test_idx, 'evaluate': evaluate_idx}
for split in ['train', 'valid', 'test', 'evaluate']:
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[splitted_idx[split]] = True
    data[f'{split}_mask'] = mask
    
# print(data)
# print(data.is_undirected()) # Undirected graph needs two edges between the nodes.