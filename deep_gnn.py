import os
import time
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, ReLU, Sequential
from tqdm import tqdm
from torch_geometric.loader import RandomNodeSampler
from torch_geometric.nn import DeepGCNLayer
from torch_geometric.nn import GENConv, GATConv, GATv2Conv, TransformerConv, GeneralConv, GMMConv, NNConv

import utils
from utils import EarlyStopping
import build_graph

utils.setup_seed(30)

data = build_graph.data

train_loader = RandomNodeSampler(data, num_parts=136, shuffle=True, num_workers=0) # batch_size=num_nodes/num_parts
test_loader = RandomNodeSampler(data, num_parts=17, num_workers=0)


GNNConv = 'TransformerConv'
skip_connection = 'res+' # res+(the pre-activation residual connection), res(the residual connection), plain(no connection)
conv_layers = 25
class DeeperGNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers):
        super().__init__()

        self.node_encoder = Linear(data.x.size(-1), hidden_channels)
        self.edge_encoder = Linear(data.edge_attr.size(-1), hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            if GNNConv == 'GENConv':
                conv = GENConv(hidden_channels, hidden_channels, learn_t=True, norm='layer')
            elif GNNConv == 'GATConv':
                conv = GATConv(hidden_channels, hidden_channels, heads=4, concat=False, edge_dim=64)
            elif GNNConv == 'GATv2Conv':
                conv = GATv2Conv(hidden_channels, hidden_channels, heads=4, concat=False, edge_dim=64)
            elif GNNConv == 'TransformerConv':
                conv = TransformerConv(hidden_channels, hidden_channels, heads=4, concat=False, edge_dim=64)
            elif GNNConv == 'GeneralConv':
                conv = GeneralConv(hidden_channels, hidden_channels, in_edge_channels=64, attention=True, heads=4)
            elif GNNConv == 'GMMConv':
                conv = GMMConv(hidden_channels, hidden_channels, dim=64, kernel_size=3)
            elif GNNConv == 'NNConv':
                nn = Sequential(Linear(64, 128), ReLU(), Linear(128, hidden_channels * hidden_channels))
                conv = NNConv(hidden_channels, hidden_channels, nn=nn)
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block=skip_connection, dropout=0.1, ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = Linear(hidden_channels, data.y.size(-1))

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        x = self.layers[0].conv(x, edge_index, edge_attr)
        
        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)
        
        x = self.layers[0].norm(x)
        x = self.layers[0].act(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        return self.lin(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeeperGNN(hidden_channels=64, num_layers=conv_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.BCEWithLogitsLoss()
early_stopping = EarlyStopping(patience=70, verbose=True, path=os.path.join(os.path.dirname(__file__), 'checkpoints', str(conv_layers)+'_'+GNNConv+'_'+skip_connection+'_Checkpoint.pt'))


def train(epoch):
    model.train()

    pbar = tqdm(total=len(train_loader))
    pbar.set_description(f'Training epoch: {epoch:04d}')

    total_loss = total_examples = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * int(data.train_mask.sum())
        total_examples += int(data.train_mask.sum())

        pbar.update(1)

    pbar.close()

    return total_loss / total_examples


@torch.no_grad()
def test():
    model.eval()

    y_true = {'train': [], 'valid': [], 'test': []}
    y_pred = {'train': [], 'valid': [], 'test': []}

    pbar = tqdm(total=len(test_loader))
    pbar.set_description(f'Testing epoch: {epoch:04d}')

    total_loss = total_examples = 0
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr)
        
        loss = criterion(out[data.valid_mask], data.y[data.valid_mask])
        total_loss += float(loss) * int(data.valid_mask.sum())
        total_examples += int(data.valid_mask.sum())

        for split in y_true.keys():
            mask = data[f'{split}_mask']
            y_true[split].append(data.y[mask].cpu())
            y_pred[split].append(out[mask].cpu())

        pbar.update(1)

    pbar.close()

    train_rocauc = utils.rocauc({
        'y_true': torch.cat(y_true['train'], dim=0),
        'y_pred': torch.cat(y_pred['train'], dim=0),
    })['rocauc']

    valid_rocauc = utils.rocauc({
        'y_true': torch.cat(y_true['valid'], dim=0),
        'y_pred': torch.cat(y_pred['valid'], dim=0),
    })['rocauc']

    test_rocauc = utils.rocauc({
        'y_true': torch.cat(y_true['test'], dim=0),
        'y_pred': torch.cat(y_pred['test'], dim=0),
    })['rocauc']

    train_prauc = utils.prauc({
        'y_true': torch.cat(y_true['train'], dim=0),
        'y_pred': torch.cat(y_pred['train'], dim=0),
    })['prauc']
    
    valid_prauc = utils.prauc({
        'y_true': torch.cat(y_true['valid'], dim=0),
        'y_pred': torch.cat(y_pred['valid'], dim=0),
    })['prauc']
    
    test_prauc = utils.prauc({
        'y_true': torch.cat(y_true['test'], dim=0),
        'y_pred': torch.cat(y_pred['test'], dim=0),
    })['prauc']

    valid_loss = total_loss / total_examples
    
    return train_rocauc, valid_rocauc, test_rocauc, train_prauc, valid_prauc, test_prauc, valid_loss, torch.cat(y_true['test'], dim=0), torch.cat(y_pred['test'], dim=0)


max_test_rocauc = 0
max_test_prauc = 0
metrics=[]
for epoch in range(1, 701):
    train_loss = train(epoch)
    train_rocauc, valid_rocauc, test_rocauc, train_prauc, valid_prauc, test_prauc, valid_loss, y_true, y_pred = test()
    
    print(f'TrainLoss: {train_loss:.4f}, ValidLoss: {valid_loss:.4f}, '
          f'TrainRocauc: {train_rocauc:.4f}, ValidRocauc: {valid_rocauc:.4f}, TestRocauc: {test_rocauc:.4f}, '
          f'TrainPrauc: {train_prauc:.4f}, ValidPrauc: {valid_prauc:.4f}, TestPrauc: {test_prauc:.4f}')
    
    metric = [epoch, train_loss, valid_loss, test_rocauc, test_prauc]
    metrics.append(metric)
    
    if test_rocauc > max_test_rocauc and test_prauc > max_test_prauc:
        max_test_rocauc = test_rocauc
        max_test_prauc = test_prauc
        fpr, tpr = utils.roccurve({
            'y_true': y_true,
            'y_pred': y_pred,
            })
        precision, recall = utils.prcurve({
            'y_true': y_true,
            'y_pred': y_pred,
            })
        roc_curve_values = pd.DataFrame({'FPR':fpr, 'TPR':tpr})
        roc_curve_values.to_csv(rf'.\drawing\roc_{str(conv_layers)}_{GNNConv}_{skip_connection}.csv', index = False)
        pr_curve_values = pd.DataFrame({'Recall':recall, 'Precision':precision})
        pr_curve_values.to_csv(rf'.\drawing\pr_{str(conv_layers)}_{GNNConv}_{skip_connection}.csv', index = False)
    
    early_stopping(valid_loss, model)
    if early_stopping.early_stop:
        print("*******!Early Stopping!*******")
        break
    
    time.sleep(0.3)

utils.save_result(metrics=metrics, path=os.path.join(os.path.dirname(__file__), 'metrics', str(conv_layers)+'_'+GNNConv+'_'+skip_connection+'.xlsx'))


@torch.no_grad()
def evaluate():
    model.load_state_dict(torch.load(rf'.\checkpoints\{str(conv_layers)}_{GNNConv}_{skip_connection}_Checkpoint.pt'))
    model.eval()
    Model = model.cpu()
    
    out = Model(data.x, data.edge_index, data.edge_attr)
    
    evaluate_true = sum(data.y[data['evaluate_mask']].tolist(), []) # 'sum' is used for list dimension reduction.
    evaluate_pred = sum((out[data['evaluate_mask']] > 0).float().tolist(), [])
    
    external_testing_results = pd.DataFrame({'TrueValue':evaluate_true, 'PredictedValue':evaluate_pred})
    external_testing_results.to_csv(rf'.\external_testing_results\{str(conv_layers)}_{GNNConv}_{skip_connection}.csv', index = False)

evaluate()