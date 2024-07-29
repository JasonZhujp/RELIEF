import os, argparse, time, random

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.transforms import SVDFeatureReduction
from model import GIN


def load4link_prediction_single_graph(dataname, svd_dim, num_per_samples=1):
    if dataname in ['PubMed', 'CiteSeer', 'Cora']:
        dataset = Planetoid(root='dataset/Planetoid', name=dataname, pre_transform=SVDFeatureReduction(svd_dim))
    elif dataname in ['Computers', 'Photo']:
        dataset = Amazon(root='dataset/Amazon', name=dataname, pre_transform=SVDFeatureReduction(svd_dim))

    data = dataset[0]
    input_dim = dataset.num_features
    output_dim = dataset.num_classes

    r"""Perform negative sampling to generate negative neighbor samples"""
    print("Negative sampling for link prediction data...")
    if data.is_directed():
        row, col = data.edge_index
        row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
        edge_index = torch.stack([row, col], dim=0)
    else:
        edge_index = data.edge_index
    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.num_edges * num_per_samples,
    )
    print("Sampling finished!")

    edge_index = torch.cat([data.edge_index, neg_edge_index], dim=-1)
    edge_label = torch.cat([torch.ones(data.num_edges), torch.zeros(neg_edge_index.size(1))], dim=0)

    return data, edge_label, edge_index, input_dim, output_dim  


class Edgepred_GPPT():
    def __init__(self, dataset_name, hid_dim=128, svd_dim=100, num_layer=2, dropout=0.2, num_epoch=100, lr=0.01, lr_descend=True, mlp=True, device=None):
        self.dataset_name = dataset_name
        self.hid_dim = hid_dim
        self.svd_dim = svd_dim
        self.num_layer = num_layer
        self.dropout = dropout
        self.epochs = num_epoch
        self.device = device
        self.lr = lr
        self.lr_descend = lr_descend
        self.mlp = mlp

        self.dataloader = self.generate_loader_data(svd_dim)
        self.initialize_gnn()  


    def generate_loader_data(self, svd_dim):    
        self.data, edge_label, edge_index, self.input_dim, self.output_dim = load4link_prediction_single_graph(self.dataset_name, svd_dim)
        self.data.to(self.device)
        edge_index = edge_index.transpose(0, 1)
        data = TensorDataset(edge_label, edge_index)
        if self.dataset_name == "Photo":
            batch_size = 128
        else:
            batch_size = 512
        return DataLoader(data, batch_size=batch_size, shuffle=True)
    

    def initialize_gnn(self):
        self.gnn = GIN(input_dim=self.input_dim, hid_dim=self.hid_dim, out_dim=self.hid_dim, num_layer=self.num_layer, drop_ratio=self.dropout)
        self.gnn.to(self.device)
        self.optimizer = Adam(self.gnn.parameters(), lr=self.lr, weight_decay=0.00005)    
    

    def pretrain_one_epoch(self):
        criterion = torch.nn.BCEWithLogitsLoss()
        accum_loss, total_step = 0, 0
        device = self.device

        self.gnn.train()
        for _, (batch_edge_label, batch_edge_index) in enumerate(self.dataloader):
            self.optimizer.zero_grad()

            batch_edge_label = batch_edge_label.to(device)
            batch_edge_index = batch_edge_index.to(device)
            node_emb = self.gnn(self.data.x.to(device), self.data.edge_index.to(device))

            batch_edge_index = batch_edge_index.transpose(0,1)  # (2, batchsize)
            batch_pred_log = (node_emb[batch_edge_index[0]] * node_emb[batch_edge_index[1]]).sum(dim=-1)  # (batchsize,)
            loss = criterion(batch_pred_log, batch_edge_label)

            loss.backward()
            self.optimizer.step()

            accum_loss += float(loss.detach().cpu().item())
            total_step += 1

        return accum_loss / total_step
    

    def pretrain(self):
        num_epoch = self.epochs
        train_loss_min = float('inf')

        for epoch in range(1, num_epoch + 1):
            st_time = time.time()
            if self.lr_descend:
                descend_linear_frac = 1.0 - (epoch - 1) / num_epoch
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = descend_linear_frac * self.lr
            train_loss = self.pretrain_one_epoch()
            
            print(f"[Pretrain] Epoch {epoch}/{num_epoch} | Train Loss {train_loss:.5f} | "
                    f"Cost Time {time.time() - st_time:.3}s")
            if train_loss_min > train_loss:
                train_loss_min = train_loss
                if self.mlp:
                    file_name = "pretrained_models/{}_{}_GIN{}_svd{}_hid{}(lr{}_{}_e{}_mlp).pth".format(
                                'GPPT', self.dataset_name, str(self.num_layer), str(self.svd_dim), str(self.hid_dim),
                                str(self.lr), str(self.lr_descend), str(self.epochs))
                else:
                    file_name = "pretrained_models/{}_{}_GIN{}_svd{}_hid{}(lr{}_{}_e{}).pth".format(
                                'GPPT', self.dataset_name, str(self.num_layer), str(self.svd_dim), str(self.hid_dim),
                                str(self.lr), str(self.lr_descend), str(self.epochs))
                torch.save(self.gnn.state_dict(), file_name)
                print("+++model saved ! {}".format(file_name))


def seed_everything(seed, device):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda:" + str(device)) if torch.cuda.is_available() else torch.device("cpu")
    return device


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--lr_descend', type=int, default=0)
    parser.add_argument('--mlp', type=int, default=0)
    parser.add_argument('--decay', type=float, default=0, help='weight decay (default: 0)')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--num_layer', type=int, default=2, help='number of GNN message passing layers (default: 2).')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout ratio (default: 0.2)')
    parser.add_argument('--JK', type=str, default="last", help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--hid_dim', type=int, default=128)
    parser.add_argument('--svd_dim', type=int, default=100)
    parser.add_argument('--dataset_name', type=str, default='CiteSeer')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--runseed', type=int, default=1)
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')

    args = parser.parse_args()
    args = parser.parse_args(args=[
        '--dataset_name', 'Cora',      '--lr', '0.01',   '--epochs', '100',  '--lr_descend', '1',
        # '--dataset_name', 'CiteSeer',  '--lr', '0.001',  '--epochs', '100',  '--lr_descend', '0',
        # '--dataset_name', 'PubMed',    '--lr', '0.01',   '--epochs', '100',  '--lr_descend', '0',
        # '--dataset_name', 'Photo',     '--lr', '0.0005', '--epochs', '100',  '--lr_descend', '1',
        # '--dataset_name', 'Computers', '--lr', '0.0001', '--epochs', '500',  '--lr_descend', '1',
        '--device', '0'
        ])
    return args


def main():
    args = set_args()
    device = seed_everything(args.seed, args.device)
    pt = Edgepred_GPPT(args.dataset_name, args.hid_dim, args.svd_dim, args.num_layer, args.dropout, args.epochs, args.lr, args.lr_descend, args.mlp, device)
    pt.pretrain()


if __name__ == "__main__":
    print('PID[{}]'.format(os.getpid()))
    main()