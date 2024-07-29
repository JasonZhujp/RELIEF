import os, argparse, time, random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch_geometric.utils import structured_negative_sampling
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.transforms import SVDFeatureReduction
from model import GIN


def Gprompt_link_loss(node_emb, pos_emb, neg_emb, temperature=0.2):
    r"""Refer to GraphPrompt original codes"""
    x = torch.exp(F.cosine_similarity(node_emb, pos_emb, dim=-1) / temperature)
    y = torch.exp(F.cosine_similarity(node_emb, neg_emb, dim=-1) / temperature)

    loss = -1 * torch.log(x / (x + y))
    return loss.mean()


def edge_index_to_sparse_matrix(edge_index: torch.LongTensor, num_node: int):
    node_idx = torch.LongTensor([i for i in range(num_node)])
    self_loop = torch.stack([node_idx, node_idx], dim=0)
    edge_index = torch.cat([edge_index, self_loop], dim=1)
    sp_adj = torch.sparse.FloatTensor(edge_index, torch.ones(edge_index.size(1)), torch.Size((num_node, num_node)))

    return sp_adj


def prepare_structured_data(graph_data: Data):
    r"""Prepare structured <i,k,j> format link prediction data"""
    node_idx = torch.LongTensor([i for i in range(graph_data.num_nodes)])
    self_loop = torch.stack([node_idx, node_idx], dim=0)
    edge_index = torch.cat([graph_data.edge_index, self_loop], dim=1)
    v, a, b = structured_negative_sampling(edge_index, graph_data.num_nodes)
    data = torch.stack([v, a, b], dim=1)

    # (num_edge, 3)
    #   for each entry (i,j,k) in data, (i,j) is a positive sample while (i,k) forms a negative sample
    return data


def load_single_graph(dataset_name, svd_dim=100):
    if dataset_name in ['PubMed', 'CiteSeer', 'Cora']:
        dataset = Planetoid(root='dataset/Planetoid', name=dataset_name, pre_transform=SVDFeatureReduction(svd_dim))
    elif dataset_name in ['Computers', 'Photo']:
        dataset = Amazon(root='dataset/Amazon', name=dataset_name, pre_transform=SVDFeatureReduction(svd_dim))
    data = dataset[0]
    input_dim = dataset.num_features
    output_dim = dataset.num_classes
    
    return data, input_dim, output_dim   


class Edgepred_Gprompt():
    def __init__(self, dataset_name, hid_dim=128, svd_dim=100, num_layer=2, dropout=0.2, lr=0.001, lr_descend=0, mlp=0, num_epoch=100, device=None):
        self.dataset_name = dataset_name
        self.hid_dim = hid_dim
        self.svd_dim = svd_dim
        self.num_layer = num_layer
        self.dropout = dropout
        self.lr = lr
        self.lr_descend = lr_descend
        self.mlp = mlp
        self.epochs = num_epoch
        self.device = device

        self.dataloader = self.generate_loader_data(svd_dim)
        self.initialize_gnn()  


    def generate_loader_data(self, svd_dim):    
        self.data, self.input_dim, self.output_dim = load_single_graph(self.dataset_name, svd_dim)
        self.adj = edge_index_to_sparse_matrix(self.data.edge_index, self.data.x.shape[0]).to(self.device)
        print("Generate structured <i,k,j> format link prediction data...")
        data = prepare_structured_data(self.data)
        print("Generate finished!")
        batch_size = 128
        return DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)
    

    def initialize_gnn(self):
        self.gnn = GIN(input_dim=self.input_dim, hid_dim=self.hid_dim, out_dim=self.hid_dim, num_layer=self.num_layer, drop_ratio=self.dropout)
        self.gnn.to(self.device)
        self.optimizer = Adam(self.gnn.parameters(), lr=self.lr, weight_decay=0.00005)
    

    def pretrain_one_epoch(self):
        accum_loss, total_step = 0, 0
        device = self.device
        self.gnn.train()
        for i, batch in enumerate(self.dataloader):
            self.optimizer.zero_grad()

            batch = batch[0]
            batch = batch.to(device)

            node_emb = self.gnn(self.data.x.to(device), self.data.edge_index.to(device))
            all_node_emb = torch.sparse.mm(self.adj, node_emb)
            node_emb = all_node_emb[batch[:, 0]]
            pos_emb, neg_emb = all_node_emb[batch[:, 1]], all_node_emb[batch[:, 2]]

            loss = Gprompt_link_loss(node_emb, pos_emb, neg_emb)
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
                                'Gprompt', self.dataset_name, str(self.num_layer), str(self.svd_dim), str(self.hid_dim),
                                str(self.lr), str(self.lr_descend), str(self.epochs))
                else:
                    file_name = "pretrained_models/{}_{}_GIN{}_svd{}_hid{}(lr{}_{}_e{}).pth".format(
                                'Gprompt', self.dataset_name, str(self.num_layer), str(self.svd_dim), str(self.hid_dim),
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
        '--dataset_name', 'Cora',     '--lr', '0.01',  '--lr_descend', '0', '--epochs', '100',
        # '--dataset_name', 'CiteSeer', '--lr', '0.001', '--lr_descend', '0', '--epochs', '100',
        # '--dataset_name', 'PubMed',   '--lr', '0.01',  '--lr_descend', '0', '--epochs', '100',
        # '--dataset_name', 'Photo',    '--lr', '0.005', '--lr_descend', '0', '--epochs', '100',
        # '--dataset_name', 'Computers','--lr', '0.005', '--lr_descend', '0', '--epochs', '100',
        '--device', '0'
    ])
    return args


def main():
    args = set_args()
    device = seed_everything(args.seed, args.device)
    pt = Edgepred_Gprompt(args.dataset_name, args.hid_dim, args.svd_dim, args.num_layer, args.dropout, args.lr, args.lr_descend, args.mlp, args.epochs, device)
    pt.pretrain()


if __name__ == "__main__":
    print('PID[{}]'.format(os.getpid()))
    main()