import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros

num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        

    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr


    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        return self.propagate(edge_index=edge_index[0], x=x, edge_attr=edge_embeddings)
        # return self.propagate(self.aggr, edge_index=edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class PromptedGNN(torch.nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations
    """

    def __init__(self, gnn_layers, emb_dim, JK="last", drop_ratio=0, prompt_type="add"):
        super().__init__()
        self.num_layer = gnn_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.prompt_type = prompt_type

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        
        if prompt_type not in ['add', 'had']:
            raise ValueError("Illegal prompt type.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)
        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for _ in range(gnn_layers):
            self.gnns.append(GINConv(emb_dim, aggr="add"))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(gnn_layers):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))


    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, prompt = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        else:
            raise ValueError("unmatched number of arguments.")
        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        if len(argv) == 4 and prompt is not None:
            x = x + prompt

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        return node_representation


class GNNBasedNet(torch.nn.Module):
    def __init__(self, emb_dim, gnn_layers, JK, graph_pooling, head_layer, num_tasks=0):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.num_layer = gnn_layers
        self.JK = JK

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        # For graph-level binary classification or discrimination
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1
        if self.JK == "concat":
            self.linears = torch.nn.ModuleList()
            for _ in range(head_layer - 1):
                self.linears.append(torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim,
                                                    self.mult * (self.num_layer + 1) * self.emb_dim))
                self.linears.append(torch.nn.ReLU())
            self.linears.append(torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks))
        else:
            self.linears = torch.nn.ModuleList()
            for _ in range(head_layer - 1):
                self.linears.append(torch.nn.Linear(self.mult * self.emb_dim, self.mult * self.emb_dim))
                self.linears[-1].reset_parameters()
                self.linears.append(torch.nn.ReLU())
            self.linears.append(torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks))
            self.linears[-1].reset_parameters()


    def forward(self, node_representation, batch):
        emb = self.pool(node_representation, batch)  # (batchsize, emb_dim)
        for i in range(len(self.linears)):
            emb = self.linears[i](emb)
        logits = emb
        return logits  # (batchsize, task_num)
