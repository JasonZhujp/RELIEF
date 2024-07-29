import os, random, argparse, pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.transforms import NormalizeFeatures, SVDFeatureReduction
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, k_hop_subgraph
from torch_geometric.loader import NeighborLoader


def load4node(args):
    if args.dataset in ['PubMed', 'CiteSeer', 'Cora']:
        if args.norm:
            dataset = Planetoid(root='dataset/Planetoid', name=args.dataset, split='random', num_train_per_class=args.shot_number, 
                                transform=NormalizeFeatures(), pre_transform=SVDFeatureReduction(out_channels=args.svd_dim))
        else:
            dataset = Planetoid(root='dataset/Planetoid', name=args.dataset, split='random', num_train_per_class=args.shot_number, 
                                pre_transform=SVDFeatureReduction(out_channels=args.svd_dim))
    
    elif args.dataset in ['Computers', 'Photo']:
        if args.norm:
            dataset = Amazon(root='dataset/Amazon', name=args.dataset,
                             transform=NormalizeFeatures(), pre_transform=SVDFeatureReduction(out_channels=args.svd_dim))
        else:
            dataset = Amazon(root='dataset/Amazon', name=args.dataset, pre_transform=SVDFeatureReduction(out_channels=args.svd_dim))
        
    print()
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    data = dataset[0]  # Get the first graph object.
    print(data)
    print('=' * 50)

    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')

    return data, dataset.num_classes


def subgraph_node_num(args, og_data: Data):
    print("Creating node induced {}-hop subgraphs...".format(args.num_hops))
    edge_index = og_data.edge_index
    all_subgraph_node_num = []

    for i in range(og_data.num_nodes):
        if (i+1) % 100 == 0 or i == 0:
            print("Computing # subgraph nodes [{}/{}]".format(i+1, og_data.num_nodes))
        subset, _, _, _ = k_hop_subgraph(node_idx=i, num_hops=args.num_hops, edge_index=edge_index)
        subgraph_node_num = subset.size(0)
        all_subgraph_node_num.append(subgraph_node_num)

    print("=== Computing # subgraph nodes finished! MIN {}  MAX {}  MEAN {:.2f}  MEDIAN {}".format(
        torch.min(torch.tensor(all_subgraph_node_num)), torch.max(torch.tensor(all_subgraph_node_num)),
        torch.mean(torch.tensor(all_subgraph_node_num).float()), torch.median(torch.tensor(all_subgraph_node_num))
    ))

    if not os.path.exists("subgraph/node_num"):
        os.makedirs("subgraph/node_num")
    file_name = "subgraph/node_num/" + args.dataset + "_" + str(args.num_hops) + "hop_node_num.dat"
    pickle.dump(all_subgraph_node_num, open(file_name, "bw"))
    print("# subgraph nodes saved at {}".format(file_name))

    return all_subgraph_node_num


def split_subgraph(args, og_data: Data, num_classes, all_subgraph_node_num):
    og_x = og_data.x
    og_y = og_data.y
    edge_index = og_data.edge_index
    subgraph_node_num = torch.tensor(all_subgraph_node_num)
    all_y = og_y
    min_nnl_idx = torch.nonzero(args.min_node_num_limit <= subgraph_node_num, as_tuple=False).view(-1).tolist()
    max_nnl_idx = torch.nonzero(args.max_node_num_limit >= subgraph_node_num, as_tuple=False).view(-1).tolist()
    valid_idx = torch.tensor(list(set(min_nnl_idx) & set(max_nnl_idx)))

    train_idx = []
    for c in range(num_classes):
        class_idx = (all_y == c).nonzero(as_tuple=False).view(-1)
        idx = valid_idx[torch.isin(valid_idx, class_idx)]
        idx = idx[torch.randperm(idx.size(0))[:args.shot_number]]
        train_idx.append(idx)
    train_idx = torch.cat(train_idx)

    remaining = valid_idx[torch.isin(valid_idx, train_idx, invert=True)]
    remaining = remaining[torch.randperm(remaining.size(0))]
    val_idx = remaining[:500]
    test_idx = remaining[500 : 500 + 500]
    assert torch.isin(train_idx, val_idx).sum()==0 and torch.isin(train_idx, test_idx).sum()==0 and torch.isin(val_idx, test_idx).sum()==0
    assert len(val_idx) == 500 and len(test_idx) == 500


    def generate_subgraph(og_x, og_y, idx):
        dataset, node_num = [], []
        for cnt, id in enumerate(idx):
            if (cnt+1) % 50 == 0 or cnt == 0:
                print("\tsubgraph {}/{}".format(cnt+1, len(idx)))
            subset, _, _, _ = k_hop_subgraph(node_idx=id.item(), num_hops=args.num_hops, edge_index=edge_index)
            sub_edge_index, _ = subgraph(subset, edge_index, relabel_nodes=True)
            induced_graph = Data(x=og_x[subset], edge_index=sub_edge_index, y=og_y[id])
            dataset.append(induced_graph)
            node_num.append(subset.size(0))
        return dataset, torch.tensor(node_num)

    def sample_subgraph(og_data, neighbor_nums, train_idx, val_idx, test_idx):
        train_dataset, val_dataset, test_dataset = [], [], []
        train_node_num, val_node_num, test_node_num = [], [], []
        nums = neighbor_nums.split("_")
        num_neighbors = [int(num) for num in nums]
        subgraph_loader = NeighborLoader(og_data, num_neighbors=num_neighbors, batch_size=1, shuffle=False)
        for i, subgraph in enumerate(subgraph_loader):
            if (i+1) % 1000 == 0 or i == 0:
                print("Processing [{}/{}]".format(i+1, og_data.num_nodes))
            if i in train_idx:
                target_node_id = subgraph.input_id.item()
                sample_subgraph = Data(x=subgraph.x, edge_index=subgraph.edge_index, y=og_data.y[target_node_id])
                train_dataset.append(sample_subgraph)
                train_node_num.append(sample_subgraph.num_nodes)
            elif i in val_idx:
                target_node_id = subgraph.input_id.item()
                sample_subgraph = Data(x=subgraph.x, edge_index=subgraph.edge_index, y=og_data.y[target_node_id])
                val_dataset.append(sample_subgraph)
                val_node_num.append(sample_subgraph.num_nodes)
            elif i in test_idx:
                target_node_id = subgraph.input_id.item()
                sample_subgraph = Data(x=subgraph.x, edge_index=subgraph.edge_index, y=og_data.y[target_node_id])
                test_dataset.append(sample_subgraph)
                test_node_num.append(sample_subgraph.num_nodes)
        return train_dataset, val_dataset, test_dataset, torch.tensor(train_node_num), torch.tensor(val_node_num), torch.tensor(test_node_num)    

    if args.dataset in ["Computers", "Photo"]:
        print("Generating Train Val Test (sampled) dataset...")
        train_dataset, val_dataset, test_dataset, train_node_num, val_node_num, test_node_num = \
            sample_subgraph(og_data, args.neighbor_nums, train_idx, val_idx, test_idx)
    else:
        print("Generating Train dataset...")
        train_dataset, train_node_num = generate_subgraph(og_x, og_y, train_idx)
        print("Generating Val dataset...")
        val_dataset, val_node_num = generate_subgraph(og_x, og_y, val_idx)
        print("Generating Test dataset...")
        test_dataset, test_node_num = generate_subgraph(og_x, og_y, test_idx)

    print("\nMIN MAX MEAN MEDIAN #nodes in train/val/test: {} {} {:.2f} {} | {} {} {:.2f} {} | {} {} {:.2f} {}".format(
        torch.min(train_node_num), torch.max(train_node_num), torch.mean(train_node_num.float()), torch.median(train_node_num),
        torch.min(val_node_num), torch.max(val_node_num), torch.mean(val_node_num.float()), torch.median(val_node_num),
        torch.min(test_node_num), torch.max(test_node_num), torch.mean(test_node_num.float()), torch.median(test_node_num)
    ))
    print("Node per class:\nTRAIN:{}\nVAL  :{}\nTEST :{}".format(
        torch.bincount(torch.tensor([d.y for d in train_dataset]), minlength=num_classes),
        torch.bincount(torch.tensor([d.y for d in val_dataset]), minlength=num_classes),
        torch.bincount(torch.tensor([d.y for d in test_dataset]), minlength=num_classes)
    ))
    
    dataset = {"train_dataset": train_dataset, "val_dataset": val_dataset, "test_dataset": test_dataset,
               "train_idx": train_idx.tolist(), "val_idx": val_idx.tolist(), "test_idx": test_idx.tolist()}
    if args.dataset in ["Computers", "Photo"]:
        file_name = "subgraph/split_data/"+args.dataset+"_"+str(args.num_hops)+"hop_svd"+str(args.svd_dim)+"_"+str(args.shot_number)+"shots_nnl"+\
                    str(args.min_node_num_limit)+"-"+str(args.max_node_num_limit)+"(" + args.neighbor_nums + ")_seed"+str(args.seed)+"_split.dat"
    else:
        file_name = "subgraph/split_data/"+args.dataset+"_"+str(args.num_hops)+"hop_svd"+str(args.svd_dim)+"_"+str(args.shot_number)+"shots_nnl"+\
                    str(args.min_node_num_limit)+"-"+str(args.max_node_num_limit)+"_seed"+str(args.seed)+"_split.dat"
    
    if not os.path.exists("subgraph/split_data"):
        os.makedirs("subgraph/split_data")
    pickle.dump(dataset, open(file_name, "bw"))
    print("Split subgraph saved at {}".format(file_name))


def main():
    args = set_args()
    seed_everything(args.seed)
    og_data, num_classes = load4node(args)
    # if it is the first time to run this code
    if args.subgraph_node_num_file == "":
        all_subgraph_node_num = subgraph_node_num(args, og_data)
    # diretly using existence `xxxx_xhop_node_num.dat` file to create and split subgraph
    else:
        all_subgraph_node_num = pickle.load(open(args.subgraph_node_num_file, 'br'))
        print("=== # subgraph nodes: MIN {}  MAX {}  MEAN {:.2f}  MEDIAN {}".format(
            torch.min(torch.tensor(all_subgraph_node_num)), torch.max(torch.tensor(all_subgraph_node_num)),
            torch.mean(torch.tensor(all_subgraph_node_num).float()), torch.median(torch.tensor(all_subgraph_node_num))
        ))
    split_subgraph(args, og_data, num_classes, all_subgraph_node_num)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hid_dim', type=int, default=128)
    parser.add_argument('--svd_dim', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--norm', type=int, default=0)
    parser.add_argument('--num_hops', type=int, default=2)
    parser.add_argument('--shot_number', type=int, default=10, help='Number of samples for each class')
    parser.add_argument('--subgraph_node_num_file', type=str, default='', help="Node number of each subgraphs (already saved under the folder `subgraph/node_num/`)")
    parser.add_argument('--min_node_num_limit', type=int, default=5, help='Ensure an appropriate number of nodes for induced subgraphs')
    parser.add_argument('--max_node_num_limit', type=int, default=200)
    parser.add_argument('--neighbor_nums', type=str, default="10_10")

    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    args = parser.parse_args(args=[
        # '--dataset', 'Cora',  '--num_hops', '4',
        # '--subgraph_node_num_file', 'subgraph/node_num/Cora_4hop_node_num.dat',
        # '--min_node_num_limit', '5', '--max_node_num_limit', '150',

        # '--dataset', 'CiteSeer', '--num_hops', '3',
        # '--subgraph_node_num_file', 'subgraph/node_num/CiteSeer_3hop_node_num.dat',
        # '--min_node_num_limit', '20', '--max_node_num_limit', '150',

        # '--dataset', 'PubMed', '--num_hops', '2',
        # '--subgraph_node_num_file', 'subgraph/node_num/PubMed_2hop_node_num.dat',
        # '--min_node_num_limit', '50', '--max_node_num_limit', '300',

        # '--dataset', 'Computers', '--num_hops', '2',
        # '--subgraph_nozde_num_file', 'subgraph/node_num/Computers_2hop_node_num.dat',
        # '--min_node_num_limit', '100', '--max_node_num_limit', '500', '--neighbor_nums', '10_10',

        # '--dataset', 'Photo', '--num_hops', '2',
        # '--subgraph_node_num_file', 'subgraph/node_num/Photo_2hop_node_num.dat',
        # '--min_node_num_limit', '200', '--max_node_num_limit', '2000',  '--neighbor_nums', '20_10',
    ])
    return args


if __name__ == "__main__":
    print('PID[{}]'.format(os.getpid()))
    main()