import argparse, random, os
import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

from torch_geometric.loader import DataLoader
import torch
import torch.optim as optim
import numpy as np
import pandas as pd

from splitters import scaffold_split_fewshot
from loader import MoleculeDataset
from model import PromptedGNN, GNNBasedNet
from agent import H_PPO
from base import *
from util import logargs


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bbbp', help='Root directory of dataset')
    parser.add_argument('--shot_number', type=float, default=50)
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting the dataset")
    parser.add_argument('--runseed', type=int, default=1, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--device', type=int, default=0, help='Which gpu to use if any')
    parser.add_argument('--check_freq', type=int, default=1, help='The frequency of performing evaluation')
    parser.add_argument('--skip_epoch', type=int, default=20, help='Number of beginning epochs that does not perform evaluation (we determine the best validation epoch after 20 epochs, set 0 for no skipping)')
    parser.add_argument('--sh_mode', type=int, default=1, help='0 for print detailed training process; 1 for only printing results')

    # === GNN Related
    parser.add_argument('--total_epochs', type=int, default=50, help='Number of training epochs (50, 100)')
    parser.add_argument('--train_loader_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--eval_loader_size', type=int, default=64, help='Validation and testing batch size')
    parser.add_argument('--gnn_file', type=str, default = '', help='File path to read the model (if there is any)')
    parser.add_argument('--emb_dim', type=int, default=300, help='Embedding dimensions')
    parser.add_argument('--dropout_ratio', type=float, default=0.5, help='Dropout ratio')
    parser.add_argument('--graph_pooling', type=str, default="mean", help='Graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last", help='How the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--gnn_layers', type=int, default=5, help='Number of GNN message passing layers')
    
    # === General Policy Network Related
    parser.add_argument('--actor_d_lr', type=float, default=5e-4, help='Learning rate of the discrete actor')
    parser.add_argument('--actor_c_lr', type=float, default=5e-4, help='Same for continuous actor')
    parser.add_argument('--critic_lr', type=float, default=5e-4, help='Learning rate of critic')
    parser.add_argument('--policy_update_nums', type=int, default=10, help='Maximum number of gradient descent steps to take on policy loss per episode (Early stopping may cause optimizer to take fewer than this.)')
    parser.add_argument('--policy_decay', type=str, default="static", help='Policy learning rate decay (static or down)')
    parser.add_argument('--max_z', type=float, default=1.0, help='The max value of continuous action each step, i.e. prompt content')
    parser.add_argument('--batch_size', type=int, default=8, help='Number of trajectories required for one policy training session')
    parser.add_argument('--minibatch_size', type=int, default=128, help='Number of transitions required for one policy gradient update')
    
    # === H-PPO Related
    parser.add_argument('--init_log_std', type=float, default=0.0, help='Initial log standard deviation of Gaussian Distribution used for sample continuous actions')
    parser.add_argument('--eps_clip_d', type=float, default=0.2, help='The clip ratio when calculate surrogate objective for discrete actor')
    parser.add_argument('--eps_clip_c', type=float, default=0.2, help='Same for continuous')
    parser.add_argument('--coeff_critic', type=float, default=0.5, help='The coefficient of TD-Error for critic')
    parser.add_argument('--coeff_entropy_d', type=float, default=0.01, help='The coefficient of distribution entropy for discrete actor')
    parser.add_argument('--target_kl_d', type=float, default=0.001, help='KL between new and old discrete policy larger than this threshold will trigger early stopping')
    parser.add_argument('--target_kl_c', type=float, default=0.1, help='Same for continuous policy')
    parser.add_argument('--max_norm_grad', type=float, default=0.5, help='Max norm of the gradients')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted of future rewards (Always between 0 and 1, close to 1)')
    parser.add_argument('--lam', type=float, default=0.95, help='Lambda for GAE-Lambda (Always between 0 and 1, close to 1)')

    # === Policy Generalization Related
    parser.add_argument('--ensemble_num', type=int, default=3, help='Number of ensemble actor members, 1 for no ensemble')
    parser.add_argument('--penalty_alpha_d', type=float, default=1e5, help='Discrete penalty coefficient for policy generalization')
    parser.add_argument('--penalty_alpha_c', type=float, default=1e-1, help='Same for continuous policy')

    # === Projection Head (tasknet) Related
    parser.add_argument('--head_layers', type=int, default=1, help='Number of MLP layers of the tasknet (1, 2, 3)')
    parser.add_argument('--tasknet_lr', type=float, default=1e-3, help='Learning rate of tasknet (5e-4, 1e-3, 1.5e-3)')
    parser.add_argument('--tasknet_epochs', type=int, default=1, help='Number of times the projection head is trained per epoch (1, 2 ,3)')
    parser.add_argument('--tasknet_decay', type=str, default="static", help='Tasknet learning rate decay (static or down)')
    parser.add_argument('--tasknet_train_mode', type=int, default=1, help='Whether apply dropout to GNN when updating tasknet (1 for use; 0 for not use)')

    args = parser.parse_args()
    args = parser.parse_args(args=[
    '--gnn_file', 'pretrained_models/infomax.pth',
    '--dataset', 'bbbp',
    '--seed', '0',
    '--eval_loader_size', '64',

    '--head_layers', '3',
    '--total_epochs', '50',
    '--tasknet_epochs', '1', 
    '--tasknet_lr', '1e-3',
    '--tasknet_decay', 'static',
    '--max_z', '0.1',
    '--policy_decay', 'down', 
    '--penalty_alpha_d', '1e5',

    '--train_loader_size', '8',
    '--batch_size', '8',
    '--minibatch_size', '64',

    '--skip_epoch', '20',
    '--check_freq', '1',
    '--sh_mode', '1'
    ])
    if not args.sh_mode:
        logargs(args)
    return args


def set_seed(runseed, device):
    random.seed(runseed)
    np.random.seed(runseed)
    torch.manual_seed(runseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(runseed)
        torch.cuda.manual_seed_all(runseed)
        torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:" + str(device)) if torch.cuda.is_available() else torch.device("cpu")
    
    return device


def data_preprocess(args):
    # Bunch of classification tasks
    if args.dataset == "tox21":
        max_node_num = 132
        num_tasks = 12
    elif args.dataset == "hiv":
        max_node_num = 222
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
        max_node_num = 46
    elif args.dataset == "bace":
        max_node_num = 97
        num_tasks = 1
    elif args.dataset == "bbbp":
        max_node_num = 132
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
        max_node_num = 124
    elif args.dataset == "sider":
        num_tasks = 27
        max_node_num = 492
    elif args.dataset == "clintox":
        max_node_num = 136
        num_tasks = 2
    else:
        raise ValueError("Invalid dataset name.")

    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)
    print("{} | {}-shots | {} | seed:{} | runseed:{} | device:{}".format(
          args.dataset, args.shot_number, args.gnn_file, args.seed, args.runseed, args.device))
    smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
    train_dataset, val_dataset, test_dataset = scaffold_split_fewshot(dataset, smiles_list, null_value=0, number_train=args.shot_number, 
                                                                        frac_valid=0.1, frac_test=0.1, seed=args.seed)

    print("TRAIN #-1: {}  #1: {}  #-1/#1: {:.4f}".format((torch.cat([data.y for data in train_dataset])==-1).sum(),
                                                         (torch.cat([data.y for data in train_dataset])==1).sum(),
                                                         (torch.cat([data.y for data in train_dataset])==-1).sum()/(torch.cat([data.y for data in train_dataset])==1).sum()))
    print("VAL   #-1: {}  #1: {}  #-1/#1: {:.4f}".format((torch.cat([data.y for data in val_dataset])==-1).sum(),
                                                        (torch.cat([data.y for data in val_dataset])==1).sum(),
                                                        (torch.cat([data.y for data in val_dataset])==-1).sum()/(torch.cat([data.y for data in val_dataset])==1).sum()))
    print("TEST  #-1: {}  #1: {}  #-1/#1: {:.4f}".format((torch.cat([data.y for data in test_dataset])==-1).sum(),
                                                        (torch.cat([data.y for data in test_dataset])==1).sum(),
                                                        (torch.cat([data.y for data in test_dataset])==-1).sum()/(torch.cat([data.y for data in test_dataset])==1).sum()))
    
    all_y = torch.cat([d.y for d in train_dataset]).view(len(train_dataset),-1)
    label_dist_list = []
    for task_id in range(num_tasks):
        task_y = all_y[:, task_id]
        label_dist_list.append(( (task_y==-1).sum().item(), (task_y==1).sum().item(), (task_y==0).sum().item() ))
    all_label_ratio_list = [ldl[0]/ldl[1] if ldl[0]*ldl[1] != 0 else None for ldl in label_dist_list]
    valid_label_ratio_list = []
    for alr in all_label_ratio_list:
        if alr is not None:
            valid_label_ratio_list.append(alr)
    mean_label_ratio = sum(valid_label_ratio_list) / len(valid_label_ratio_list) if len(valid_label_ratio_list) else float("inf")
    print("Train Valid Task Ratio: {:.4f} ({}/{})".format(len(valid_label_ratio_list)/num_tasks, len(valid_label_ratio_list), num_tasks))
    print("Train Mean Imbalance Ratio(#-1/#1): {:.4f}".format(mean_label_ratio))
    print("Dataset: {} | Train: {}  Val: {}  Test: {}".format(len(dataset), len(train_dataset), len(val_dataset), len(test_dataset)))
    train_nodes = torch.tensor([d.num_nodes for d in train_dataset]).float()
    val_nodes = torch.tensor([d.num_nodes for d in val_dataset]).float()
    test_nodes = torch.tensor([d.num_nodes for d in test_dataset]).float()
    print("Min Max Mean Median # nodes: {:.0f} {:.0f} {:.2f} {:.2f} | {:.0f} {:.0f} {:.2f} {:.2f} | {:.0f} {:.0f} {:.2f} {:.2f}".format(
        train_nodes.min(), train_nodes.max(), train_nodes.mean(), train_nodes.median(),
        val_nodes.min(), val_nodes.max(), val_nodes.mean(), val_nodes.median(),
        test_nodes.min(), test_nodes.max(), test_nodes.mean(), test_nodes.median()))

    val_dataset = sorted(val_dataset, key=lambda data: data.num_nodes)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_loader_size, shuffle=False, num_workers=args.num_workers)
    test_dataset = sorted(test_dataset, key=lambda data: data.num_nodes)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_loader_size, shuffle=False, num_workers=args.num_workers)
    print("First Train Data:", train_dataset[0])

    max_node_num = max([train_nodes.max(), val_nodes.max(), test_nodes.max()]).long().item()

    return train_dataset, val_loader, test_loader, max_node_num, num_tasks


def train_loaders_process(args, train_dataset: MoleculeDataset, overlap_ratio=0.2, train_loader_batchsize="total"):
    torch.manual_seed(args.seed)
    train_dataset = train_dataset.shuffle()
    torch.manual_seed(args.runseed)
    if args.ensemble_num > 1:
        part_num = len(train_dataset) // args.ensemble_num
        idx_list = [torch.arange(i*part_num, (i+1)*part_num if i<args.ensemble_num-1 else len(train_dataset)) for i in range(args.ensemble_num)]
        train_dataset_list, imb_ratios = [], []
        for i, idx in enumerate(idx_list):
            for j in range(args.ensemble_num):
                if i == j:
                    continue
                idx = torch.cat([idx, idx_list[j][-int(part_num*overlap_ratio):]], dim=0) 
            ensemble_dataset = train_dataset[idx]
            train_dataset_list.append(ensemble_dataset)
            labels = torch.cat([ed.y for ed in ensemble_dataset])
            imb_ratios.append(((labels==-1).sum() / (labels==1).sum()).item())
        ens_loaders = [DataLoader(td, batch_size=args.train_loader_size, shuffle=True, num_workers=args.num_workers) for td in train_dataset_list]
        if not args.sh_mode:
            print("\nPolicy Train Loaders: {}".format([len(tdl) for tdl in train_dataset_list]))
            print("Imbalance Rates(#-1/#1): {}".format([round(ir, 2) for ir in imb_ratios]))
    else:
        ens_loaders = [DataLoader(train_dataset, batch_size=args.train_loader_size, shuffle=True, num_workers=args.num_workers)]

    if train_loader_batchsize == "total":
        batch_size = min(len(train_dataset), 128)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.train_loader_size, shuffle=True, num_workers=args.num_workers)

    return ens_loaders, train_loader


def model_setup(args, num_tasks, max_node_num, device):
    gnn = PromptedGNN(args.gnn_layers, args.emb_dim, args.JK, args.dropout_ratio, args.gnn_type, prompt_type="add").to(device)
    if not args.gnn_file == "":
        if not args.sh_mode:
            print("Loading pretrained gnn weights...")
        gnn.load_state_dict(torch.load(args.gnn_file, map_location=device))

    if not args.sh_mode:
        print("Initializing tasknet linears...")
    tasknet = GNNBasedNet(args.emb_dim, args.gnn_layers, args.JK, args.graph_pooling, args.head_layers, num_tasks).to(device)
    tasknet_optim = optim.Adam(tasknet.parameters(), lr=args.tasknet_lr)

    if not args.sh_mode:
        print("Initializing policy...")
    policy = H_PPO(gnn, args.policy_d_type, args.ensemble_num, args.penalty_alpha_d, args.penalty_alpha_c, args.emb_dim, max_node_num,
                   args.max_action_con, args.init_log_std, args.eps_clip_d, args.eps_clip_c, args.coeff_critic,
                   args.coeff_entropy_d, args.coeff_entropy_c, args.max_norm_grad, device)
    policy_optims = []
    for i in range(args.ensemble_num):
        param_group = []
        param_group.append({"name": "actor_d", "params": policy.actor_ds[i].parameters(), "lr": args.actor_d_lr})
        param_group.append({"name": "actor_c", "params": policy.actor_cs[i].parameters(), "lr": args.actor_c_lr})
        param_group.append({"name": "critic", "params": policy.critic.parameters(), "lr": args.critic_lr})
        policy_optims.append(optim.Adam(param_group, eps=1e-5))  # [Trick] set Adam epsilon=1e-5
    
    return gnn, tasknet, policy, tasknet_optim, policy_optims


def main():
    print('PID[{}]'.format(os.getpid()))
    torch.set_default_dtype(torch.float32)
    np.set_printoptions(precision=32)
    
    auc_list = []
    for i in range(1, 6):
        args = set_args()
        args.runseed = i
        device = set_seed(args.runseed, args.device)
        if args.sh_mode:
            print("=" * 120)
            print("total epoch:{} | tasknet epoch:{} | head layer:{} | tasknet lr:{} | tasknet lr decay:{} | policy lr decay:{} |\n"
                  "ensemble number:{} | max_z:{} | penalty(discrete):{} | penalty(continuous):{}".format(
                   args.total_epochs, args.head_layers, args.tasknet_epochs, args.tasknet_lr, args.tasknet_decay, args.policy_decay,
                   args.ensemble_num, args.max_z, args.penalty_alpha_d, args.penalty_alpha_c))
            print("=" * 120)
        train_dataset, val_loader, test_loader, max_node_num, num_tasks = data_preprocess(args)
        gnn, tasknet, policy, tasknet_optim, policy_optims = model_setup(args, num_tasks, max_node_num, device)
        ens_loaders, train_loader = train_loaders_process(args, train_dataset)

        best_infos = None
        aucs = (-1,-1,-1)
        for warmup_epoch in range(1, args.policy_warmup_epochs+1):
            policy, tasknet, aucs, best_infos = train_policy(args, warmup_epoch, True, gnn, tasknet, tasknet_optim, policy, policy_optims,
                                                            ens_loaders, train_loader, val_loader, test_loader, aucs, best_infos, device)
    
        # ======== DYNAMIC TRAINING ======= #
        for policy_epoch in range(1, args.policy_epochs+1):
            policy, tasknet, aucs, best_infos, r_lists = train_policy(args, policy_epoch, False, gnn, tasknet, tasknet_optim, policy, policy_optims,
                                                            ens_loaders, train_loader, val_loader, test_loader, aucs, best_infos, device)
               
        print("====== BEST INFOS: EPOCH{} - TRAIN AUC {:.2f} | VAL AUC {:.2f} | TEST AUC {:.2f}\n\n".format(*best_infos))
        auc_list.append(best_infos[-1])

    print("================ AVG_AUC {:.2f} ================".format(sum(auc_list)/len(auc_list)))

        
if __name__ == "__main__":
    main()
