import argparse, random, os, gc, pickle
import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

from torch_geometric.loader import DataLoader
import torch
import torch.optim as optim
import numpy as np


from model import GIN
from agent import H_PPO
from base import *
from util import logargs


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--subgraph_file', type=str, default='', help='Root directory of split subgraphs')
    parser.add_argument('--shot_number', type=float, default=50, help='set shot_number > 1 as number of samples used for training; set 0 < shot_number <= 1 as ratio of entire training samples used for training')
    parser.add_argument('--runseed', type=int, default=1, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--device', type=int, default=0, help='Which gpu to use if any')
    parser.add_argument('--check_freq', type=int, default=2, help='The frequency of performing evaluation')
    parser.add_argument('--skip_epoch', type=int, default=20, help='Number of beginning epochs that does not perform evaluation (we determine the best validation epoch after 20 epochs, set 0 for no skipping)')
    parser.add_argument('--sh_mode', type=int, default=1, help='0 for printing detailed training process; 1 for only printing results')

    # === GNN Related
    parser.add_argument('--total_epochs', type=int, default=50, help='Number of training epochs (50, 100)')
    parser.add_argument('--train_loader_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--eval_loader_size', type=int, default=64, help='Validation and testing batch size')
    parser.add_argument('--gnn_file', type=str, default = '', help='File path to read the model (if there is any)')
    parser.add_argument('--hid_dim', type=int, default=128)
    parser.add_argument('--svd_dim', type=int, default=100, help='use SVD to reduce the initial node feature dimension')
    parser.add_argument('--dropout_ratio', type=float, default=0.5, help='Dropout ratio')
    parser.add_argument('--graph_pooling', type=str, default="mean", help='Graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last", help='How the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_layers', type=int, default=2, help='Number of GNN message passing layers')
    
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
    parser.add_argument('--warmup_epochs', type=int, default=0, help='Only training the tasknet when warming up')
    parser.add_argument('--head_layers', type=int, default=1, help='Number of MLP layers of the tasknet (1, 2, 3)')
    parser.add_argument('--tasknet_lr', type=float, default=1e-3, help='Learning rate of tasknet (5e-4, 1e-3, 1.5e-3)')
    parser.add_argument('--tasknet_epochs', type=int, default=1, help='Number of times the projection head is trained per epoch (1, 2 ,3)')
    parser.add_argument('--tasknet_decay', type=str, default="static", help='Tasknet learning rate decay (static or down)')
    parser.add_argument('--tasknet_train_mode', type=int, default=1, help='Whether apply dropout to GNN when updating tasknet (1 for use; 0 for not use)')

    args = parser.parse_args()
    # args = parser.parse_args(args=[
    # '--dataset', 'Cora',
    # '--subgraph_file', 'Cora_4hop_svd100_10shots_nnl5-150_seed0_split.dat',
    # '--gnn_file', 'Gprompt_Cora.pth',

    # '--warmup_epochs', '0',
    # '--train_loader_size', '8',
    # '--batch_size', '8',
    # '--minibatch_size', '64',

    # '--total_epochs', '50',
    # '--head_layers', '3',
    # '--tasknet_epochs', '3', 
    # '--tasknet_lr', '1e-3',
    # '--tasknet_decay', 'static',
    # '--policy_decay', 'static', 
    # '--max_z', '0.5',
    # '--penalty_alpha_d', '1e0',
    # '--tasknet_train_mode', '1',

    # '--device', '7',
    # '--sh_mode', '0'
    # ])
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


def load_subgraph(dataset, subgraph_file, eval_loader_size, warmup_epochs):
    if dataset == "Cora":
        num_classes = 7
    elif dataset == "CiteSeer":
        num_classes = 6
    elif dataset == "PubMed":
        num_classes = 3
    elif dataset == "Computers":
        num_classes = 10
    elif dataset == "Photo":
        num_classes = 8

    split_data = pickle.load(open('subgraph/split_data/'+subgraph_file, "br"))
    train_dataset, val_dataset, test_dataset = split_data["train_dataset"], split_data["val_dataset"], split_data["test_dataset"]
    for d in train_dataset:
        d.x = d.x.to(torch.float64)
    for d in val_dataset:
        d.x = d.x.to(torch.float64)
    for d in test_dataset:
        d.x = d.x.to(torch.float64)
    train_node_num = torch.tensor([d.num_nodes for d in train_dataset])
    val_node_num = torch.tensor([d.num_nodes for d in val_dataset])
    test_node_num = torch.tensor([d.num_nodes for d in test_dataset])
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

    warmup_train_loader = None
    if warmup_epochs > 0:
        warmup_train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=eval_loader_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=eval_loader_size, shuffle=False)

    return num_classes, train_dataset, warmup_train_loader, val_loader, test_loader, \
           max([max(train_node_num), max(val_node_num), max(test_node_num)])


def model_setup(args, num_classes, max_node_num, device):
    gnn = GIN(args.svd_dim, args.hid_dim, args.hid_dim, args.gnn_layers, args.JK, args.dropout_ratio, args.graph_pooling).to(device)
    if args.gnn_file != "" and args.gnn_file is not None:
        if not args.sh_mode:
            print("Loading pretrained gnn weights...")
        gnn.load_state_dict(torch.load('pretrained_models/'+args.gnn_file, map_location=device))

    if not args.sh_mode:
        print("Initializing tasknet linears...")
    tasknet = torch.nn.Sequential()
    for _ in range(args.head_layers - 1):
        tasknet.append(torch.nn.Linear(args.hid_dim, args.hid_dim))
        tasknet[-1].reset_parameters()
        tasknet.append(torch.nn.ReLU())
    tasknet.append(torch.nn.Linear(args.hid_dim, num_classes))
    tasknet[-1].reset_parameters()
    tasknet.to(device)
    tasknet_optim = optim.Adam(tasknet.parameters(), lr=args.tasknet_lr, weight_decay=5e-4)

    if not args.sh_mode:
        print("Initializing policy...")
    policy = H_PPO(gnn, args.ensemble_num, args.penalty_alpha_d, args.penalty_alpha_c, args.hid_dim, args.svd_dim, max_node_num,
                   args.max_z, args.init_log_std, args.eps_clip_d, args.eps_clip_c, args.coeff_critic, args.coeff_entropy_d, args.max_norm_grad, device)
    policy_optims = []
    for i in range(args.ensemble_num):
        param_group = []
        param_group.append({"name": "actor_d", "params": policy.actor_ds[i].parameters(), "lr": args.actor_d_lr})
        param_group.append({"name": "actor_c", "params": policy.actor_cs[i].parameters(), "lr": args.actor_c_lr})
        param_group.append({"name": "critic", "params": policy.critic.parameters(), "lr": args.critic_lr})
        policy_optims.append(optim.Adam(param_group, eps=1e-5))  # [Trick] set Adam epsilon=1e-5
        policy.actor_cs[i].log_std = policy.actor_cs[i].log_std.to(device)

    return gnn, tasknet, policy, tasknet_optim, policy_optims


def train_loaders_process(args, train_dataset, overlap_ratio=0.2):
    if args.ensemble_num > 1:
        random.shuffle(train_dataset)
        part_num = len(train_dataset) // args.ensemble_num
        idx_list = [torch.arange(i*part_num, (i+1)*part_num if i<args.ensemble_num-1 else len(train_dataset)) for i in range(args.ensemble_num)]
        train_dataset_list = []
        for i, idx in enumerate(idx_list):
            for j in range(args.ensemble_num):
                if i == j:
                    continue
                idx = torch.cat([idx, idx_list[j][-int(part_num*overlap_ratio):]], dim=0)
            ensemble_dataset_list = []
            for id in idx:
                ensemble_dataset_list.append(train_dataset[id])
            train_dataset_list.append(ensemble_dataset_list)
        ens_loaders = [DataLoader(td, batch_size=args.train_loader_size, shuffle=True) for td in train_dataset_list]
        if not args.sh_mode:
            print("")
            for i in range(args.ensemble_num):
                print("Policy Train Loaders{}: {} [{}]".format(
                    i+1, torch.bincount(torch.stack([tdl.y for tdl in train_dataset_list[i]])), len(train_dataset_list[i])))
            
    else:
        ens_loaders = [DataLoader(train_dataset, batch_size=args.train_loader_size, shuffle=True)]

    batch_size = min(len(train_dataset), 128)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    return ens_loaders, train_loader


def main():
    print('PID[{}]'.format(os.getpid()))
    torch.set_default_dtype(torch.float64)
    np.set_printoptions(precision=64)
    
    acc_list, f1_list = [], []
    # run 5 random seeds
    for i in range(1, 6):
        args = set_args()
        args.runseed = i
        device = set_seed(args.runseed, args.device)
        if args.sh_mode:
            print("=" * 120)
            print("total epoch:{} | head layer:{} | tasknet epoch:{} |tasknet lr:{} | tasknet lr decay:{} | policy lr decay:{} |\n"
                  "ensemble number:{} | max_z:{} | penalty discrete:{} | penalty continuous:{}".format(
                   args.total_epochs, args.head_layers, args.tasknet_epochs, args.tasknet_lr, args.tasknet_decay, args.policy_decay,
                   args.ensemble_num, args.max_z, args.penalty_alpha_d, args.penalty_alpha_c))
            print("=" * 120)
            print('{} | subgraph file:{} | gnn file:{} | runseed:{} | device:{}'.format(
                  args.dataset, args.subgraph_file, args.gnn_file, args.runseed, args.device))
        num_classes, train_dataset, warmup_train_loader, val_loader, test_loader, max_node_num = \
            load_subgraph(args.dataset, args.subgraph_file, args.eval_loader_size, args.warmup_epochs)
        gnn, tasknet, policy, tasknet_optim, policy_optims = model_setup(args, num_classes, max_node_num, device)

        # ====== Warm Up ====== #
        best_infos = None
        accs, f1s = (-1, -1, -1), (-1,-1,-1)
        if args.warmup_epochs > 0:
            tasknet, accs, f1s = tasknet_warmup(args, gnn, tasknet, tasknet_optim, warmup_train_loader, val_loader, test_loader, device)
            del warmup_train_loader; gc.collect()
        
        ens_loaders, train_loader = train_loaders_process(args, train_dataset)
        # ====== Alternating Training ====== #
        for epoch in range(1, args.total_epochs + 1):
            policy, tasknet, accs, f1s, best_infos = train_policy(args, epoch, gnn, tasknet, tasknet_optim, policy, policy_optims,
                                                                  ens_loaders, train_loader, val_loader, test_loader, accs, f1s, best_infos, device)
               
        print("====== BEST INFOS: EPOCH{} - TRAIN ACC {:.2f} | VAL ACC {:.2f} | TEST ACC {:.2f} | TRAIN F1 {:.2f} | VAL F1 {:.2f} | TEST F1 {:.2f}\n\n".format(*best_infos))
        acc_list.append(best_infos[3])
        f1_list.append(best_infos[-1])

    print("======= {:.2f} | {:.2f} =======".format(sum(acc_list)/len(acc_list), sum(f1_list)/len(f1_list)))

        
if __name__ == "__main__":
    main()
