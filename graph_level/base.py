from typing import List
import copy, time, gc

from torch_geometric.data import Batch
from torch_scatter import scatter
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score

from model import PromptedGNN, GNNBasedNet
from agent import H_PPO
from reward import compute_adv_ret, reshape_reward, reward_reshape, RewardScaling, Normalization
from util import get_time

criterion = nn.BCEWithLogitsLoss(reduction="none")

def grad_norm_split(gnn: PromptedGNN, tasknet: GNNBasedNet, batch_data: Batch, gn_split="mean", top_ratio=1.0, return_type="Batch", return_info=True):

    if not return_info and gn_split == "top" and top_ratio == 1:
        if return_type == "list":
            return None, batch_data.to_data_list()
        elif return_type == "Batch":
            return None, batch_data

    device = batch_data.x.device
    task_loss, _, _ = tasknet_loss(gnn, tasknet, batch_data, prompt=None, require_grad=True, keep_loss_dim=True)
    grads = [torch.autograd.grad(tl, tasknet.parameters(), retain_graph=True) for tl in task_loss]
    grad_norm = torch.tensor([torch.norm(torch.cat([g.view(-1) for g in grad])) for grad in grads])
    
    greater_grad_norm_idx, less_grad_norm_idx = None, None
    if gn_split == "top" and top_ratio < 1:
        greater_grad_norm_idx = grad_norm.sort(descending=True)[1][:int(top_ratio * grad_norm.size(0))]
        less_grad_norm_idx = grad_norm.sort(descending=True)[1][int(top_ratio * grad_norm.size(0)):]
    if gn_split == "mean":
        greater_grad_norm_idx = torch.nonzero(grad_norm > grad_norm.mean()).squeeze(-1)
        less_grad_norm_idx = torch.nonzero(grad_norm <= grad_norm.mean()).squeeze(-1)
    if gn_split == "top" and top_ratio == 1:
        greater_grad_norm_idx = torch.arange(0, grad_norm.size(0))
    pr_idx = greater_grad_norm_idx.sort(descending=True)[0].tolist()

    og_data_list = batch_data.to_data_list()
    pr_data_list = []
    if gn_split == "top" and top_ratio == 1:
        pr_data_list = og_data_list
        og_data_list = []
    else:
        if len(pr_idx):
            for idx in pr_idx:
                pr_data_list.append(og_data_list.pop(idx))

    if return_type == "Batch":
        og_data, pr_data = None, None
        if len(og_data_list):
            og_data = Batch.from_data_list(og_data_list)
        if len(pr_data_list):
            pr_data = Batch.from_data_list(pr_data_list)
        
        if return_info:
            return og_data.to(device) if og_data is not None else og_data, pr_data.to(device) if pr_data is not None else pr_data,\
                grad_norm[greater_grad_norm_idx].tolist() if greater_grad_norm_idx is not None else [],\
                grad_norm[less_grad_norm_idx].tolist() if less_grad_norm_idx is not None else [],\
                grad_norm.tolist()
        else:
            return og_data.to(device) if og_data is not None else og_data, pr_data.to(device) if pr_data is not None else pr_data
    
    else:
        return og_data_list, pr_data_list


def attach_prompt(args, policy: H_PPO, data, data_type, train_jsd, gnn=None, tasknet=None, compute_reward=0, compute_jsd=1, compute_prr=0):
    device = data.x.device

    graph_num = data.num_graphs
    nodes_per_graph = scatter(torch.ones_like(data.batch), data.batch, reduce='add')
    policy.prompt_slices = [torch.zeros(nodes, args.emb_dim).to(device) for nodes in nodes_per_graph]
    policy.step_state = torch.zeros(graph_num, policy.max_num_nodes, args.emb_dim).to(device)
    base_mask = torch.zeros(graph_num, policy.max_num_nodes).to(device)
    base_idx = torch.arange(0, policy.max_num_nodes).to(device)
    policy.batch_mask = torch.where(base_idx < nodes_per_graph.unsqueeze(-1), base_mask, torch.ones_like(base_mask))

    if data_type != "train":
        if args.jsd_coeff <= 10:
            budget_acc_jsd = [nodes.item() * train_jsd * args.jsd_coeff for nodes in nodes_per_graph]
        # budget_acc_jsd = [nodes.item() * train_jsd * args.jsd_coeff for nodes in nodes_per_graph]
        total_prompt_step_num = 0
    if compute_jsd:
        jsd_list = [0] * graph_num
    if compute_reward:
        init_r = generate_reward(gnn, tasknet, args.reward_type, data)
        r_list = [copy.deepcopy([]) for _ in range(graph_num)]
        for rl, ir in zip(r_list, init_r):
            rl.append(ir)
    
    truncate_flag = torch.zeros(graph_num)
    batch_max_step = nodes_per_graph.max().item()
    for step in range(batch_max_step):
        if truncate_flag.sum() == graph_num:
            break

        if compute_jsd:
            batch_jsd = policy.eval_prompt(args, data, nodes_per_graph, step, truncate_flag, args.state_norm, compute_jsd=1)
            jsd_list = [jsd + b_jsd for jsd, b_jsd in zip(jsd_list, batch_jsd)]
        else:
            policy.eval_prompt(args, data, nodes_per_graph, step, truncate_flag, args.state_norm)

        if compute_reward:
            rs = generate_reward(gnn, tasknet, args.reward_type, data, torch.cat(policy.prompt_slices))
            for i in range(graph_num):
                if step < nodes_per_graph[i] and truncate_flag[i] == 0:
                    r_list[i].append(rs[i])
        
        if data_type != "train":
            total_prompt_step_num += (~truncate_flag.bool() & (step<nodes_per_graph).cpu()).sum()
            if args.jsd_coeff <= 10:
                jsd_truncate_idx = torch.where(torch.tensor(jsd_list) > torch.tensor(budget_acc_jsd))[0]
                truncate_flag[jsd_truncate_idx] = 1

    reward_list, total_jsd_list, pr_ratio_list = None, None, None
    if compute_reward:
        reward_list = [reward_reshape(torch.stack(r).cpu().numpy(), reward_clip=args.reward_clip) for r in r_list]
    if compute_prr:
        pr_ratio_list = [torch.any(p!=0, dim=-1).sum().item()/p.size(0) for p in policy.prompt_slices]
    
    if data_type == "train":
        if compute_jsd:
            total_jsd_list = (torch.tensor(jsd_list) / nodes_per_graph.cpu()).tolist()
        return torch.cat(policy.prompt_slices), reward_list, total_jsd_list, pr_ratio_list
    else:
        return torch.cat(policy.prompt_slices), reward_list, total_prompt_step_num.item(), pr_ratio_list


def attach_random_prompt(args, data, device):
    nodes_per_graph = scatter(torch.ones_like(data.batch), data.batch, reduce='add')
    random_prompt_slices = [torch.zeros(nodes, args.emb_dim) for nodes in nodes_per_graph]

    batch_max_step = nodes_per_graph.max().item()
    for step in range(batch_max_step):
        batch_action_d = torch.zeros_like(nodes_per_graph, dtype=torch.int32)
        for i in range(nodes_per_graph.shape[0]):
            if step < nodes_per_graph[i]:
                batch_action_d[i] = torch.randint(0, nodes_per_graph[i].item(), (1,), dtype=torch.int32)
        means = torch.empty((len(nodes_per_graph), args.emb_dim)).uniform_(-args.max_action_con, args.max_action_con)
        batch_action_c = means
        # update prompt
        for i in range(len(random_prompt_slices)):
            if step < nodes_per_graph[i]:
                random_prompt_slices[i][batch_action_d[i]] += batch_action_c[i]

    pr_ratio_list = [torch.any(p!=0, dim=-1).sum().item()/p.size(0) for p in random_prompt_slices]
    return torch.cat(random_prompt_slices).to(device), pr_ratio_list


def compute_auc(score, y, info="", first=False):
    if score.size(0) == 0:
        return -1
     
    auc_list = []
    # for each task
    for i in range(y.shape[1]):
        is_valid = y[:, i]**2 > 0
        label = ((y[is_valid, i] + 1) / 2).long()
        # AUC is only defined when there is at least one positive data.
        if torch.sum(label==0) > 0 and torch.sum(label==1) > 0:
            auc_list.append(roc_auc_score(label, score[is_valid, i]))
        else:
            # print("\t({}) targets of task {} are all {}!".format(info, i, 0 if torch.sum(label==0) == 0 else 1))
            pass
    
    if info == "Total_AUC" and len(auc_list) < y.shape[1] and first:
        print("\tSome targets are missing! Missing ratio:{:.6f}".format(1 - float(len(auc_list))/y.shape[1]))

    return sum(auc_list)/len(auc_list) * 100 if len(auc_list)>0 else -1


def tasknet_loss(gnn: PromptedGNN, tasknet: GNNBasedNet, data, prompt, require_grad=True, keep_loss_dim=False, policy_gnn_update=False):
    if policy_gnn_update:
        gnn.train()
    
    with torch.no_grad():
        node_emb = gnn(data.x, data.edge_index, data.edge_attr, prompt)
    if require_grad:
        logit = tasknet(node_emb, data.batch)  # (batchsize, task_num)
    else:
        with torch.no_grad():
            logit = tasknet(node_emb, data.batch)
    # y = data.y.view(logit.shape).type(torch.float64)
    y = data.y.view(logit.shape).type(data.x.dtype)
    is_valid = y**2 > 0
    if torch.sum(is_valid) == 0:
        return None

    task_loss = criterion(logit, (y+1)/2)  # {-1,1} -> {0,1}
    task_loss = torch.where(is_valid, task_loss, torch.zeros(task_loss.shape).to(task_loss.dtype).to(task_loss.device))
    if keep_loss_dim:
        task_loss = torch.sum(task_loss, dim=-1) / torch.sum(is_valid, dim=-1)
    else:
        task_loss = torch.sum(task_loss) / torch.sum(is_valid)

    if policy_gnn_update:
        gnn.eval()

    return task_loss, logit.detach().cpu(), y.cpu()


def generate_reward(gnn: PromptedGNN, tasknet: GNNBasedNet, reward_type, data, prompt=None):
    with torch.no_grad():
        node_emb = gnn(data.x, data.edge_index, data.edge_attr, prompt)
        logit = tasknet(node_emb, data.batch)
    y = data.y.view(logit.shape).type(node_emb.dtype)
    is_valid = y**2 > 0
    if torch.sum(is_valid) == 0:
        return None
    
    if reward_type == "bce":
        reward = criterion(logit, (y+1)/2)
    elif reward_type == "hinge":
        reward = criterion_hinge(logit, (y+1)/2)
    elif reward_type == "focal":
        reward = criterion_focal(logit, (y+1)/2)
    else:
        raise ValueError("Invalid reward type - {}!".format(reward_type))
    reward = torch.sum(reward, dim=-1) / torch.sum(is_valid, dim=-1)
    
    return reward


def train_tasknet(args, gnn: PromptedGNN, tasknet: GNNBasedNet, optimizer, train_loader, val_loader, test_loader, device):
    if args.tasknet_warmup_epochs == 0:
        return tasknet, (-1,-1,-1)
    print("\n========= Train Tasknet ========= {}".format(get_time()))
    for epoch in range(1, args.tasknet_warmup_epochs+1):
        gnn.train(); tasknet.train()
        tasknet_epoch_loss = 0
        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            loss, _, _ = tasknet_loss(gnn, tasknet, batch_data, prompt=None)
            if loss is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tasknet_epoch_loss += loss.detach().item()
        tasknet_epoch_loss /= len(train_loader)
        
        _, train_auc = evaluate_tasknet(gnn, tasknet, train_loader, epoch==1, device)
        val_loss, val_auc = evaluate_tasknet(gnn, tasknet, val_loader, epoch==1, device)
        test_loss, test_auc = evaluate_tasknet(gnn, tasknet, test_loader, epoch==1, device)
        print("[{}/{}] TRAIN LOSS:{:.4f} AUC:{:.2f} | VAL LOSS:{:.4f} AUC:{:.2f} | TEST LOSS:{:.4f} AUC:{:.2f}".format(
                epoch, args.tasknet_warmup_epochs, tasknet_epoch_loss, train_auc, val_loss, val_auc, test_loss, test_auc))
        
    return tasknet, (train_auc, val_auc, test_auc)
    

def evaluate_tasknet(gnn: PromptedGNN, tasknet: GNNBasedNet, loader, first, device):
    gnn.eval(); tasknet.eval()
    epoch_loss = 0
    logits, ys = [], []
    for batch_data in loader:
        batch_data = batch_data.to(device)
        loss_return = tasknet_loss(gnn, tasknet, batch_data, prompt=None, require_grad=False)
        if loss_return is not None:
            loss, logit, y = loss_return
            logits.append(logit); ys.append(y)
            epoch_loss += loss.item()
        else:
            continue
    epoch_loss /= len(loader)
    logits = torch.cat(logits)
    ys = torch.cat(ys)
    auc = compute_auc(logits, ys, info="Total_AUC", first=first)

    return epoch_loss, auc


def train_policy(args, policy_epoch, warmup, gnn: PromptedGNN, tasknet: GNNBasedNet, tasknet_optim, policy: H_PPO, policy_optims: List[optim.Adam],
                 ens_loaders, train_loader, val_loader, test_loader, aucs, best_infos, device):
    
    old_train_auc, old_val_auc, old_test_auc = aucs
    if best_infos is None:
        best_epoch, best_train_auc, best_val_auc, best_test_auc = (-1, -1, -1, -1)
    else:
        best_epoch, best_train_auc, best_val_auc, best_test_auc = best_infos
    # ===== Annealing the learning rate ===== #
    if not warmup:
        descend_linear_frac = 1.0 - (policy_epoch - 1) / (args.policy_epochs)
        ascend_linear_frac = policy_epoch / args.policy_epochs
        policy_linear_frac, tasknet_linear_frac = 1.0, 1.0
        policy_symble, tasknet_symble = "->", "->"

        if args.policy_linear == "up":
            # policy_linear_frac = 0.1 + 0.9 * ascend_linear_frac  # start from 0.1
            policy_linear_frac = ascend_linear_frac
            policy_symble = "↑"
        elif args.policy_linear == "down":
            policy_linear_frac = descend_linear_frac
            policy_symble = "↓"
        for i in range(args.ensemble_num):
            for param_group in policy_optims[i].param_groups:
                if param_group["name"] == "actor_d":
                    param_group["lr"] = policy_linear_frac * args.actor_d_lr
                elif param_group["name"] == "actor_c":
                    param_group["lr"] = policy_linear_frac * args.actor_c_lr
                elif param_group["name"] == "critic":
                    param_group["lr"] = policy_linear_frac * args.critic_lr

        if args.tasknet_linear == "up":
            tasknet_linear_frac = ascend_linear_frac
            tasknet_symble = "↑"
        elif args.tasknet_linear == "down":
            tasknet_linear_frac = descend_linear_frac
            tasknet_symble = "↓"
        for param_group in tasknet_optim.param_groups:
            param_group["lr"] = tasknet_linear_frac * args.tasknet_lr_scale * args.tasknet_lr

        policy.coeff_ent_d = args.coeff_entropy_d * descend_linear_frac

        for i in range(args.ensemble_num):
            new_log_std = args.init_log_std + (-5 - args.init_log_std) * (policy_epoch - 1) / args.policy_epochs
            policy.actor_cs[i].log_std = (torch.ones(args.emb_dim) * new_log_std).to(device)

        if not args.sh_mode:
            print("\n========= EPOCH {}/{} ========= policy_lr*{:.2f} tasknet_lr*{:.2f} (policy:{} tasknet:{} ent_d:{}) [{}]".format(
                policy_epoch, args.policy_epochs, policy_linear_frac, tasknet_linear_frac * args.tasknet_lr_scale,
                policy_symble, tasknet_symble, "↓", get_time()))
    else:
        if not args.sh_mode:
            print("\n========= WARMUP EPOCH {}/{} ========= [{}]".format(policy_epoch, args.policy_warmup_epochs, get_time()))
    
    # ===== Collect transitions and update the policy ===== #
    tasknet.eval(); policy.train_or_eval(mode='train')
    reward_transform = None
    if args.reward_norm_type == "scale":
        reward_transform = RewardScaling(shape=1, gamma=args.gamma)
    elif args.reward_norm_type == "norm":
        reward_transform = Normalization(shape=1)
    
    if args.random_d * args.random_c == 1:
        pass
    # ===== Train each ensemble actor member ===== #
    else:
        for a_idx in range(args.ensemble_num):
            gnn.eval()

            batch_data_list = []
            for batch_idx, batch_data in enumerate(ens_loaders[a_idx]):
                # Filter out graphs that would not be prompted
                batch_data = batch_data.to(device)
                tasknet_optim.zero_grad()
                _, data_list = grad_norm_split(gnn, tasknet, batch_data, args.gn_split, args.top_ratio, "list", return_info=False)
                batch_data_list.extend(data_list)
                # Collect enough amount of graphs or it's the last batch of data
                if len(batch_data_list) < args.batch_size and batch_idx < len(ens_loaders[a_idx])-1 or len(batch_data_list) == 0:
                    continue
                batch_data = Batch.from_data_list(batch_data_list).to(device)
                batch_data_list = []

                nodes_per_graph = scatter(torch.ones_like(batch_data.batch), batch_data.batch, reduce='add')
                with torch.no_grad():
                    init_r = generate_reward(gnn, tasknet, args.reward_type, batch_data)
                assert len(init_r) == len(nodes_per_graph)

                graph_num = batch_data.num_graphs
                batch_max_step = nodes_per_graph.max().item()
                state = torch.zeros([graph_num, batch_max_step, policy.max_num_nodes, args.emb_dim]).to(device)
                action_d = torch.zeros([graph_num, batch_max_step]).long().to(device)
                action_c = torch.zeros([graph_num, batch_max_step, args.emb_dim]).to(device)
                logprob_d = torch.zeros([graph_num, batch_max_step]).to(device)
                logprob_c = torch.zeros([graph_num, batch_max_step]).to(device)
                reward = torch.zeros([graph_num, batch_max_step+1]).to(device)
                reward[:, 0] = init_r
                done = torch.zeros([graph_num, batch_max_step]).to(device)
                valid_mask = (torch.arange(batch_max_step).expand(graph_num,-1).to(device)) < nodes_per_graph.unsqueeze(1)  # (num_graphs, batch_max_step)

                # ==== Collect a batch of episodes ==== #
                policy.prompt_slices = [torch.zeros(nodes, args.emb_dim).to(device) for nodes in nodes_per_graph]
                policy.step_state = torch.zeros(graph_num, policy.max_num_nodes, args.emb_dim).to(device)
                base_mask = torch.zeros(graph_num, policy.max_num_nodes).to(device)
                base_idx = torch.arange(0, policy.max_num_nodes).to(device)
                policy.batch_mask = torch.where(base_idx < nodes_per_graph.unsqueeze(-1), base_mask, torch.ones_like(base_mask))
                for step in range(batch_max_step):
                    done[nodes_per_graph == step + 1, step] = 1.0
                    # Attach a new prompt vector on a specific node for each graph
                    s, a_d, a_c, lp_d, lp_c = policy.train_prompt(args, a_idx, batch_data, nodes_per_graph, step, args.state_norm)
                    state[:, step] = s
                    action_d[:, step] = a_d
                    action_c[:, step] = a_c
                    logprob_d[:, step] = lp_d
                    logprob_c[:, step] = lp_c
                    # Execute downstream task with prompted graph to recieve reward `r`
                    r = generate_reward(gnn, tasknet, args.reward_type, batch_data, torch.cat(policy.prompt_slices))
                    reward[:, step+1] = r

                # Mask invalid element and compute rewards, advantages, returns
                next_state = torch.zeros_like(state)
                for i in range(graph_num):
                    if nodes_per_graph[i] > 1:
                        valid_s = state[i, :nodes_per_graph[i]]
                        next_state[i, :nodes_per_graph[i]-1] = valid_s[1:]
                        next_state[i, nodes_per_graph[i]-1] = valid_s[-1]
                    else:
                        next_state[i, 0] = state[i, 0]
                reward = reshape_reward(reward, nodes_per_graph, args.reward_clip)
                state = state[valid_mask]; next_state = next_state[valid_mask]; done = done[valid_mask]
                advantage, approx_return, approx_return0, scaled_reward = \
                    compute_adv_ret(args, policy.critic, state, reward, next_state, done, nodes_per_graph.tolist(), reward_transform)
                action_d = action_d[valid_mask]; action_c = action_c[valid_mask]
                logprob_d = logprob_d[valid_mask]; logprob_c = logprob_c[valid_mask]
                if state.size(0) > args.minibatch_size:
                    experience = (state, action_d, logprob_d, action_c, logprob_c, advantage, approx_return)
                    policy.train_policy(args, a_idx, experience, nodes_per_graph, policy_optims[a_idx], approx_return0, scaled_reward, batch_idx+1, len(ens_loaders[a_idx]))
                    # del experience
                    # gc.collect(); torch.cuda.empty_cache()
                else:
                    if not args.sh_mode:
                        print("No enough trainsitions!")
                # del state, action_d, action_c, logprob_d, logprob_c, reward, done, valid_mask
                # gc.collect(); torch.cuda.empty_cache()
            
    
    # ===== Update tasknet according to linked policy ===== #
    if not warmup:
        if policy_epoch % args.tasknet_freq == 0:
            gnn.eval(); policy.train_or_eval('eval'); tasknet.train()
            for task_epoch in range(1, args.tasknet_epochs+1):
                epoch_loss = 0
                for batch_data in train_loader:
                    batch_data = batch_data.to(device)
                    _, pr_data = grad_norm_split(gnn, tasknet, batch_data, args.gn_split, args.top_ratio, "Batch", return_info=False)

                    if args.random_d * args.random_c == 1:
                        prompt, _ = attach_random_prompt(args, batch_data, device)
                    else:
                        prompt, _, _, _ = attach_prompt(args, policy, pr_data, "train", None, gnn, tasknet, compute_reward=0, compute_jsd=0, compute_prr=0)

                    if args.tasknet_train_mode:
                        pr_loss, _, _ = tasknet_loss(gnn, tasknet, pr_data, prompt, policy_gnn_update=True)
                    else:
                        pr_loss, _, _ = tasknet_loss(gnn, tasknet, pr_data, prompt)
                    tasknet_optim.zero_grad()
                    pr_loss.backward()
                    tasknet_optim.step()
                    epoch_loss += pr_loss.detach().item()
                if not args.sh_mode:
                    if args.tasknet_epochs == task_epoch:
                        print("[{}/{}] Prompted Tasknet Loss with linked policy: {:.5f}".format(
                            task_epoch, args.tasknet_epochs, epoch_loss/len(train_loader)))

    # ===== Evaluate policy ===== #
    if policy_epoch % args.check_freq == 0 or policy_epoch == 1 or policy_epoch == args.policy_epochs or warmup:

        if policy_epoch < args.flash_epoch and policy_epoch != 1:
            train_auc, val_auc, test_auc = old_train_auc, old_val_auc, old_test_auc
            train_r_list, val_r_list, test_r_list = None, None, None
        else:
            if args.sh_mode:
                train_r_list, val_r_list, test_r_list = None, None, None
                if policy_epoch % args.check_freq == 0 or policy_epoch == 1 or policy_epoch == args.policy_epochs:
                    if args.skip_train_eval and args.gn_split == "top" and args.top_ratio == 1:
                        train_jsd = float("inf")
                        train_auc = 0
                        train_sh_info = [0, 0]
                    else:
                        train_jsd, train_auc, train_sh_info = evaluate_policy_sh(args, gnn, tasknet, policy, train_loader, "train", None, device)
                    val_auc, val_sh_info = evaluate_policy_sh(args, gnn, tasknet, policy, val_loader, "val", train_jsd, device)
                    test_auc, test_sh_info = evaluate_policy_sh(args, gnn, tasknet, policy, test_loader, "test", train_jsd, device)
                    print("==== epoch{:02d} [{}] AUC {:.2f} {:.2f} {:.2f} | Loss {:.5f} {:.5f} {:.5f} | "
                          "TEST P_min {:.3f} P_max {:.3f} P_norm {:.6f} P_Ratio {:.3f}".format(
                        policy_epoch, get_time(),
                        train_sh_info[0], val_sh_info[0], test_sh_info[0], train_sh_info[1], val_sh_info[1], test_sh_info[1],
                        test_sh_info[-4], test_sh_info[-3], test_sh_info[-2], test_sh_info[-1]
                    ))
            else:
                print("\nEVAL [{}]".format(get_time()))
                train_jsd, train_auc, train_r_list = evaluate_policy(args, gnn, tasknet, policy, train_loader, "train", None, old_train_auc, device)
                val_auc, val_r_list = evaluate_policy(args, gnn, tasknet, policy, val_loader, "val", train_jsd, old_val_auc, device)
                test_auc, test_r_list = evaluate_policy(args, gnn, tasknet, policy, test_loader, "test", train_jsd, old_test_auc, device)
            
            if val_auc > best_val_auc and policy_epoch >= 20:
                best_epoch = policy_epoch; best_train_auc = train_auc; best_val_auc = val_auc; best_test_auc = test_auc
            
    else:
        train_auc, val_auc, test_auc = old_train_auc, old_val_auc, old_test_auc
        train_r_list, val_r_list, test_r_list = None, None, None

    return policy, tasknet, (train_auc, val_auc, test_auc), (best_epoch, best_train_auc, best_val_auc, best_test_auc), (train_r_list, val_r_list, test_r_list)


def evaluate_policy(args, gnn: PromptedGNN, tasknet: GNNBasedNet, policy: H_PPO, loader, data_type, train_jsd, old_auc, device):
    start_time = time.time()
    gnn.eval(); tasknet.eval(); policy.train_or_eval(mode='eval')

    og_ys, pr_ys, og_logits, pr_logits, opr_logits, og_losses, pr_losses, opr_losses = [copy.deepcopy([]) for _ in range(8)]
    rewards, jsd_list = [], []
    grad_norms, pr_grad_norms, og_grad_norms = [], [], []

    num_pr_step, num_total_node = 0, 0
    pr_ratio_list = []
    p_min, p_max = float("inf"), -float("inf")
    for batch_data in loader:
        batch_data = batch_data.to(device)
        if data_type == "train":
            og_data, pr_data, pr_gn, og_gn, gn = grad_norm_split(gnn, tasknet, batch_data, args.gn_split, args.top_ratio)
            grad_norms.extend(gn)
            if len(pr_gn):
                pr_grad_norms.extend(pr_gn)
            if len(og_gn):
                og_grad_norms.extend(og_gn)
        else:
            num_total_node += batch_data.x.size(0)
            og_data, pr_data = None, batch_data

        if pr_data is not None:
            if data_type == "train":
                if args.random_d * args.random_c == 1:
                    prompt, _ = attach_random_prompt(args, batch_data, device)
                else:
                    prompt, r_list, jsd, pr_ratio = attach_prompt(args, policy, pr_data, data_type, None, gnn, tasknet,
                                                                compute_reward=1, compute_jsd=1, compute_prr=1)
                    jsd_list.extend(jsd); pr_ratio_list.extend(pr_ratio)
            else:
                if args.random_d * args.random_c == 1:
                    prompt, _ = attach_random_prompt(args, batch_data, device)
                else:
                    prompt, r_list, pr_steps, pr_ratio = attach_prompt(args, policy, pr_data, data_type, train_jsd, gnn, tasknet,
                                                                    compute_reward=1, compute_jsd=1, compute_prr=1)
                    num_pr_step += pr_steps
                    pr_ratio_list.extend(pr_ratio)
            
            if pr_data is not None:
                pr_loss, pr_logit, pr_y = tasknet_loss(gnn, tasknet, pr_data, prompt, require_grad=False, keep_loss_dim=True)
                opr_loss, opr_logit, _ = tasknet_loss(gnn, tasknet, pr_data, None, require_grad=False, keep_loss_dim=True)
                pr_losses.append(pr_loss.cpu()); pr_logits.append(pr_logit); pr_ys.append(pr_y)
                opr_losses.append(opr_loss.cpu()); opr_logits.append(opr_logit)
                rewards.extend([torch.from_numpy(r) for r in r_list])

            valid_prompt = prompt[prompt.any(dim=-1)!=0]
            if valid_prompt.min() < p_min:
                p_min = valid_prompt.min()
            if valid_prompt.max() > p_max:
                p_max = valid_prompt.max()

        if og_data is not None:
            og_loss, og_logit, og_y = tasknet_loss(gnn, tasknet, og_data, None, require_grad=False, keep_loss_dim=True)
            og_losses.append(og_loss.cpu()); og_logits.append(og_logit); og_ys.append(og_y)

    if len(og_ys):
        og_ys = torch.cat(og_ys); og_logits = torch.cat(og_logits); og_losses = torch.cat(og_losses)
        # og_logits0 = og_logits[og_ys == 0].mean().item(); og_logits1 = og_logits[og_ys == 1].mean().item()
    else:
        og_ys, og_logits, og_losses = [torch.tensor([]) for _ in range(3)]
        # og_logits0, og_logits1 = -float('inf'), float('inf')
    if len(pr_ys):
        pr_ys = torch.cat(pr_ys)
        pr_logits = torch.cat(pr_logits); opr_logits = torch.cat(opr_logits)
        pr_losses = torch.cat(pr_losses); opr_losses = torch.cat(opr_losses).mean().item()
        max_opr_logits0 = opr_logits[pr_ys == -1].max().item() if opr_logits[pr_ys == -1].numel() > 0 else -float("inf")
        max_pr_logits0 = pr_logits[pr_ys == -1].max().item() if pr_logits[pr_ys == -1].numel() > 0 else -float("inf")
        min_opr_logits1 = opr_logits[pr_ys == 1].min().item() if opr_logits[pr_ys == 1].numel() > 0 else -float("inf")
        min_pr_logits1 = pr_logits[pr_ys == 1].min().item() if pr_logits[pr_ys == 1].numel() > 0 else -float("inf")
        opr_logits0 = opr_logits[pr_ys == -1].mean().item() if opr_logits[pr_ys == -1].numel() > 0 else -float("inf")
        opr_logits1 = opr_logits[pr_ys == 1].mean().item() if opr_logits[pr_ys == 1].numel() > 0 else -float("inf")
        pr_logits0 = pr_logits[pr_ys == -1].mean().item() if pr_logits[pr_ys == -1].numel() > 0 else -float("inf")
        pr_logits1 = pr_logits[pr_ys == 1].mean().item() if pr_logits[pr_ys == 1].numel() > 0 else -float("inf")
    else:
        pr_ys, pr_logits, pr_losses = [torch.tensor([]) for _ in range(3)]
        opr_logits0, pr_logits0, opr_logits1, pr_logits1 = -float('inf'), -float('inf'), float('inf'), float('inf')

    ys = torch.cat([og_ys, pr_ys]); logits = torch.cat([og_logits, pr_logits]); losses = torch.cat([og_losses, pr_losses]).mean().item()
    og_losses = og_losses.mean().item(); pr_losses = pr_losses.mean().item()
    r_sum_list = [reward.sum().item() for reward in rewards]

    auc = compute_auc(logits, ys, "Total_AUC"); og_auc = compute_auc(og_logits, og_ys, "OG_AUC")
    pr_auc = compute_auc(pr_logits, pr_ys, "PR_AUC"); opr_auc = compute_auc(opr_logits, pr_ys, "OPR_AUC")

    end_time = time.time()
    print_str = "train"
    if data_type == "train":
        mean_jsd = sum(jsd_list)/len(jsd_list) if len(jsd_list) > 0 else float("inf")
        # print("TRAIN: {:.2f}s | JSD per step {:.2e} | GRAD_NORM (split by {}) pr {:.3f}  og {:.3f}  mean {:.3f}  median {:.3f}".format(
        #     end_time-start_time, mean_jsd, args.gn_split, 
        #     sum(pr_grad_norms)/len(pr_grad_norms) if len(pr_grad_norms) else -1, 
        #     sum(og_grad_norms)/len(og_grad_norms) if len(og_grad_norms) else -1, 
        #     torch.tensor(grad_norms).mean().item(), torch.tensor(grad_norms).median().item()
        # ))
    else:
        print_str = "VAL  :" if data_type == "val" else "TEST :"
        # print("{} {:.2f}s | PROMPT budget {}/{} ({:.2f}%)".format(
        #     print_str, end_time-start_time, num_pr_step, num_total_node, num_pr_step*100/num_total_node))

    print("{} AUC {:.2f} RewardSum {:.10f}".format(
          print_str, auc, sum(r_sum_list)/len(r_sum_list) if len(r_sum_list) else -1))
    # print("       AUC {:.2f} -> {:.2f}  OG_AUC {:.2f}  PR_AUC {:.2f} -> {:.2f} ({}{:.2f}%)\n"
    #       "       Lgt0 {:.4e} -> {:.4e}  Max Lgt0 {:.4e} -> {:.4e} | Lgt1 {:.4e} -> {:.4e}  Min Lgt1 {:.4e} -> {:.4e}\n"
    #       "       Loss {:.4f}  OG_LOSS {:.4f}  PR_Loss {:.4f} -> {:.4f} ({}{:.2f}%) | PR_RewardSum {:.5f} | PR_Ratio {:.4f} | P {:.3f} {:.3f}\n".format(
    #     old_auc, auc, og_auc, opr_auc, pr_auc, "↑" if pr_auc>=opr_auc else "↓",
    #     abs(pr_auc-opr_auc)*100/opr_auc if opr_auc > 0 else 0,
    #     opr_logits0, pr_logits0, max_opr_logits0, max_pr_logits0, opr_logits1, pr_logits1, min_opr_logits1, min_pr_logits1,
    #     losses, og_losses, opr_losses, pr_losses, "↑" if pr_losses>=opr_losses else "↓",
    #     abs(pr_losses-opr_losses)*100/opr_losses, sum(r_sum_list)/len(r_sum_list) if len(r_sum_list) else -1,
    #     sum(pr_ratio_list)/len(pr_ratio_list), p_min, p_max))

    if data_type == "train":
        return mean_jsd, auc, r_sum_list
    else:
        return auc, r_sum_list


def evaluate_policy_sh(args, gnn: PromptedGNN, tasknet: GNNBasedNet, policy: H_PPO, loader, data_type, train_jsd, device):
    gnn.eval(); tasknet.eval(); policy.train_or_eval(mode='eval')

    og_ys, pr_ys, og_logits, pr_logits, og_losses, pr_losses = [copy.deepcopy([]) for _ in range(6)]
    jsd_list, pr_ratio_list = [], []
    p_min, p_max, p_norm = float("inf"), -float("inf"), []

    for batch_data in loader:
        batch_data = batch_data.to(device)
        if data_type == "train":
            og_data, pr_data = grad_norm_split(gnn, tasknet, batch_data, args.gn_split, args.top_ratio, return_info=False)
        else:
            og_data, pr_data = None, batch_data

        if pr_data is not None:
            if data_type == "train":
                if args.random_d * args.random_c == 1:
                    prompt, pr_ratio = attach_random_prompt(args, batch_data, device)
                    pr_ratio_list.extend(pr_ratio)
                else:
                    if args.jsd_coeff > 10:  # no need to compute jsd
                        prompt, _, _, _ = attach_prompt(args, policy, pr_data, data_type, None, gnn, tasknet,
                                                        compute_reward=0, compute_jsd=0, compute_prr=0)
                    else:
                        prompt, _, jsd, _ = attach_prompt(args, policy, pr_data, data_type, None, gnn, tasknet,
                                                          compute_reward=0, compute_jsd=1, compute_prr=0)
                        jsd_list.extend(jsd)
                    # prompt, _, jsd, pr_ratio = attach_prompt(args, policy, pr_data, data_type, None, gnn, tasknet,
                    #                                          compute_reward=0, compute_jsd=1, compute_prr=1)
                    # jsd_list.extend(jsd); pr_ratio_list.extend(pr_ratio)
            else:
                if args.random_d * args.random_c == 1:
                    prompt, pr_ratio = attach_random_prompt(args, batch_data, device)
                    pr_ratio_list.extend(pr_ratio)
                else:
                    if args.jsd_coeff > 10:  # no need to compute jsd or use training_jsd
                        prompt, _, _, pr_ratio = attach_prompt(args, policy, pr_data, data_type, None, gnn, tasknet,
                                                               compute_reward=0, compute_jsd=0, compute_prr=1)
                    else:
                        prompt, _, _, pr_ratio = attach_prompt(args, policy, pr_data, data_type, train_jsd, gnn, tasknet,
                                                               compute_reward=0, compute_jsd=1, compute_prr=1)
                    pr_ratio_list.extend(pr_ratio)
                    # prompt, _, _, pr_ratio = attach_prompt(args, policy, pr_data, data_type, train_jsd, gnn, tasknet,
                    #                                        compute_reward=0, compute_jsd=1, compute_prr=1)
                    # pr_ratio_list.extend(pr_ratio)
            if pr_data is not None:
                pr_loss, pr_logit, pr_y = tasknet_loss(gnn, tasknet, pr_data, prompt, require_grad=False, keep_loss_dim=True)
                pr_losses.append(pr_loss.cpu()); pr_logits.append(pr_logit); pr_ys.append(pr_y)

        if og_data is not None:
            og_loss, og_logit, og_y = tasknet_loss(gnn, tasknet, og_data, None, require_grad=False, keep_loss_dim=True)
            og_losses.append(og_loss.cpu()); og_logits.append(og_logit); og_ys.append(og_y)

        valid_prompt = prompt[prompt.any(dim=1)!=0]
        if valid_prompt.min() < p_min:
            p_min = valid_prompt.min()
        if valid_prompt.max() > p_max:
            p_max = valid_prompt.max()
        # p_norm.extend(torch.sqrt(torch.sum(valid_prompt ** 2, dim=-1)).tolist())
        p_norm.extend(torch.mean(torch.abs(valid_prompt), dim=-1).tolist())

    if len(og_ys):
        og_ys = torch.cat(og_ys); og_logits = torch.cat(og_logits); og_losses = torch.cat(og_losses)
    else:
        og_ys, og_logits, og_losses = [torch.tensor([]) for _ in range(3)]

    if len(pr_ys):
        pr_ys = torch.cat(pr_ys); pr_logits = torch.cat(pr_logits); pr_losses = torch.cat(pr_losses)
    else:
        pr_ys, pr_logits, pr_losses = [torch.tensor([]) for _ in range(3)]

    ys = torch.cat([og_ys, pr_ys]); logits = torch.cat([og_logits, pr_logits]); losses = torch.cat([og_losses, pr_losses]).mean().item()
    auc = compute_auc(logits, ys, "Total_AUC")
    sh_mode_info = [auc, losses, p_min, p_max, sum(p_norm)/len(p_norm), sum(pr_ratio_list)/len(pr_ratio_list) if len(pr_ratio_list) else 0]

    if data_type == "train":
        mean_jsd = sum(jsd_list)/len(jsd_list) if len(jsd_list) > 0 else float("inf")
        return mean_jsd, auc, sh_mode_info
    else:
        return auc, sh_mode_info
