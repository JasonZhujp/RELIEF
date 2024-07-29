import copy, time

import torch
from torch_geometric.data import Batch
from torch_scatter import scatter
from sklearn.metrics import f1_score, accuracy_score

from reward import compute_adv_ret, reshape_reward, reward_reshape, RewardScaling, Normalization
from agent import H_PPO

criterion = torch.nn.CrossEntropyLoss(reduction="none")

def attach_prompt(args, policy: H_PPO, data, data_type, gnn=None, tasknet=None, compute_reward=0, compute_prr=0):
    device = data.x.device

    graph_num = data.num_graphs
    nodes_per_graph = scatter(torch.ones_like(data.batch), data.batch, reduce='add')
    policy.prompt_slices = [torch.zeros(nodes, args.svd_dim).to(device) for nodes in nodes_per_graph]
    policy.step_state = torch.zeros(graph_num, policy.max_num_nodes, args.hid_dim).to(device)
    base_mask = torch.zeros(graph_num, policy.max_num_nodes).to(device)
    base_idx = torch.arange(0, policy.max_num_nodes).to(device)
    policy.batch_mask = torch.where(base_idx < nodes_per_graph.unsqueeze(-1), base_mask, torch.ones_like(base_mask))

    if data_type != "train":
        total_prompt_step_num = 0
    if compute_reward:
        init_r = generate_reward(gnn, tasknet, data)
        r_list = [copy.deepcopy([]) for _ in range(graph_num)]
        for rl, ir in zip(r_list, init_r):
            rl.append(ir)
    
    truncate_flag = torch.zeros(graph_num)
    batch_max_step = nodes_per_graph.max().item()
    for step in range(batch_max_step):
        if truncate_flag.sum() == graph_num:
            break
        else:
            policy.eval_prompt(data, nodes_per_graph, step, truncate_flag)

        if compute_reward:
            rs = generate_reward(gnn, tasknet, data, torch.cat(policy.prompt_slices))
            for i in range(graph_num):
                if step < nodes_per_graph[i] and truncate_flag[i] == 0:
                    r_list[i].append(rs[i])
        
        if data_type != "train":
            total_prompt_step_num += (~truncate_flag.bool() & (step<nodes_per_graph).cpu()).sum()

    reward_list, pr_ratio_list = None, None
    if compute_reward:
        reward_list = [reward_reshape(torch.stack(r).cpu().numpy(), reward_clip=args.reward_clip) for r in r_list]
    if compute_prr:
        pr_ratio_list = [torch.any(p!=0, dim=-1).sum().item()/p.size(0) for p in policy.prompt_slices]
    
    return torch.cat(policy.prompt_slices), reward_list, pr_ratio_list


def compute_acc_f1(pred, y):
    if pred.size(0) == 0 or y.size(0) == 0:
        return -1, -1
    acc = accuracy_score(y, pred)
    macro_f1 = f1_score(y, pred, average='macro')

    return acc*100, macro_f1*100


def tasknet_loss(gnn, tasknet, data, prompt=None, require_grad=True, keep_loss_dim=False, policy_gnn_update=False):
    if policy_gnn_update:
        gnn.train()

    with torch.no_grad():
        graph_emb = gnn(data.x, data.edge_index, prompt, data.batch)
    if require_grad:
        logit = tasknet(graph_emb)
    else:
        with torch.no_grad():
            logit = tasknet(graph_emb)

    y = data.y
    loss = criterion(logit, y)
    if not keep_loss_dim:
        loss = torch.mean(loss)

    if policy_gnn_update:
        gnn.eval()

    return loss, logit.detach().cpu(), data.y.cpu()


def generate_reward(gnn, tasknet, data, prompt=None):
    # prompted_x = data.x if prompt is None else data.x + prompt
    with torch.no_grad():
        graph_emb = gnn(data.x, data.edge_index, prompt, data.batch)
        logit = tasknet(graph_emb)

    y = data.y
    reward = criterion(logit, y)    
    return reward


def tasknet_warmup(args, gnn, tasknet, tasknet_optim, train_loader, val_loader, test_loader, device):
    print("\n====== Warmup tasknet ======")
    for epoch in range(1, args.warmup_epochs+1):
        gnn.train(); tasknet.train()
        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            tasknet_optim.zero_grad()
            loss, _, _ = tasknet_loss(gnn, tasknet, batch_data)
            loss.backward()
            tasknet_optim.step()
        
        if epoch == 1 or epoch % 5 == 0 or epoch == args.warmup_epochs:
            train_loss, train_acc, train_f1 = evaluate_tasknet(gnn, tasknet, train_loader, device)
            val_loss, val_acc, val_f1 = evaluate_tasknet(gnn, tasknet, val_loader, device)
            test_loss, test_acc, test_f1 = evaluate_tasknet(gnn, tasknet, test_loader, device)
            print("[Epoch {:02d}/{}] LOSS:{:.5f} {:.5f} {:.5f} | ACC:{:.2f} {:.2f} {:.2f} | F1:{:.2f} {:.2f} {:.2f}".format(
                    epoch, args.warmup_epochs, train_loss, val_loss, test_loss,
                    train_acc, val_acc, test_acc, train_f1, train_f1, test_f1))
        
    return tasknet, (train_acc, val_acc, test_acc), (train_f1, val_f1, test_f1)
    

def evaluate_tasknet(gnn, tasknet, loader, device):
    gnn.eval(); tasknet.eval()
    epoch_loss = 0
    preds, ys = [], []
    for batch_data in loader:
        batch_data = batch_data.to(device)
        loss, logit, y = tasknet_loss(gnn, tasknet, batch_data, prompt=None, require_grad=False)
        pred = logit.argmax(dim=1)
        preds.append(pred); ys.append(y)
        epoch_loss += loss.item()
    epoch_loss /= len(loader)
    preds = torch.cat(preds); ys = torch.cat(ys)
    acc, f1 = compute_acc_f1(preds, ys)

    return epoch_loss, acc, f1


def train_policy(args, epoch, gnn, tasknet, tasknet_optim, policy: H_PPO, policy_optims,
                 ens_loaders, train_loader, val_loader, test_loader, accs, f1s, best_infos, device):
    
    old_train_acc, old_val_acc, old_test_acc = accs
    old_train_f1, old_val_f1, old_test_f1 = f1s
    if best_infos is None:
        best_epoch, best_train_acc, best_val_acc, best_test_acc, best_train_f1, best_val_f1, best_test_f1 = (-1, -1, -1, -1, -1, -1, -1)
    else:
        best_epoch, best_train_acc, best_val_acc, best_test_acc, best_train_f1, best_val_f1, best_test_f1 = best_infos

    # ====== Annealing the learning rate ====== #
    descend_decay_frac = 1.0 - (epoch - 1) / (args.total_epochs)
    policy_decay_frac, tasknet_decay_frac = 1.0, 1.0

    if args.policy_decay == "down":
        policy_decay_frac = descend_decay_frac
    for i in range(args.ensemble_num):
        for param_group in policy_optims[i].param_groups:
            if param_group["name"] == "actor_d":
                param_group["lr"] = policy_decay_frac * args.actor_d_lr
            elif param_group["name"] == "actor_c":
                param_group["lr"] = policy_decay_frac * args.actor_c_lr
            elif param_group["name"] == "critic":
                param_group["lr"] = policy_decay_frac * args.critic_lr

    if args.tasknet_decay == "down":
        tasknet_decay_frac = descend_decay_frac
    for param_group in tasknet_optim.param_groups:
        param_group["lr"] = tasknet_decay_frac * args.tasknet_lr

    # decreasing the entropy of discrete actors for more stable descrete actions sampling
    policy.coeff_ent_d = args.coeff_entropy_d * descend_decay_frac

    # decreasing the log_std for more stable continuous actions sampling
    for i in range(args.ensemble_num):
        new_log_std = args.init_log_std + (-5 - args.init_log_std) * (epoch - 1) / args.total_epochs
        policy.actor_cs[i].log_std = (torch.ones(args.svd_dim) * new_log_std).to(device)

    if not args.sh_mode:
        print("\n===== EPOCH {}/{} ===== policy lr*{:.2f} tasknet lr*{:.2f}".format(
              epoch, args.total_epochs, policy_decay_frac, tasknet_decay_frac))
    

    # ====== Agent sampling to collect transitions ====== #
    tasknet.eval(); policy.train_or_eval(mode='train')
    reward_transform = Normalization(shape=1)
    
    # train each ensemble sub-actors (sub-policies)
    for a_idx in range(args.ensemble_num):
        gnn.eval()

        batch_data_list = []
        for batch_idx, batch_data in enumerate(ens_loaders[a_idx]):
            batch_data_list.extend(batch_data.to_data_list())
            # guarantee enough amount of graphs or it's the last batch of data
            if len(batch_data_list) < args.batch_size and batch_idx < len(ens_loaders[a_idx])-1:
                continue
            batch_data = Batch.from_data_list(batch_data_list).to(device)
            batch_data_list = []

            nodes_per_graph = scatter(torch.ones_like(batch_data.batch), batch_data.batch, reduce='add')
            graph_num = batch_data.num_graphs
            batch_max_step = nodes_per_graph.max().item()
            state = torch.zeros([graph_num, batch_max_step, policy.max_num_nodes, args.hid_dim]).to(device)
            action_d = torch.zeros([graph_num, batch_max_step]).long().to(device)
            action_c = torch.zeros([graph_num, batch_max_step, args.svd_dim]).to(device)
            logprob_d = torch.zeros([graph_num, batch_max_step]).to(device)
            logprob_c = torch.zeros([graph_num, batch_max_step]).to(device)
            reward = torch.zeros([graph_num, batch_max_step+1]).to(device)
            with torch.no_grad():
                init_r = generate_reward(gnn, tasknet, batch_data)
            reward[:, 0] = init_r
            done = torch.zeros([graph_num, batch_max_step]).to(device)
            # mask invalid step for each graph
            valid_mask = (torch.arange(batch_max_step).expand(graph_num,-1).to(device)) < nodes_per_graph.unsqueeze(1)  # (num_graphs, batch_max_step)

            # collect a batch of episodes
            policy.prompt_slices = [torch.zeros(nodes, args.svd_dim).to(device) for nodes in nodes_per_graph]
            policy.step_state = torch.zeros(graph_num, policy.max_num_nodes, args.hid_dim).to(device)
            # mask nodes corresponding to `zero padding`
            base_mask = torch.zeros(graph_num, policy.max_num_nodes).to(device)
            base_idx = torch.arange(0, policy.max_num_nodes).to(device)
            policy.batch_mask = torch.where(base_idx < nodes_per_graph.unsqueeze(-1), base_mask, torch.ones_like(base_mask))
            for step in range(batch_max_step):
                done[nodes_per_graph == step + 1, step] = 1.0
                # attach a new feature prompt on a specific node for each graph
                s, a_d, a_c, lp_d, lp_c = policy.train_prompt(a_idx, batch_data, nodes_per_graph, step)
                state[:, step] = s
                action_d[:, step] = a_d
                action_c[:, step] = a_c
                logprob_d[:, step] = lp_d
                logprob_c[:, step] = lp_c
                # execute downstream task with prompted graph to recieve reward
                r = generate_reward(gnn, tasknet, batch_data, prompt=torch.cat(policy.prompt_slices))
                reward[:, step+1] = r

            # mask invalid step and compute scaled rewards, advantages, returns
            next_state = torch.zeros_like(state)
            for i in range(graph_num):
                if nodes_per_graph[i] > 1:
                    valid_s = state[i, :nodes_per_graph[i]]
                    next_state[i, :nodes_per_graph[i]-1] = valid_s[1:]
                    next_state[i, nodes_per_graph[i]-1] = valid_s[-1]
                else:
                    next_state[i, 0] = state[i, 0]
            reward = reshape_reward(reward, nodes_per_graph)
            state = state[valid_mask]; next_state = next_state[valid_mask]; done = done[valid_mask]
            advantage, approx_return, approx_return0, scaled_reward = \
                compute_adv_ret(args, policy.critic, state, reward, next_state, done, nodes_per_graph.tolist(), reward_transform)
            action_d = action_d[valid_mask]; action_c = action_c[valid_mask]
            logprob_d = logprob_d[valid_mask]; logprob_c = logprob_c[valid_mask]

            if state.size(0) > args.minibatch_size:
                experience = (state, action_d, logprob_d, action_c, logprob_c, advantage, approx_return)
                policy.train_policy(args, a_idx, experience, nodes_per_graph, policy_optims[a_idx], approx_return0, scaled_reward, batch_idx+1, len(ens_loaders[a_idx]))
            else:
                if not args.sh_mode:
                    print("No enough trainsitions!")


    # ====== Update tasknet according to joint policy ====== #
    gnn.eval(); policy.train_or_eval('eval'); tasknet.train()
    for task_epoch in range(1, args.tasknet_epochs+1):
        epoch_loss = 0
        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            prompt, _, _ = attach_prompt(args, policy, batch_data, "train", gnn, tasknet, compute_reward=0, compute_prr=0)

            if args.tasknet_train_mode:
                pr_loss, _, _ = tasknet_loss(gnn, tasknet, batch_data, prompt, policy_gnn_update=True)
            else:
                pr_loss, _, _ = tasknet_loss(gnn, tasknet, batch_data, prompt)
            tasknet_optim.zero_grad()
            pr_loss.backward()
            tasknet_optim.step()
            epoch_loss += pr_loss.detach().item()
        if not args.sh_mode:
            print("[{}/{}] Prompted Tasknet Loss with joint policy: {:.5f}".format(
                task_epoch, args.tasknet_epochs, epoch_loss/len(train_loader)))


    # ====== Evaluation ====== #
    if epoch % args.check_freq == 0 or epoch == 1 or epoch == args.total_epochs:

        if epoch < args.skip_epoch and epoch != 1:
            train_acc, val_acc, test_acc = old_train_acc, old_val_acc, old_test_acc
            train_f1, val_f1, test_f1 = old_train_f1, old_val_f1, old_test_f1
        else:
            train_acc, train_f1, train_loss = evaluate_policy(args, gnn, tasknet, policy, train_loader, "train", device)
            val_acc, val_f1, val_loss = evaluate_policy(args, gnn, tasknet, policy, val_loader, "val", device)
            test_acc, test_f1, test_info = evaluate_policy(args, gnn, tasknet, policy, test_loader, "test", device)
            print("==== Epoch{:02d} ACC {:.2f} {:.2f} {:.2f} | F1 {:.2f} {:.2f} {:.2f} | Loss {:.5f} {:.5f} {:.5f} | "
                  "TEST P_min {:.3f} P_max {:.3f} P_norm {:.3f} P_Ratio {:.3f}".format(
                  epoch,
                  train_acc, val_acc, test_acc, train_f1, val_f1, test_f1, train_loss, val_loss, test_info[0],
                  test_info[1], test_info[2], test_info[3], test_info[4]                   
            ))
            
            if val_f1 > best_val_f1 and epoch >= 20:
                best_epoch = epoch
                best_train_acc = train_acc; best_val_acc = val_acc; best_test_acc = test_acc
                best_train_f1 = train_f1; best_val_f1 = val_f1; best_test_f1 = test_f1
            
    else:
        train_acc, val_acc, test_acc = old_train_acc, old_val_acc, old_test_acc
        train_f1, val_f1, test_f1 = old_train_f1, old_val_f1, old_test_f1

    return policy, tasknet, (train_acc, val_acc, test_acc), (train_f1, val_f1, test_f1),\
           (best_epoch, best_train_acc, best_val_acc, best_test_acc, best_train_f1, best_val_f1, best_test_f1)


def evaluate_policy(args, gnn, tasknet, policy, loader, data_type, device):
    gnn.eval(); tasknet.eval(); policy.train_or_eval(mode='eval')

    pr_ys, pr_preds, pr_losses, pr_ratio_list = [], [], [], []
    p_min, p_max, p_norm = float("inf"), -float("inf"), []

    for batch_data in loader:
        batch_data = batch_data.to(device)

        if data_type == "test":
            prompt, _, pr_ratio = attach_prompt(args, policy, batch_data, data_type, gnn, tasknet, compute_reward=0, compute_prr=1)
            pr_ratio_list.extend(pr_ratio)
        else:
            prompt, _, _, = attach_prompt(args, policy, batch_data, data_type, gnn, tasknet, compute_reward=0, compute_prr=0)

        pr_loss, pr_logit, pr_y = tasknet_loss(gnn, tasknet, batch_data, prompt, require_grad=False, keep_loss_dim=True)
        pr_ys.append(pr_y); pr_preds.append(pr_logit.argmax(dim=1)); pr_losses.append(pr_loss.cpu())

        if data_type == "test":    
            valid_prompt = prompt[prompt.any(dim=1)!=0]
            if valid_prompt.min() < p_min:
                p_min = valid_prompt.min()
            if valid_prompt.max() > p_max:
                p_max = valid_prompt.max()
            p_norm.extend(torch.mean(torch.abs(valid_prompt), dim=-1).tolist())

    ys = torch.cat(pr_ys); preds = torch.cat(pr_preds); losses = torch.cat(pr_losses).mean().item()
    acc, f1 = compute_acc_f1(preds, ys)
    
    if data_type == "test":
        test_info = [losses, p_min, p_max, sum(p_norm)/len(p_norm), sum(pr_ratio_list)/len(pr_ratio_list)]
        return acc, f1, test_info
    else:
        return acc, f1, losses
