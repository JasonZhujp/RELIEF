import copy, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np

from reward import RunningMeanStd
from model import PromptedGNN


# [Trick] orthogonal initialization
def orthogonal_init(layer, gain=np.sqrt(2)):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


def xavier_init(layer):
    nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('relu'))
    nn.init.constant_(layer.bias, 0)


class ActorDiscrete(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        # [Trick] Tanh is more suitable for PPO compared to ReLU
        self.linears = nn.Sequential(
            nn.Linear(emb_dim, int(emb_dim * 0.5)),
            nn.Tanh(),
            nn.Linear(int(emb_dim * 0.5), int(emb_dim * 0.25)),
            nn.Tanh(),
            nn.Linear(int(emb_dim * 0.25), 1)
        )
        orthogonal_init(self.linears[0])
        orthogonal_init(self.linears[2])
        orthogonal_init(self.linears[4], gain=0.01)


    def forward(self, state, mask):
        output = self.linears(state).squeeze(-1)  # (batchsize, max_node)
        # mask logits corresponding to invalid nodes for each graph
        output = torch.where(mask==0, output, torch.full_like(output, float('-inf')))
        action_prob = F.softmax(output, dim=-1)
        return action_prob

   
    def get_action_logprob(self, state, mask, deterministic=False):
        action_prob = self.forward(state, mask)
        dist = Categorical(action_prob)
        if deterministic:
            action_d = torch.argmax(action_prob, dim=-1)  # (batchsize, )
        else:
            action_d = dist.sample()  # (batchsize, )
        logprob = dist.log_prob(action_d)  # (batchsize, )
        return action_d, logprob
    

    def pi(self, state, mask, require_grad=False):
        if require_grad:
            action_prob = self.forward(state, mask)
        else:
            with torch.no_grad():
                action_prob = self.forward(state, mask)
        return action_prob


    def logprob_entropy(self, state, action_d, mask):
        action_prob = self.forward(state, mask)
        dist = Categorical(action_prob)
        return dist.log_prob(action_d), dist.entropy()


class ActorContinuous(nn.Module):
    def __init__(self, emb_dim, max_action_con, init_log_std):
        super().__init__()
        self.max_action = max_action_con
        # self.log_std = nn.Parameter(torch.ones(emb_dim)*init_log_std)  # Use 'nn.Paremeter' to train log_std automatically
        self.log_std = (torch.ones(emb_dim) * init_log_std)
        self.linears = nn.Sequential(
            nn.Linear(emb_dim, int(emb_dim * 0.5)),
            nn.Tanh(),
            nn.Linear(int(emb_dim * 0.5), int(emb_dim * 0.5)),
            nn.Tanh(),
            nn.Linear(int(emb_dim * 0.5), emb_dim)
        )
        # self.mean_layer = nn.Linear(int(emb_dim/2), emb_dim)
        orthogonal_init(self.linears[0])
        orthogonal_init(self.linears[2])
        orthogonal_init(self.linears[4], gain=0.01)


    def forward(self, state, action_d):
        output = self.linears(state)  # (batchsize, max_num_node, emb_dim)
        index = action_d.view(action_d.size(0), 1, 1).expand(action_d.size(0), 1, output.size(-1))  # (batchsize,) -> (batchsize,1,emb_dim)
        output = torch.gather(output, dim=1, index=index).squeeze(1)  # (batchsize, emb_dim)
        mean = torch.tanh(output) * self.max_action
        return mean
    

    def get_action_logprob(self, state, action_d, deterministic=False):
        mean = self.forward(state, action_d)
        std = torch.exp(self.log_std.expand_as(mean))
        dist = Normal(mean, std)
        if deterministic:
            action = mean
        else:
            action = dist.sample()  # (batchsize, emb_dim)
        logprob = torch.sum(dist.log_prob(action), dim=-1)  # (batchsize,)
        return action, logprob
    

    def logprob_entropy(self, state, action_d, action_c):
        mean = self.forward(state, action_d)  # (batchsize, emb_dim)
        std = torch.exp(self.log_std.expand_as(mean))
        dist = Normal(mean, std)
        return torch.sum(dist.log_prob(action_c), dim=-1), torch.sum(dist.entropy(), dim=-1), dist
    

class Critic(nn.Module):
    def __init__(self, emb_dim, max_num_nodes):
        super().__init__()
        input_dim = int(emb_dim * max_num_nodes)
        self.flatten = nn.Flatten()
        self.linears = nn.Sequential(
            nn.Linear(input_dim, int(input_dim * 0.05)),
            nn.ReLU(),
            nn.Linear(int(input_dim * 0.05), int(input_dim * 0.025)),
            nn.ReLU(),
            nn.Linear(int(input_dim * 0.025), 1)
        )  # 1 stands for the state-value function V(s)
        # orthogonal_init(self.linears[0])
        # orthogonal_init(self.linears[2])
        # orthogonal_init(self.linears[4], gain=0.01)
        xavier_init(self.linears[0])
        xavier_init(self.linears[2])
        xavier_init(self.linears[4])


    def forward(self, state):
        state_value = self.linears(self.flatten(state)).squeeze(-1)
        return state_value  # (batchsize, )


class H_PPO():
    def __init__(self, gnn: PromptedGNN, policy_d_type, ensemble_num, penalty_alpha_d, penalty_alpha_c, emb_dim, max_num_nodes, max_action, init_log_std,
                 eps_clip_d, eps_clip_c, coeff_critic, coeff_entropy_d, coeff_entropy_c, max_norm_grad, device):

        self.gnn = gnn
        self.policy_d_type = policy_d_type
        self.ensemble_num = ensemble_num
        self.penalty_alpha_d = penalty_alpha_d
        self.penalty_alpha_c = penalty_alpha_c
        self.actor_ds = [copy.deepcopy(ActorDiscrete(emb_dim).to(device)) for _ in range(ensemble_num)]
        self.actor_cs = [copy.deepcopy(ActorContinuous(emb_dim, max_action, init_log_std).to(device)) for _ in range(ensemble_num)]
        self.critic = Critic(emb_dim, max_num_nodes).to(device)

        self.eps_clip_d = eps_clip_d
        self.eps_clip_c = eps_clip_c
        self.coeff_critic = coeff_critic
        self.coeff_ent_d = coeff_entropy_d
        self.coeff_ent_c = coeff_entropy_c
        self.max_norm_grad = max_norm_grad
        self.max_num_nodes = max_num_nodes

        self.state_scaler = RunningMeanStd(shape=emb_dim)
        self.device = device

  
    def train_or_eval(self, mode):
        if mode == 'train':
            for i in range(self.ensemble_num):
                self.actor_ds[i].train(); self.actor_cs[i].train()
            self.critic.train()
        elif mode == 'eval':
            for i in range(self.ensemble_num):
                self.actor_ds[i].eval(); self.actor_cs[i].eval()
            self.critic.eval()
        else:
            raise ValueError("Invalid model mode! Choose between 'train' and 'eval'.")


    def _node_to_state(self, batch_data, prompt, nodes_per_graph):
        with torch.no_grad():
            node_emb = self.gnn(batch_data.x, batch_data.edge_index, batch_data.edge_attr, prompt)
        state_slices = torch.split(node_emb, nodes_per_graph.tolist(), dim=0)
        for i, ss in enumerate(state_slices):
            self.step_state[i, :ss.size(0), :] = ss

        return self.step_state
    

    def train_prompt(self, args, a_idx:int, batch_data, nodes_per_graph, step, state_norm=0):
        """
        At each training time step, stochastically decide which node to operate on (discrete action) 
        and what prompt content to attach (continuous action).
        """
        with torch.no_grad():
            self.gnn.eval()
            batch_state = self._node_to_state(batch_data, torch.cat(self.prompt_slices), nodes_per_graph)
            # [Trick] State Normalization
            if state_norm:
                flatten_emb = batch_state.view(-1, batch_state.shape[-1]).cpu().numpy()
                valid_idx = np.any(flatten_emb!=0, axis=-1)
                valid_emb = flatten_emb[valid_idx]
                for i in range(valid_emb.shape[0]):
                    self.state_scaler.update(valid_emb[i])
                flatten_emb[valid_idx] = (flatten_emb[valid_idx] - self.state_scaler.mean) / (self.state_scaler.std + 1e-8)
                batch_state = torch.from_numpy(flatten_emb).view_as(batch_state).to(self.device)

            # discrete action
            if args.random_d:
                batch_action_d = torch.zeros_like(nodes_per_graph, dtype=torch.int64).to(batch_state.device)
                for i in range(nodes_per_graph.shape[0]):
                    if step < nodes_per_graph[i]:
                        batch_action_d[i] = torch.randint(0, nodes_per_graph[i].item(), (1,), dtype=torch.int64)
                batch_logp_d = torch.zeros_like(batch_action_d)
            else:
                batch_action_d, batch_logp_d = self.actor_ds[a_idx].get_action_logprob(batch_state, self.batch_mask)
            # batch_action_d, batch_logp_d = self.actor_ds[a_idx].get_action_logprob(batch_state, self.batch_mask)

            # continuous action
            if args.random_c:
                means = torch.empty((len(nodes_per_graph), args.emb_dim)).uniform_(-args.max_action_con, args.max_action_con)
                st_means = torch.normal(means, torch.ones_like(means) * math.exp(args.init_log_std)).to(batch_state.device)
                batch_action_c = torch.clamp(st_means, -args.max_action_con, args.max_action_con)
                batch_logp_c = torch.zeros(len(nodes_per_graph))
            else:
                batch_action_c, batch_logp_c = self.actor_cs[a_idx].get_action_logprob(batch_state, batch_action_d)
            # batch_action_c, batch_logp_c = self.actor_cs[a_idx].get_action_logprob(batch_state, batch_action_d)

            # update prompt
            for i in range(len(self.prompt_slices)):
                if step < nodes_per_graph[i]:
                    self.prompt_slices[i][batch_action_d[i]] += batch_action_c[i]

        return batch_state, batch_action_d, batch_action_c, batch_logp_d, batch_logp_c


    def _jsd(self, P, Q):
        M = 0.5 * (P + Q)
        
        kld_P_M = F.kl_div(torch.log(P), M, log_target=False, reduction="none")
        kld_P_M[torch.isnan(kld_P_M)] = 0
        kld_P_M = kld_P_M.sum(dim=-1)

        kld_Q_M = F.kl_div(torch.log(Q), M, log_target=False, reduction="none")
        kld_Q_M[torch.isnan(kld_Q_M)] = 0
        kld_Q_M = kld_Q_M.sum(dim=-1)

        jsd_value = 0.5 * (kld_P_M + kld_Q_M)
        return jsd_value.cpu()  # (batchsize, )


    def _compute_max_jsd(self, dists):
        num_dist = len(dists)
        max_jsd = torch.zeros(dists[0].shape[0])
        
        for i in range(num_dist):
            for j in range(i + 1, num_dist):
                jsd_value = self._jsd(dists[i], dists[j])
                max_jsd = torch.where(jsd_value > max_jsd, jsd_value, max_jsd)
        return max_jsd  # (batchsize, )


    def _get_eval_action_d(self, batch_state, batch_mask, compute_jsd):
        pis = []
        for i in range(self.ensemble_num):
            pis.append(self.actor_ds[i].pi(batch_state, batch_mask, require_grad=False))
        max_jsd = None
        if compute_jsd:
            max_jsd = self._compute_max_jsd(pis)
        ens_max_pi = torch.stack(pis, dim=1).max(dim=1)[0]
        # == determinisitic policy
        if self.policy_d_type == "de":
            ens_action_d = torch.argmax(ens_max_pi, dim=-1)
        # == stochastic policy
        elif self.policy_d_type == "st":
            ens_max_prob = F.normalize(ens_max_pi + 1e-10, p=1, dim=-1)
            dist = Categorical(ens_max_prob)
            ens_action_d = dist.sample()

        return ens_action_d, max_jsd
    
    
    def _get_eval_action_c(self, batch_state, batch_action_d):
        means = []
        for i in range(self.ensemble_num):
            with torch.no_grad():
                mean = self.actor_cs[i].forward(batch_state, batch_action_d)
            means.append(mean)
        ens_mean = torch.stack(means, dim=1).mean(dim=1)
        
        return ens_mean


    def eval_prompt(self, args, batch_data, nodes_per_graph, step, truncate_flag, state_norm=0, compute_jsd=0):
        """
        At each evaluation time step, deterministically decide which node to operate on (discrete action) 
        and what prompt content to attach (continuous action).
        """
        with torch.no_grad():
            self.gnn.eval()
            batch_state = self._node_to_state(batch_data, torch.cat(self.prompt_slices), nodes_per_graph)
            # [Trick] State Normalization
            if state_norm:
                flatten_emb = batch_state.view(-1, batch_state.shape[-1]).cpu().numpy()
                valid_idx = np.any(flatten_emb!=0, axis=-1)
                valid_emb = flatten_emb[valid_idx]
                for i in range(valid_emb.shape[0]):
                    self.state_scaler.update(valid_emb[i])
                flatten_emb[valid_idx] = (flatten_emb[valid_idx] - self.state_scaler.mean) / (self.state_scaler.std + 1e-8)
                batch_state = torch.from_numpy(flatten_emb).view_as(batch_state).to(self.device)

            # discrete action (When evaluating the policy, we select the action with the highest probability)
            if args.random_d:
                batch_action_d = torch.zeros_like(nodes_per_graph, dtype=torch.int64).to(self.device)
                for i in range(nodes_per_graph.shape[0]):
                    if step < nodes_per_graph[i]:
                        batch_action_d[i] = torch.randint(0, nodes_per_graph[i].item(), (1,), dtype=torch.int64)
                max_jsd = None
            else:
                batch_action_d, max_jsd = self._get_eval_action_d(batch_state, self.batch_mask, compute_jsd)
            # batch_action_d, max_jsd = self._get_eval_action_d(batch_state, self.batch_mask, compute_jsd)

            # continuous action (When evaluating the policy, we only use the mean)
            if args.random_c:
                batch_action_c = torch.empty((len(nodes_per_graph), args.emb_dim)).uniform_(-args.max_action_con, args.max_action_con).to(self.device)
            else:
                batch_action_c = self._get_eval_action_c(batch_state, batch_action_d)
            # batch_action_c = self._get_eval_action_c(batch_state, batch_action_d)
            
        # update prompt
        # for i in range(len(self.prompt_slices)):
        #     if step < nodes_per_graph[i] and truncate_flag[i] == 0:
        #         self.prompt_slices[i][batch_action_d[i]] += batch_action_c[i]
        for i in range(len(self.prompt_slices)):
            if step < nodes_per_graph[i] and truncate_flag[i] == 0:
                if batch_action_d[i] < self.prompt_slices[i].size(0):
                    self.prompt_slices[i][batch_action_d[i]] += batch_action_c[i]
                else:
                    print("Invalid action_d selected node {}, total #node {}.".format(
                        batch_action_d[i].item(), len(self.prompt_slices[i])))

        if compute_jsd:
            return max_jsd.tolist()
    

    def _ensemble_penalty_d(self, a_idx:int, state, mask):
        pis = []
        for i in range(self.ensemble_num):
            pi = self.actor_ds[i].pi(state, mask, require_grad=True if i==a_idx else False)
            pis.append(pi)
        
        with torch.no_grad():
            ens_max_pi = torch.stack(pis, dim=1).max(dim=1)[0]
            ens_pi = F.normalize(ens_max_pi, p=1, dim=-1)
        cur_pi = F.normalize(pis[a_idx] + 1e-10, p=1, dim=-1)
        ens_pi = F.normalize(ens_pi + 1e-10, p=1, dim=-1)
        kl_penalty_d = F.kl_div(torch.log(cur_pi), ens_pi, log_target=False, reduction='none').sum(dim=-1)  # (minibatch,)

        ens_action_d = torch.argmax(ens_max_pi, dim=-1)  # saved for computing ensemble_panalty_c for actor_cs later

        return kl_penalty_d * self.penalty_alpha_d, ens_action_d
    

    def _ensemble_penalty_c(self, a_idx, state, action_d):
        means, stds = [], []
        for i in range(self.ensemble_num):
            if i == a_idx:
                mean = self.actor_cs[i].forward(state, action_d)
                std = torch.exp(self.actor_cs[i].log_std.expand_as(mean))
            else:
                with torch.no_grad():
                    mean = self.actor_cs[i].forward(state, action_d)
                    std = torch.exp(self.actor_cs[i].log_std.expand_as(mean))
            means.append(mean); stds.append(std)
        
        with torch.no_grad():
            ens_mean = torch.stack(means, dim=1).mean(dim=1)
            ens_std = torch.stack(stds, dim=1).mean(dim=1)
        cur_dist = Normal(means[a_idx], stds[a_idx])
        ens_dist = Normal(ens_mean, ens_std)
        kl_penalty_c = torch.distributions.kl.kl_divergence(cur_dist, ens_dist).sum(dim=-1)

        return kl_penalty_c * self.penalty_alpha_c


    def _actor_loss(self, args, a_idx:int, state, action_d, action_c, adv, logprob_old_d, logprob_old_c, node_num):
        """
        PPO surrogate objective for updating actors.
        """
        # ========= compute loss for the discrete actor
        base_mask = torch.zeros(state.size(0), self.max_num_nodes).to(self.device)
        base_idx = torch.arange(0, self.max_num_nodes).to(self.device)
        mask = torch.where(base_idx < node_num.unsqueeze(-1), base_mask, torch.ones_like(base_mask))  # (minibatch, #max_node)

        if args.random_d:
            actor_loss_d = torch.zeros(1).to(mask.device)
            entropy_d = torch.zeros(1)
            approx_kl_d = torch.zeros(1)
            ens_pen_d = torch.zeros(1)
            ens_action_d = action_d
        else:
            logprob_new_d, entropy_d = self.actor_ds[a_idx].logprob_entropy(state, action_d, mask)  # (minibatch,), (minibatch,)
            ratio_d = torch.exp(logprob_new_d - logprob_old_d)  # a/b = exp(log(a)-log(b))
            surr1_d = ratio_d * adv
            surr2_d = torch.clamp(ratio_d, 1 - self.eps_clip_d, 1 + self.eps_clip_d) * adv
            if self.ensemble_num > 1 and self.penalty_alpha_d > 0:
                ens_pen_d, ens_action_d = self._ensemble_penalty_d(a_idx, state, mask)  # (minibatch,)
            else:
                ens_pen_d = torch.zeros(1).to(self.device)
            actor_loss_d = -torch.min(surr1_d, surr2_d) - self.coeff_ent_d * entropy_d + ens_pen_d  # (minibatch,)
            with torch.no_grad():
                approx_kl_d = ((ratio_d - 1) - (logprob_new_d - logprob_old_d)).mean()

        # ========= calculate loss for the continuous actor
        if args.random_c:
            actor_loss_c = torch.zeros(1).to(mask.device)
            entropy_c = torch.zeros(1)
            ens_pen_c = torch.zeros(1)
            approx_kl_c = torch.zeros(1)
            dist_c = None
        else:
            logprob_new_c, entropy_c, dist_c = self.actor_cs[a_idx].logprob_entropy(state, action_d, action_c)  # (minibatch,), (minibatch,)
            ratio_c = torch.exp(logprob_new_c - logprob_old_c)
            surr1_c = ratio_c * adv
            surr2_c = torch.clamp(ratio_c, 1-self.eps_clip_c, 1+self.eps_clip_c) * adv
            if self.ensemble_num > 1 and self.penalty_alpha_c > 0:
                ens_pen_c = self._ensemble_penalty_c(a_idx, state, ens_action_d)
            else:
                ens_pen_c = torch.zeros(1).to(self.device)
            actor_loss_c = -torch.min(surr1_c, surr2_c) + ens_pen_c  # (batchprompt,)
            with torch.no_grad():
                approx_kl_c = ((ratio_c - 1) - (logprob_new_c - logprob_old_c)).mean()

        return actor_loss_d.mean(), actor_loss_c.mean(), entropy_d.mean().item(), entropy_c.mean().item(), \
               ens_pen_d.mean().item(), ens_pen_c.mean().item(), approx_kl_d.item(), approx_kl_c.item(), \
               dist_c.loc.min().item() if dist_c is not None else None, \
               dist_c.loc.max().item() if dist_c is not None else None, \
               dist_c.scale.mean().item() if dist_c is not None else None


    def _critic_loss(self, state, ret):
        """
        TD-Error for updating the critic.
        """
        state_value = self.critic(state).squeeze(-1)  # (minibatch_promptnum, )
        critic_loss = F.mse_loss(ret, state_value)

        return critic_loss * self.coeff_critic
    

    def train_policy(self, args, a_idx, experience, node_num, optim, ret0, scaled_reward, batch_id, total_batch):
        """
        Update critic and two actors of the `agent` for seveal times after several episodes are collected.
        """
        state, action_d, logprob_old_d, action_c, logprob_old_c, adv, ret = experience
        node_num = torch.stack([x for x in node_num for _ in range(x)])
        # [Trick] advantage normalization (considering the whole batch rather than minibatch)
        adv = ((adv - adv.mean()) / (adv.std() + 1e-8))
        
        actor_loss_d_batch, actor_loss_c_batch, critic_loss_batch, ens_pen_d_batch, ens_pen_c_batch, \
        entropy_d_batch, entropy_c_batch, kl_d_batch, kl_c_batch, p_min_batch, p_max_batch, p_std_batch = [0 for _ in range(12)]

        minibatch_num = int(len(state) / args.minibatch_size)
        update_num_cnt = 0
        for _ in range(args.policy_update_nums):
            index = torch.randperm(len(state))
            kl_d_minibatch, kl_c_minibatch = 0, 0

            for minibatch in range(minibatch_num):
                idx = index[minibatch*args.minibatch_size : (minibatch+1)*args.minibatch_size]
                actor_loss_d, actor_loss_c, ent_d, ent_c, ens_pen_d, ens_pen_c, kl_d, kl_c, p_min, p_max, p_std = \
                    self._actor_loss(args, a_idx, state[idx], action_d[idx], action_c[idx], adv[idx], logprob_old_d[idx], logprob_old_c[idx], node_num[idx])
                kl_d_minibatch += kl_d
                kl_c_minibatch += kl_c
                critic_loss = self._critic_loss(state[idx], ret[idx])

                optim.zero_grad()
                loss = actor_loss_d + actor_loss_c + critic_loss
                loss.backward()
                # [Trick] gradient clip
                torch.nn.utils.clip_grad_norm_(list(self.actor_ds[a_idx].parameters()) +
                                               list(self.actor_cs[a_idx].parameters()) +
                                               list(self.critic.parameters()), 
                                               norm_type=2, max_norm=self.max_norm_grad)
                optim.step()

                if not args.sh_mode:
                    actor_loss_d_batch += actor_loss_d.item()
                    actor_loss_c_batch += actor_loss_c.item()
                    critic_loss_batch += critic_loss.item()
                    p_min_batch += p_min
                    p_max_batch += p_max
                    p_std_batch += p_std
                    ens_pen_d_batch += ens_pen_d
                    ens_pen_c_batch += ens_pen_c
                    # entropy_d_batch += entropy_d
                    # entropy_c_batch += entropy_c
                    kl_d_batch += kl_d
                    kl_c_batch += kl_c

            update_num_cnt += 1
            if args.target_kl_d is not None:
                if kl_d_minibatch/minibatch_num > args.target_kl_d:
                    break
            if args.target_kl_c is not None:
                if kl_c_minibatch/minibatch_num > args.target_kl_c:
                    break
        
        if not args.sh_mode:
            # sr_sum_list = [sr.sum().item() for sr in scaled_reward]
            # sr_sum_mean = sum(sr_sum_list) / len(sr_sum_list)
            scr = torch.cat(scaled_reward)
            scr_min, scr_max, scr_median, scr_std = scr.min().item(), scr.max().item(), scr.median().item(), scr.std().item()
            # Approx_return0 info
            ret0_min, ret0_max, ret0_median, ret0_std = ret0.min().item(), ret0.max().item(), ret0.median().item(), ret0.std().item()
            # Advantage info
            # adv_min, adv_max = adv.min().item(), adv.max().item()
            
            print("[P{} B{:02d}/{}] Loss {:.3f} {:.3f} {:.2f} | PEN {:.1e} {:.1e} | KL {:.1e} {:.1e} | r {:.1f} {:.1f} ±{:.2f} | "
                "Ret0 {:.1f} {:.1f} ±{:.2f} | M {:.1e} {:.1e} {:.4e}".format(
                a_idx+1, batch_id, total_batch,
                actor_loss_d_batch / minibatch_num / update_num_cnt, actor_loss_c_batch / minibatch_num / update_num_cnt, critic_loss_batch / minibatch_num / update_num_cnt,
                ens_pen_d_batch / minibatch_num / update_num_cnt, ens_pen_c_batch / minibatch_num / update_num_cnt, 
                kl_d_batch / minibatch_num / update_num_cnt, kl_c_batch / minibatch_num / update_num_cnt,
                scr_min, scr_max, scr_std, ret0_min, ret0_max, ret0_std,
                p_min_batch / minibatch_num / update_num_cnt, p_max_batch / minibatch_num / update_num_cnt, p_std_batch / minibatch_num / update_num_cnt))
