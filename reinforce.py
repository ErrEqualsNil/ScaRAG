import argparse
import datetime
import os

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from EnvSimulator.Env_simple import Env


def get_float_tensor_from_key(transition, name):
    return torch.FloatTensor(np.array([t[name] for t in transition])).to("cuda")


class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, mode="categorical", output_dim=5):
        super().__init__()
        self.mode = mode
        self.state_linear1 = nn.Linear(input_dim - 768, hidden_dim // 4)
        self.state_linear2 = nn.Linear(hidden_dim // 4, hidden_dim // 4)

        self.linear1 = nn.Linear(768 + hidden_dim // 4, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        if self.mode == "categorical":
            self.linear3 = nn.Linear(hidden_dim, 2 ** output_dim)
        else:
            self.linear3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, state):
        if state.dim() == 1:
            state = state.view(1, -1)
        s_state = state[:, 768:]
        s_feat = torch.tanh(self.state_linear1(s_state))
        s_feat = torch.tanh(self.state_linear2(s_feat))
        feat = torch.concat([s_feat, state[:, :768]], dim=-1)
        feat = torch.tanh(self.linear1(feat))
        feat = torch.tanh(self.linear2(feat))
        if self.mode == "categorical":
            logits = self.linear3(feat)
            logits = logits.masked_fill(self.mask, float('-inf'))
            action_prob = torch.softmax(logits, dim=-1)
        else:
            action_prob = torch.sigmoid(self.linear3(feat))
        return action_prob.squeeze(0)


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=2048):
        super().__init__()
        self.state_linear1 = nn.Linear(input_dim - 768, hidden_dim // 4)
        self.state_linear2 = nn.Linear(hidden_dim // 4, hidden_dim // 4)

        self.linear1 = nn.Linear(768 + hidden_dim // 4, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        s_state = state[:, 768:]
        s_feat = torch.tanh(self.state_linear1(s_state))
        s_feat = torch.tanh(self.state_linear2(s_feat))
        feat = torch.concat([s_feat, state[:, :768]], dim=-1)
        feat = torch.tanh(self.linear1(feat))
        feat = torch.tanh(self.linear2(feat))
        feat = torch.tanh(self.linear3(feat))
        value = self.linear4(feat)
        return value


class Agent:
    def __init__(self, args):
        self.args = args

        self.gae_lambda = 0.98
        self.reuse_times = args.reuse_times
        self.CLIP = 0.2
        self.mini_batch_size = args.mini_batch_size

        self.actor = Actor(768 + self.args.num_shard * 3, mode=self.args.mode, output_dim=self.args.num_shard).to("cuda")
        self.critic = Critic(768 + self.args.num_shard * 3).to("cuda")

        self.optim = torch.optim.Adam([
                {'params': self.actor.parameters(), "lr": args.a_lr, 'initial_lr': args.a_lr},
                {'params': self.critic.parameters(), "lr": args.c_lr, 'initial_lr': args.c_lr},
            ], eps=1e-5
        )

    @torch.no_grad()
    def make_action(self, state):
        state = torch.FloatTensor(state).to("cuda")

        action_prob = self.actor(state)
        if self.actor.mode == "categorical":
            action_dist = torch.distributions.Categorical(action_prob)
            action = action_dist.sample().item()
        else:
            action_dist = torch.distributions.Bernoulli(action_prob)
            action = action_dist.sample().cpu().numpy().tolist()

        return action, action_prob.cpu().numpy()

    def learn(self, transition, total_steps):
        current_time, state, action, action_prob, reward, next_state, next_time = (
            get_float_tensor_from_key(transition, "current_time"), get_float_tensor_from_key(transition, "state"), get_float_tensor_from_key(transition, "action"),
            get_float_tensor_from_key(transition, "action_prob"), get_float_tensor_from_key(transition, "reward"), get_float_tensor_from_key(transition, "next_state"),
            get_float_tensor_from_key(transition, "next_time")
        )
        current_time, next_time, reward = current_time.reshape(-1, 1), next_time.reshape(-1, 1), reward.reshape(-1, 1)

        with torch.no_grad():
            adv = []
            gae = 0
            vs = self.critic(state)
            vs_ = self.critic(next_state)
            deltas = reward + torch.exp(-self.args.a * (next_time - current_time)) * vs_ - vs
            for delta, n_time, c_time in zip(
                    reversed(deltas.flatten().cpu().numpy()),
                    reversed(next_time.flatten().cpu().numpy()),
                    reversed(current_time.flatten().cpu().numpy())
            ):
                gae = delta + np.exp(-self.args.a * (n_time - c_time)) * self.gae_lambda * gae
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1).to("cuda")
            v_target = adv + vs
            adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        actor_loss_list = np.zeros(self.reuse_times)
        critic_loss_list = np.zeros(self.reuse_times)
        entropy_list = np.zeros(self.reuse_times)
        kl_div_list = np.zeros(self.reuse_times)
        count = np.zeros(self.reuse_times)
        for t in range(self.reuse_times):
            for index in BatchSampler(SubsetRandomSampler(range(len(transition))), self.mini_batch_size, False):
                count[t] += 1
                index = torch.tensor(index)

                new_action_prob = self.actor(state[index])
                if self.actor.mode == "categorical":
                    new_action_dist = torch.distributions.Categorical(new_action_prob)
                    old_action_dist = torch.distributions.Categorical(action_prob[index])
                    new_action_log_prob = new_action_dist.log_prob(action[index].squeeze())
                    old_action_log_prob = old_action_dist.log_prob(action[index].squeeze())
                    ratio = torch.exp(new_action_log_prob - old_action_log_prob).unsqueeze(1)
                    surr1 = ratio * adv[index]
                    surr2 = torch.clamp(ratio, 1 - self.CLIP, 1 + self.CLIP) * adv[index]
                    actor_loss = (
                            - torch.min(surr1, surr2)
                            - self.args.entropy_coefficient * new_action_dist.entropy().unsqueeze(-1)
                    ).mean()

                    entropy_list[t] += (
                        new_action_dist.entropy().unsqueeze(1)
                    ).mean().cpu().detach().numpy()
                    kl_div_list[t] += (
                        torch.distributions.kl_divergence(new_action_dist, old_action_dist)
                    ).mean().cpu().detach().numpy()
                else:
                    new_action_dist = torch.distributions.Bernoulli(new_action_prob)
                    old_action_dist = torch.distributions.Bernoulli(action_prob[index])
                    new_action_log_prob = new_action_dist.log_prob(action[index].squeeze())
                    old_action_log_prob = old_action_dist.log_prob(action[index].squeeze())
                    ratio = torch.exp(new_action_log_prob - old_action_log_prob).mean(dim=-1, keepdim=True)

                    surr1 = ratio * adv[index]
                    surr2 = torch.clamp(ratio, 1 - self.CLIP, 1 + self.CLIP) * adv[index]
                    actor_loss = (
                        - torch.min(surr1, surr2)
                        - self.args.entropy_coefficient * new_action_dist.entropy().mean(dim=-1, keepdim=True)
                    ).mean()

                    entropy_list[t] += (
                        new_action_dist.entropy().mean(dim=-1, keepdim=True)
                    ).mean().cpu().detach().numpy()
                    kl_div_list[t] += (
                        torch.distributions.kl_divergence(new_action_dist, old_action_dist).mean(dim=-1, keepdim=True)
                    ).mean().cpu().detach().numpy()

                actor_loss_list[t] += actor_loss.cpu().detach().numpy()

                v_s = self.critic(state[index])
                critic_loss = torch.nn.functional.mse_loss(v_s, v_target[index])

                loss = actor_loss + critic_loss

                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optim.step()

                critic_loss_list[t] += critic_loss.cpu().detach().numpy()

        self.lr_decay(total_steps)
        return np.average(actor_loss_list / count), np.average(critic_loss_list / count), np.average(entropy_list / count), np.average(kl_div_list / count)

    def lr_decay(self, total_steps):
        for p in self.optim.param_groups:
            p['lr'] = p['initial_lr'] * (1 - total_steps / self.args.epoch)

    def save(self, save_folder_path, step_num):
        os.makedirs(save_folder_path, exist_ok=True)
        torch.save(self.actor.state_dict(), f"{save_folder_path}/actor_{step_num}.pth")
        torch.save(self.critic.state_dict(), f"{save_folder_path}/critic_{step_num}.pth")

    def load(self, save_folder_path, step_num):
        self.actor.load_state_dict(torch.load(f"{save_folder_path}/actor_{step_num}.pth"))
        self.critic.load_state_dict(torch.load(f"{save_folder_path}/critic_{step_num}.pth"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--need_log", default=True, type=bool)
    parser.add_argument("--name", default="geq_0.2", type=str)

    parser.add_argument("--epoch", default=200, type=int)
    parser.add_argument("--a", default=0.1, type=float)
    parser.add_argument("--reuse_times", default=8, type=int)
    parser.add_argument("--a_lr", default=1e-4, type=float)
    parser.add_argument("--c_lr", default=1e-3, type=float)
    parser.add_argument("--mini_batch_size", default=256, type=int)
    parser.add_argument("--num_task_in_episode", default=1024, type=int)
    parser.add_argument("--mode", default="Bernoulli", type=str)

    parser.add_argument("--tolerable_delay", default=2, type=float)
    parser.add_argument("--task_generate_interval_mean", default=0.2, type=float)

    parser.add_argument("--entropy_coefficient", default=0.005, type=float)
    parser.add_argument("--seed", default=-1, type=int)

    parser.add_argument("--weight", default=0.3, type=float)

    return parser.parse_args()


def main():
    args = parse_args()
    args.repeat_idx = [1, 2, 3, 4, 5]
    name = f"{args.name}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"

    args.num_shard = 5
    env = Env(
        "data_geq/train/qn_tensors.pt", args,
        qc_cache_path="data_geq/train/qc_cache_dict.bin",
        shard_path="data_geq/train/geq_[0-9].jsonl",
    )
    agent = Agent(args)

    update_step = 1
    while update_step <= args.epoch:
        if args.seed != -1:
            env.reset(seed=args.seed + update_step)
        else:
            env.reset(seed=update_step)

        while len(env.events) != 0:
            isScheduledRequired, state, retrieval_time_cost, network_delays = env.process_event()
            if isScheduledRequired:
                action, action_prob = agent.make_action(state)
                env.make_action(state, action, action_prob, retrieval_time_cost, network_delays)

        env.process_transitions()
        # train
        actor_loss, critic_loss, entropy, kl_div = agent.learn(env.transitions, update_step)
        performances = env.get_performance()
        log_dict = {
            "Training_Metric/ActorLoss": actor_loss,
            "Training_Metric/CriticLoss": critic_loss,
            "Training_Metric/Entropy": entropy,

            "Training_Metric/KL_Div": kl_div,

            "Result/Reward": performances[0],
            "Result/EstimatedSupportiveDocuments": performances[1],
            "Result/RespAccuracy":  performances[2],

            "Result/RetrieveOverhead": performances[3],
            "Result/invalid_action_penalty": performances[4],
        }
        print(actor_loss, critic_loss, performances[0])

        # save model
        if update_step % 50 == 0 and update_step != 0 and args.need_log:
            agent.save(f"data_geq/model_outputs/{name}", update_step)
        update_step += 1


if __name__ == '__main__':
    main()
