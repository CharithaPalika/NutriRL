import torch
import numpy as np
from tqdm.autonotebook import tqdm
from matplotlib import pyplot as plt
import warnings
from collections import deque, namedtuple
from .base_agent import BaseAgent
from models import QNetwork
import pandas as pd
from utils import EnvData, choose_action

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False





# =========================================================
# n-Step Replay Buffer
# =========================================================
class NStepReplayBuffer:
    def __init__(self, capacity, n_step, gamma):
        self.buffer = deque(maxlen=capacity)
        self.n_step = n_step
        self.gamma = gamma
        self.n_queue = deque(maxlen=n_step)

    def push(self, transition):
        self.n_queue.append(transition)

        if len(self.n_queue) < self.n_step:
            return

        R, next_obs, done = 0.0, None, False
        for i, (_, _, r, nobs, d) in enumerate(self.n_queue):
            R += (self.gamma ** i) * r
            next_obs, done = nobs, d
            if d:
                break

        obs, act, _, _, _ = self.n_queue[0]
        self.buffer.append((obs, act, R, next_obs, done))

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        obs, act, rew, next_obs, done = zip(*(self.buffer[i] for i in idx))
        return (
            np.array(obs),
            np.array(act),
            np.array(rew),
            np.array(next_obs),
            np.array(done),
        )

    def __len__(self):
        return len(self.buffer)


# =========================================================
# DDQN Agent
# =========================================================
class DDQNAgent(BaseAgent):
    """
    Double DQN with n-step returns using action-conditioned QNetwork
    """

    def __init__(self, env, device="cpu", **kwargs):
        super().__init__(env, device=device, **kwargs)

        # ---------------- Defaults ----------------
        defaults = dict(
            gamma=0.99,
            n_step=5,
            buffer_size=300_000,
            max_q_clip=500.0,
            max_grad_norm=10.0,
        )

        unknown = set(kwargs) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown init args: {unknown}")

        self.args = {**defaults, **kwargs}

        self.gamma = self.args["gamma"]
        self.n_step = self.args["n_step"]
        self.buffer_size = self.args["buffer_size"]
        self.max_q_clip = self.args["max_q_clip"]
        self.max_grad_norm = self.args["max_grad_norm"]

        self.env = env
        self.device = device

        # ---------------- Env specs ----------------
        try:
            self.state_dim = env.observation_space["physiological_state"].shape[0]
            self.food_dim = env.observation_space["food_embedding"].shape[0]
            self.act_dim = env.action_space.n
        except Exception as e:
            raise ValueError("Environment observation space mismatch") from e

        # ---------------- Networks ----------------
        self.q_net = QNetwork(
            num_states=self.state_dim,
            food_embedding_size=self.food_dim,
            action_size=self.act_dim,
        ).to(self.device)

        self.target_q_net = QNetwork(
            num_states=self.state_dim,
            food_embedding_size=self.food_dim,
            action_size=self.act_dim,
        ).to(self.device)

        self.target_q_net.load_state_dict(self.q_net.state_dict())

        # ---------------- Optimizer ----------------
        self.optimizer = None

        self.replay_buffer = NStepReplayBuffer(
            self.args["buffer_size"], self.n_step, self.gamma
        )

        # self.epsilon = self.eps_start
        self.total_steps = 0


    # =====================================================
    # Helpers
    # =====================================================

    def _init_optimizer(self):
        self.optimizer = torch.optim.Adam(
                self.q_net.parameters(), lr=self.lr_start
            )

    def _one_hot(self, action_idx, batch_size):
        a = torch.zeros(batch_size, self.act_dim, device=self.device)
        a.scatter_(1, action_idx.unsqueeze(1), 1.0)
        return a


    # =====================================================
    # Action selection
    # =====================================================
    def act(self, obs, deterministic=False):
        phys = torch.tensor(
            obs["physiological_state"], dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        food = torch.tensor(
            obs["food_embedding"], dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        if (not deterministic) and np.random.rand() < self.epsilon:
            action = self.env.action_space.sample()
            q_val = None
        else:
            with torch.no_grad():
                q_vals = []
                for a in range(self.act_dim):
                    a_onehot = torch.zeros(1, self.act_dim, device=self.device)
                    a_onehot[0, a] = 1.0
                    q = self.q_net(phys, food, a_onehot)
                    q_vals.append(q.item())
                action = int(np.argmax(q_vals))
                q_val = torch.tensor(q_vals[action], device=self.device)

        return (
            torch.tensor(action),
            None,
            q_val if q_val is not None else torch.tensor(0.0),
            None,
        )


    # =====================================================
    # Rollout
    # =====================================================
    def generate_episode(self, log_wandb=False, episode_idx=None):
        
        env = self.env
        memory = deque(maxlen=env.max_steps)
        env.reset()
        curr_obs = env._get_obs()
        done = False
        target = np.array(env._target_location)
        actions = []
        rewards = []
        q_values = []
        curr_phy_states = []
        distances = []

        while not done:

            action_t, _, q_val, _ = self.act(curr_obs)
            action = action_t.item()

            next_obs, reward, terminated, done, info = env.step(action)
            actions.append(action)
            rewards.append(reward)
            q_values.append(q_val.item() if q_val is not None else 0.0)
            phy = curr_obs["physiological_state"]
            curr_phy_states.append([
                phy[0] * env.Carbs_norm,
                phy[1] * env.Fat_norm,
                phy[2] * env.Protein_norm,
            ])
            distances.append(np.linalg.norm(phy - target, ord=1))

            memory.append(
                EnvData(
                    torch.tensor(curr_obs["physiological_state"]).unsqueeze(0),
                    torch.tensor(curr_obs["food_embedding"]).unsqueeze(0),
                    action,
                    reward,
                    torch.tensor(next_obs["physiological_state"]).unsqueeze(0),
                    torch.tensor(next_obs["food_embedding"]).unsqueeze(0),
                    None,
                    None,
                    q_val,
                    0,
                    0,
                    0,
                )
            )

            curr_obs = next_obs


        if log_wandb and not WANDB_AVAILABLE:
            warnings.warn("Wandb not available, not logging episode data", 
                          RuntimeWarning)
            

        if log_wandb and WANDB_AVAILABLE:
            inference_df = self.infer_episode(memory)
            self._log_episode(
                episode_idx,
                actions,
                rewards,
                q_values,
                curr_phy_states,
                distances,
                env,
                inference_df
            )
        return memory

    def _log_episode(
        self,
        episode_idx,
        actions,
        rewards,
        q_values,
        curr_phy_states,
        distances,
        env,
        inference_df,
    ):
        actions = np.array(actions)
        rewards = np.array(rewards)
        q_values = np.array(q_values)
        curr_phy_states = np.array(curr_phy_states)
        distances = np.array(distances)
        steps = np.arange(len(actions))
        action_table = wandb.Table(columns=["timestep", "action"])
        for i, a in enumerate(actions):
            action_table.add_data(i, a)
        wandb.log(
            {
                f"inference_{episode_idx}/actions": wandb.plot.scatter(
                    action_table,
                    x="timestep",
                    y="action",
                    title="Action selection (| = eat / no-eat)",
                ),
                f"inference_{episode_idx}/reward": wandb.plot.line_series(
                    xs=steps,
                    ys=[rewards],
                    keys=["Reward"],
                    title="Reward across episode",
                    xname="Timestep",
                ),
                f"inference_{episode_idx}/q value": wandb.plot.line_series(
                    xs=steps,
                    ys=[q_values],
                    keys=["Q(s)"],
                    title="Critic value",
                    xname="Timestep",
                ),

                f"inference_{episode_idx}/macros": wandb.plot.line_series(
                    xs=steps,
                    ys=[
                        curr_phy_states[:, 0],
                        curr_phy_states[:, 1],
                        curr_phy_states[:, 2],
                    ],
                    keys=["Carbs", "Fat", "Protein"],
                    title="Physiological state",
                    xname="Timestep",
                ),
                f"inference_{episode_idx}/distance_to_target": wandb.plot.line_series(
                    xs=steps,
                    ys=[distances],
                    keys=["L1 distance"],
                    title="Homeostatic error",
                    xname="Timestep",
                ),
            }
        )
        fig = env.plot_consumption(max_time=50)
        wandb.log({f"inference_{episode_idx}/consumption_plot": wandb.Image(fig)})
        plt.close(fig)
        wandb.log(
            {
                f"inference_{episode_idx}/Companison table": wandb.Table(
                    dataframe=inference_df
                )
            }
        )

    
    def infer_episode(self, memory):
        "Extracts a single episode from memory and infers the overall carbs, fat, and protein Vs targets"
        carbs = np.array([m.curr_phy_state[:,0].item() for m in memory]) * self.env.Carbs_norm
        fats = np.array([m.curr_phy_state[:,1].item() for m in memory]) * self.env.Fat_norm
        proteins = np.array([m.curr_phy_state[:,2].item() for m in memory]) * self.env.Protein_norm
        target_carbs = self.env._target_location[0] * self.env.Carbs_norm
        target_fats = self.env._target_location[1] * self.env.Fat_norm
        target_proteins = self.env._target_location[2] * self.env.Protein_norm
        target = [target_carbs, target_fats, target_proteins]
        actual = [carbs[-1], fats[-1], proteins[-1]]
        ingredients = ["carbs", "fats", "proteins"]
        df = pd.DataFrame({"Ingredients": ingredients, "Target": target, "Actual": actual})
        return df

    # =====================================================
    # Update
    # =====================================================
    def _update(self):
        obs, act, rew, next_obs, done = self.replay_buffer.sample(
            self.batch_size
        )

        phys = torch.tensor(
            [o["physiological_state"] for o in obs],
            dtype=torch.float32,
            device=self.device,
        )
        food = torch.tensor(
            [o["food_embedding"] for o in obs],
            dtype=torch.float32,
            device=self.device,
        )

        next_phys = torch.tensor(
            [o["physiological_state"] for o in next_obs],
            dtype=torch.float32,
            device=self.device,
        )
        next_food = torch.tensor(
            [o["food_embedding"] for o in next_obs],
            dtype=torch.float32,
            device=self.device,
        )

        act_idx = torch.tensor(act, dtype=torch.long, device=self.device)
        act_onehot = self._one_hot(act_idx, act_idx.size(0))

        rew = torch.tensor(rew, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            q_next = []
            for a in range(self.act_dim):
                a_onehot = torch.zeros(
                    next_phys.size(0), self.act_dim, device=self.device
                )
                a_onehot[:, a] = 1.0
                q = self.q_net(next_phys, next_food, a_onehot)
                q_next.append(q.squeeze(1))

            q_next = torch.stack(q_next, dim=1)
            next_actions = q_next.argmax(dim=1)

            q_target = []
            for a in range(self.act_dim):
                a_onehot = torch.zeros(
                    next_phys.size(0), self.act_dim, device=self.device
                )
                a_onehot[:, a] = 1.0
                q = self.target_q_net(next_phys, next_food, a_onehot)
                q_target.append(q.squeeze(1))

            q_target = torch.stack(q_target, dim=1)
            next_q = q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            target_q = rew + (self.gamma ** self.n_step) * (1 - done) * next_q
            target_q = target_q.clamp(-self.max_q_clip, self.max_q_clip)

        q = self.q_net(phys, food, act_onehot).squeeze(1)

        loss = torch.nn.functional.smooth_l1_loss(q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return loss.item()


    # =====================================================
    def _update_lr(self):
        frac = min(1.0, self.total_steps / self.eps_decay_steps)
        lr = self.lr_start + frac * (self.lr_end - self.lr_start)
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    # =====================================================
    def train(self, log_wandb=False, printing=True, **kwargs):
        train_defaults = dict(num_episodes=20000,
                              min_replay_size=10000,
                              batch_size=128,
                              train_every=8,
                              target_update_every=8000,
                              eps_start=1.0,
                              eps_end=0.01,
                              eps_decay_steps=200_000,
                              lr_start=1e-3,
                              lr_end=1e-4,)
        unknown = set(kwargs) - set(train_defaults)
        if unknown:
            raise ValueError(f"Unknown train args: {unknown}")
        train_args = {**train_defaults, **kwargs}
        # self.n_step = train_args["n_step"]
        self.batch_size = train_args["batch_size"]
        self.train_every = train_args["train_every"]
        self.target_update_every = train_args["target_update_every"]
        self.min_replay_size = train_args["min_replay_size"]
        self.eps_start = train_args["eps_start"]
        self.epsilon = self.eps_start
        self.eps_end = train_args["eps_end"]
        self.eps_decay_steps = train_args["eps_decay_steps"]
        self.lr_start = train_args["lr_start"]
        self.lr_end = train_args["lr_end"]
        self._init_optimizer()
        returns = []
        eat_counts = []

        for ep in tqdm(range(train_args["num_episodes"])):
            curr_obs , _ = self.env.reset()
            # curr_obs = self.env._get_obs()
            done = False
            ep_returns = 0
            ep_eats = 0
            ep_losses = []
            actions, rewards, q_values = [], [], []
            while not done:
                self.total_steps += 1
                self.epsilon = max(
                    self.eps_end,
                    self.eps_start - self.total_steps / self.eps_decay_steps,
                )
                action_t, _, q_val, _ = self.act(curr_obs)
                action = action_t.item()
                next_obs, reward, terminated, done, info = self.env.step(action)
                self.replay_buffer.push(
                    (curr_obs, action, reward, next_obs, done)
                )

                curr_obs = next_obs
                ep_returns += reward
                ep_eats += action
                if (
                    len(self.replay_buffer) >= self.min_replay_size
                    and self.total_steps % self.train_every == 0
                ):
                    loss_val = self._update()
                    ep_losses.append(loss_val)

                    self._update_lr()
                if self.total_steps % self.target_update_every == 0:
                    self.target_q_net.load_state_dict(self.q_net.state_dict())

            eat_counts.append(ep_eats)
            returns.append(ep_returns)
            if printing and (ep + 1) % 100 == 0:
                print(
                    f"ep={ep+1} | "
                    f"mean return={np.mean(returns[-100:]):.2f} | "
                    f"mean eats={np.mean(eat_counts[-100:]):.2f}"
                )

            if log_wandb:
                wandb.log(
                    {
                        "train/return": ep_returns,
                        "train/foods_eaten": ep_eats,
                        "train/episode_loss": np.mean(ep_losses) if len(ep_losses) > 0 else 0.0,


                    },
                    step=ep,
                )


        return returns, eat_counts

    def save_model(self, path, log_wandb=False):
        torch.save(self.q_net.state_dict(), path + "_q_network.pt")
        if log_wandb and not WANDB_AVAILABLE:
            warnings.warn(
                "wandb logging requested but wandb is not installed. "
                "Continuing without saving artifacts.",
                RuntimeWarning,
            )
        if log_wandb and WANDB_AVAILABLE:
            artifact = wandb.Artifact("DDQNAgent", type="model")
            artifact.add_file(path + "_q_network.pt")
            wandb.log_artifact(artifact)


    def load_model(self, path):
        self.q_net.load_state_dict(torch.load(path + "_q_network.pt"))
