import torch
import numpy as np
from tqdm.autonotebook import tqdm
from matplotlib import pyplot as plt
import torch.nn.functional as F
import warnings
import copy
from collections import deque, defaultdict, namedtuple
from utils import choose_action, ReplayBuffer, EnvData
from .base_agent import BaseAgent
from models import Actor, QNetwork
import pandas as pd

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False



import torch
import numpy as np
from tqdm.autonotebook import tqdm
from matplotlib import pyplot as plt
import torch.nn.functional as F
import warnings
import copy
from collections import deque, defaultdict, namedtuple
from utils import choose_action, ReplayBuffer, EnvData
from .base_agent import BaseAgent
from models import Actor, QNetwork
import pandas as pd

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class SACAgent(BaseAgent):
    """
    Soft Actor-Critic Agent
    Supports:
        - Separate Actor / Q-network networks
    """

    def __init__(self, env, device="cpu", **kwargs):
        super().__init__(env, device=device, **kwargs)

        defaults = dict(
            gamma=0.99,
            alpha=0.2,
            tau=0.005,
            buffer_size=100000,
            actor_network=None,
            q_network=None,
        )
        self.device = device

        unknown = set(kwargs) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown init args: {unknown}")

        self.args = {**defaults, **kwargs}

        self.gamma = self.args["gamma"]
        self.alpha = self.args["alpha"]
        self.tau = self.args["tau"]
        self.env = env

        try:
            self.action_shape = env.action_space.n
            state_shape = env.observation_space["physiological_state"].shape[0]
            food_embed_size = env.observation_space["food_embedding"].shape[0]
        except Exception as e:
            raise ValueError(
                "Environment observation/action space does not match expected format."
            ) from e

        self.actor = (
            self.args["actor_network"]
            if self.args["actor_network"] is not None
            else Actor(
                food_embedding_size=food_embed_size,
                num_states=state_shape,
                num_actions=self.action_shape,
            )
        ).to(self.device)

        self.q1 = (
            copy.deepcopy(self.args["q_network"])
            if self.args["q_network"] is not None
            else QNetwork(
                food_embedding_size=food_embed_size,
                num_states=state_shape,
                action_size=self.action_shape,
            )
        ).to(self.device)

        self.q2 = copy.deepcopy(self.q1).to(self.device)

        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)

        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False

        self.actor_optimizer = None
        self.q1_optimizer = None
        self.q2_optimizer = None

        self.memory_buffer = ReplayBuffer(self.args["buffer_size"])
        self.total_steps = 0

        # Pre-allocate action indices tensor (used repeatedly)
        self.action_range = torch.arange(
            self.action_shape, device=self.device
        ).float().unsqueeze(-1)

    def _init_optimizers(self, actor_lr, q_network_lr):
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=q_network_lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=q_network_lr)

    @torch.no_grad()
    def soft_update(self, net, target_net):
        for p, tp in zip(net.parameters(), target_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def train(self, log_wandb=False, printing=True, **train_args):

        defaults = dict(
            num_episodes=2500,
            actor_lr=3e-4,
            q_network_lr=3e-4,
            batch_size=64,
        )
        episode_entropies = []
        training_args = {**defaults, **train_args}
        if set(train_args) - set(defaults):
            raise ValueError(f"Unknown train args: {set(train_args) - set(defaults)}")

        num_episodes = training_args["num_episodes"]
        self.batch_size = training_args["batch_size"]

        self._init_optimizers(
            actor_lr=training_args["actor_lr"],
            q_network_lr=training_args["q_network_lr"],
        )

        score_monitor = []
        food_pick_monitor = []

        episode_q1_losses = []
        episode_q2_losses = []
        episode_policy_losses = []
        episode_rewards = []
        episode_foods = []

        for i in tqdm(range(num_episodes)):
            episode_q1_losses.clear()
            episode_q2_losses.clear()
            episode_policy_losses.clear()

            curr_obs, _ = self.env.reset()
            done = False
            ep_rew = 0
            num_eats = 0

            while not done:

                if i >= 5 and log_wandb:
                    wandb.log(
                        {
                            "train/return": episode_rewards[-1],
                            "train/foods_eaten": episode_foods[-1],
                            "train/q1_loss": np.mean(episode_q1_losses),
                            "train/q2_loss": np.mean(episode_q2_losses),
                            "train/actor_loss": np.mean(episode_policy_losses),
                            "train/entropy": np.mean(episode_entropies),
                        },
                        step=i - 5,
                    )

                curr_phy_state = torch.tensor(
                    curr_obs["physiological_state"],
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)

                curr_food_state = torch.tensor(
                    curr_obs["food_embedding"],
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)

                with torch.no_grad():
                    action_logits = self.actor(curr_phy_state, curr_food_state)
                    action, log_prob = choose_action(action_logits)

                    q1_vals = self.q1(
                        curr_phy_state,
                        curr_food_state,
                        action.float().unsqueeze(1),
                    )
                    q2_vals = self.q2(
                        curr_phy_state,
                        curr_food_state,
                        action.float().unsqueeze(1),
                    )

                next_obs, reward, terminated, done, info = self.env.step(action.item())
                done_flag = terminated or done

                ep_rew += reward
                num_eats += int(action.item() == 1)

                next_phy_state = torch.tensor(
                    next_obs["physiological_state"],
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)

                next_food_state = torch.tensor(
                    next_obs["food_embedding"],
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)

                self.memory_buffer.push(
                    EnvData(
                        curr_phy_state,
                        curr_food_state,
                        action,
                        reward,
                        next_phy_state,
                        next_food_state,
                        action_logits,
                        log_prob,
                        0,
                        q1_vals,
                        q2_vals,
                        done_flag,
                    )
                )

                curr_obs = next_obs

                if done_flag:
                    episode_rewards.append(ep_rew)
                    episode_foods.append(num_eats)

                if len(self.memory_buffer) >= self.batch_size:
                    batch_memory = self.memory_buffer.sample(self.batch_size)

                    batch_curr_phy_states = torch.cat(
                        [m.curr_phy_state for m in batch_memory], dim=0
                    )
                    batch_curr_food_states = torch.cat(
                        [m.curr_food_state for m in batch_memory], dim=0
                    )
                    batch_actions = torch.tensor(
                        [m.action for m in batch_memory],
                        dtype=torch.float32,
                        device=self.device,
                    ).unsqueeze(1)

                    batch_rewards = torch.tensor(
                        [m.reward for m in batch_memory],
                        dtype=torch.float32,
                        device=self.device,
                    ).unsqueeze(1)

                    batch_next_phy_states = torch.cat(
                        [m.next_phy_state for m in batch_memory], dim=0
                    )
                    batch_next_food_states = torch.cat(
                        [m.next_food_state for m in batch_memory], dim=0
                    )

                    batch_dones = torch.tensor(
                        [m.done for m in batch_memory],
                        dtype=torch.float32,
                        device=self.device,
                    ).unsqueeze(1)

                    with torch.no_grad():
                        next_logits = self.actor(
                            batch_next_phy_states, batch_next_food_states
                        )
                        next_probs = F.softmax(next_logits, dim=1)
                        next_log_probs = F.log_softmax(next_logits, dim=1)

                        B, A = next_probs.shape

                        phy_exp = batch_next_phy_states.unsqueeze(1).expand(-1, A, -1)
                        food_exp = batch_next_food_states.unsqueeze(1).expand(-1, A, -1)
                        actions = self.action_range.expand(B, -1, -1)

                        q1_vals = self.q1_target(
                            phy_exp.reshape(-1, phy_exp.size(-1)),
                            food_exp.reshape(-1, food_exp.size(-1)),
                            actions.reshape(-1, 1),
                        ).view(B, A)

                        q2_vals = self.q2_target(
                            phy_exp.reshape(-1, phy_exp.size(-1)),
                            food_exp.reshape(-1, food_exp.size(-1)),
                            actions.reshape(-1, 1),
                        ).view(B, A)

                        min_q = torch.min(q1_vals, q2_vals)
                        v_next = (next_probs * (min_q - self.alpha * next_log_probs)).sum(
                            dim=1, keepdim=True
                        )

                        q_target = batch_rewards + (1 - batch_dones) * self.gamma * v_next

                    q1_current = self.q1(
                        batch_curr_phy_states, batch_curr_food_states, batch_actions
                    )
                    q2_current = self.q2(
                        batch_curr_phy_states, batch_curr_food_states, batch_actions
                    )

                    q1_loss = F.mse_loss(q1_current, q_target)
                    q2_loss = F.mse_loss(q2_current, q_target)

                    self.q1_optimizer.zero_grad()
                    q1_loss.backward()
                    self.q1_optimizer.step()

                    self.q2_optimizer.zero_grad()
                    q2_loss.backward()
                    self.q2_optimizer.step()

                    action_logits = self.actor(
                        batch_curr_phy_states, batch_curr_food_states
                    )
                    action_probs = F.softmax(action_logits, dim=1)
                    log_action_probs = F.log_softmax(action_logits, dim=1)
                    

                    B, A = action_probs.shape
                    phy_exp = batch_curr_phy_states.unsqueeze(1).expand(-1, A, -1)
                    food_exp = batch_curr_food_states.unsqueeze(1).expand(-1, A, -1)
                    actions = self.action_range.expand(B, -1, -1)

                    q1_vals = self.q1(
                        phy_exp.reshape(-1, phy_exp.size(-1)),
                        food_exp.reshape(-1, food_exp.size(-1)),
                        actions.reshape(-1, 1),
                    ).view(B, A)

                    q2_vals = self.q2(
                        phy_exp.reshape(-1, phy_exp.size(-1)),
                        food_exp.reshape(-1, food_exp.size(-1)),
                        actions.reshape(-1, 1),
                    ).view(B, A)

                    min_q_vals = torch.min(q1_vals, q2_vals)

                    policy_loss = (
                        action_probs
                        * (self.alpha * log_action_probs - min_q_vals.detach())
                    ).sum(dim=1).mean()
                    entropy = -(action_probs * log_action_probs).sum(dim=1).mean()
                    self.actor_optimizer.zero_grad()
                    policy_loss.backward()
                    self.actor_optimizer.step()

                    episode_entropies.append(entropy.item())
                    episode_q1_losses.append(q1_loss.item())
                    episode_q2_losses.append(q2_loss.item())
                    episode_policy_losses.append(policy_loss.item())

                    self.soft_update(self.q1, self.q1_target)
                    self.soft_update(self.q2, self.q2_target)

                    self.total_steps += 1

            if i % 25 == 0 and printing:
                ep_memory = self.generate_episode()
                episode_df = self.infer_episode(ep_memory)

                actions = np.array([item.action for item in ep_memory])
                rewards = np.array([float(item.reward) for item in ep_memory])
                eat_steps = np.where(actions == 1)[0]

                avg_reward = np.sum(rewards)
                score_monitor.append(avg_reward)
                food_pick_monitor.append(len(eat_steps))

                print(
                    f"Episode {i}: Average Reward: {avg_reward}, Foods Picked: {len(eat_steps)}"
                )

        return score_monitor, food_pick_monitor

    # ---- ALL OTHER FUNCTIONS BELOW ARE UNCHANGED ----
    # act, generate_episode, _log_episode, infer_episode,
    # save_model, load_model


    def act(self, obs, deterministic=False):
        """
        Given an observation, returns:
            action (tensor), log_prob, value, action_logits
        """
        curr_phy_state = torch.tensor(
            obs["physiological_state"],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        curr_food_state = torch.tensor(
            obs["food_embedding"],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        action_logits = self.actor(curr_phy_state, curr_food_state)

        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
            log_prob = None
        else:
            action, log_prob = choose_action(action_logits)

        q1_vals = self.q1(
            curr_phy_state,
            curr_food_state,
            torch.tensor(action, dtype=torch.float32).unsqueeze(dim=1),
        )
        q2_vals = self.q2(
            curr_phy_state,
            curr_food_state,
            torch.tensor(action, dtype=torch.float32).unsqueeze(dim=1),
        )

        return action, log_prob, q1_vals, q2_vals, action_logits

    def generate_episode(self, log_wandb=False, episode_idx=None):
        env = self.env
        memory = deque(maxlen=env.max_steps)
        env.reset()
        curr_obs = env._get_obs()
        done = False
        target = np.array(env._target_location)
        actions = []
        rewards = []
        q1_values = []
        q2_values = []
        action_logits_all = []
        curr_phy_states = []
        distances = []

        while not done:
            action_t, log_prob, q1, q2, action_logits = self.act(curr_obs)
            # env requires python int
            action_env = action_t.item()
            next_obs, reward, terminated, done, info = env.step(action_env)
            phy = curr_obs["physiological_state"]
            actions.append(action_env)
            rewards.append(reward)
            q1_values.append(q1.item())
            q2_values.append(q2.item())

            action_logits_all.append(action_logits.squeeze(0).detach().cpu().numpy())
            curr_phy_states.append(
                [
                    phy[0] * env.Carbs_norm,
                    phy[1] * env.Fat_norm,
                    phy[2] * env.Protein_norm,
                ]
            )
            distances.append(np.linalg.norm(phy - target, ord=1))
            memory.append(
                EnvData(
                    torch.tensor(curr_obs["physiological_state"]).unsqueeze(0),
                    torch.tensor(curr_obs["food_embedding"]).unsqueeze(0),
                    action_env,
                    reward,
                    torch.tensor(next_obs["physiological_state"]).unsqueeze(0),
                    torch.tensor(next_obs["food_embedding"]).unsqueeze(0),
                    action_logits,
                    log_prob,
                    0,
                    q1,
                    q2,
                    done
                )
            )
            curr_obs = next_obs

        if log_wandb and not WANDB_AVAILABLE:
            warnings.warn(
                "Wandb not available, not logging episode data", RuntimeWarning
            )

        if log_wandb and WANDB_AVAILABLE:
            inference_df = self.infer_episode(memory)
            self._log_episode(
                episode_idx,
                actions,
                rewards,
                q1_values,
                q2_values,
                action_logits_all,
                curr_phy_states,
                distances,
                env,
                inference_df,
            )
        return memory

    def _log_episode(
        self,
        episode_idx,
        actions,
        rewards,
        q1_values,
        q2_values,
        action_logits_all,
        curr_phy_states,
        distances,
        env,
        inference_df,
    ):
        actions = np.array(actions)
        rewards = np.array(rewards)
        q1_values = np.array(q1_values)
        q2_values = np.array(q2_values)
        action_logits_all = np.array(action_logits_all)
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
                f"inference_{episode_idx}/q1 value": wandb.plot.line_series(
                    xs=steps,
                    ys=[q1_values],
                    keys=["V(s)"],
                    title="Critic value",
                    xname="Timestep",
                ),
                f"inference_{episode_idx}/q2 value": wandb.plot.line_series(
                    xs=steps,
                    ys=[q2_values],
                    keys=["V(s)"],
                    title="Critic value",
                    xname="Timestep",
                ),
                f"inference_{episode_idx}/action_logits": wandb.plot.line_series(
                    xs=steps,
                    ys=[
                        action_logits_all[:, 0],
                        action_logits_all[:, 1],
                    ],
                    keys=["Do not eat", "Eat"],
                    title="Action logits",
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
        carbs = (
            np.array([m.curr_phy_state[:, 0].item() for m in memory])
            * self.env.Carbs_norm
        )
        fats = (
            np.array([m.curr_phy_state[:, 1].item() for m in memory])
            * self.env.Fat_norm
        )
        proteins = (
            np.array([m.curr_phy_state[:, 2].item() for m in memory])
            * self.env.Protein_norm
        )
        target_carbs = self.env._target_location[0] * self.env.Carbs_norm
        target_fats = self.env._target_location[1] * self.env.Fat_norm
        target_proteins = self.env._target_location[2] * self.env.Protein_norm
        target = [target_carbs, target_fats, target_proteins]
        actual = [carbs[-1], fats[-1], proteins[-1]]
        ingredients = ["carbs", "fats", "proteins"]
        df = pd.DataFrame(
            {"Ingredients": ingredients, "Target": target, "Actual": actual}
        )
        return df

    def save_model(self, path, log_wandb=False):
        torch.save(self.actor.state_dict(), path + "_actor.pt")
        torch.save(self.q1.state_dict(), path + "_q1.pt")
        torch.save(self.q2.state_dict(), path + "_q2.pt")
        if log_wandb and not WANDB_AVAILABLE:
            warnings.warn(
                "wandb logging requested but wandb is not installed. "
                "Continuing without saving artifacts.",
                RuntimeWarning,
            )
        if log_wandb and WANDB_AVAILABLE:
            artifact = wandb.Artifact("SACAgent", type="model")
            artifact.add_file(path + "_actor.pt")
            artifact.add_file(path + "_q1.pt")
            artifact.add_file(path + "_q2.pt")
            wandb.log_artifact(artifact)

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(path + "_actor.pt"))
        self.q1.load_state_dict(torch.load(path + "_q1.pt"))
        self.q2.load_state_dict(torch.load(path + "_q2.pt"))
