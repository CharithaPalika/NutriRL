import torch
import numpy as np
from tqdm.autonotebook import tqdm
from matplotlib import pyplot as plt
import warnings
from collections import deque, defaultdict, namedtuple
from utils import choose_action, EnvData
from .base_agent import BaseAgent 
from models import SharedActorCritic, Actor, Critic
import pandas as pd
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class MCAgent(BaseAgent):
    """
    Monte-Carlo Actor-Critic Agent
    Supports:
        - Shared Actor-Critic
        - Separate Actor / Critic
    """

    def __init__(self, env, 
                 device = "cpu", **kwargs):
        super().__init__(env, device=device, **kwargs)

        defaults = dict(
            gamma=0.99,
            lam=0.95,
            shared=True,
            shared_ac_network=None,
            actor_network=None,
            critic_network=None,
        )

        unknown = set(kwargs) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown init args: {unknown}")

        self.args = {**defaults, **kwargs}

        # Architectural checks 
        if self.args["shared"]:
            if (
                self.args["actor_network"] is not None
                or self.args["critic_network"] is not None
            ):
                raise ValueError(
                    "Invalid configuration: shared=True but actor_network "
                    "or critic_network was provided. "
                    "Use shared_ac_network instead."
                )
        else:
            if self.args["shared_ac_network"] is not None:
                raise ValueError(
                    "Invalid configuration: shared=False but shared_ac_network "
                    "was provided. Use actor_network / critic_network instead."
                )

        # params
        self.gamma = self.args["gamma"]
        self.lam = self.args["lam"]
        self.env = env

        # environment specs
        try:
            action_shape = env.action_space.n
            state_shape = env.observation_space["physiological_state"].shape[0]
            food_embed_size = env.observation_space["food_embeddings"].shape[-1]
        except Exception as e:
            raise ValueError(
                "Environment observation/action space does not match expected format."
            ) from e

        # last action index is "skip"; eating slot i is action i.
        self.skip_action = env.num_foods

        # loading models
        if self.args["shared"]:
            self.policy = (
                self.args["shared_ac_network"]
                if self.args["shared_ac_network"] is not None
                else SharedActorCritic(
                    food_embedding_size=food_embed_size,
                    num_states=state_shape,
                    num_actions=action_shape,
                )
            ).to(self.device)

            self.actor = None
            self.critic = None

        else:
            self.actor = (
                self.args["actor_network"]
                if self.args["actor_network"] is not None
                else Actor(
                    food_embedding_size=food_embed_size,
                    num_states=state_shape,
                    num_actions=action_shape,
                )
            ).to(self.device)

            self.critic = (
                self.args["critic_network"]
                if self.args["critic_network"] is not None
                else Critic(
                    food_embedding_size=food_embed_size,
                    num_states=state_shape,
                )
            ).to(self.device)

            self.policy = None

        # Optimizers
        self.policy_optimizer = None
        self.actor_optimizer = None
        self.critic_optimizer = None


    def _init_optimizers(self, actor_lr, critic_lr, shared_ac_lr):

        if self.args["shared"]:
            self.policy_optimizer = torch.optim.Adam(
                self.policy.parameters(), lr=shared_ac_lr
            )
        else:
            self.actor_optimizer = torch.optim.Adam(
                self.actor.parameters(), lr=actor_lr
            )
            self.critic_optimizer = torch.optim.Adam(
                self.critic.parameters(), lr=critic_lr
            )

    def train(self, log_wandb=False, printing=True, **kwargs):
        train_defaults = dict(
            num_episodes=20000,
            shared_ac_lr=1e-4,
            actor_lr=1e-4,
            critic_lr=1e-4,
        )
        unknown = set(kwargs) - set(train_defaults)
        if unknown:
            raise ValueError(f"Unknown train args: {unknown}")
        train_args = {**train_defaults, **kwargs}
        self._init_optimizers(
            actor_lr=train_args["actor_lr"],
            critic_lr=train_args["critic_lr"],
            shared_ac_lr=train_args["shared_ac_lr"],
        )
        score_monitor = []
        food_pick_monitor = []
        if log_wandb and not WANDB_AVAILABLE:
            warnings.warn(
                "wandb logging requested but wandb is not installed. "
                "Continuing training without logging.",
                RuntimeWarning,
            )
            log_wandb = False
        for ep in tqdm(range(train_args["num_episodes"])):
           memory = self.generate_episode(
               log_wandb=False,
               episode_idx=ep,
           )
           rewards = [m.reward for m in memory]
           values = [m.value.squeeze() for m in memory]
           ep_return = sum(rewards)

           # count steps where the agent actually ate something (any slot != skip)
           ep_eats = sum(1 for m in memory if m.action != self.skip_action)
           
           score_monitor.append(ep_return)
           food_pick_monitor.append(ep_eats)
           advantages, returns = self._compute_gae(
               rewards=rewards,
               values=values,
               gamma=self.gamma,
               lam=self.lam
           )
           actor_loss = []
           value_loss = []
           for t, m in enumerate(memory):
            adv = advantages[t].detach()
            actor_loss.append(-m.log_probs * adv)
            td_err = returns[t] - m.value.squeeze()
            value_loss.append(td_err ** 2)
            
           actor_loss_val = torch.stack(actor_loss).sum()
           value_loss_val = torch.stack(value_loss).sum()
           total_loss = actor_loss_val + 0.5 * value_loss_val
           # backpropagate and update networks
           if self.args["shared"]:
               # update policy network
               self.policy_optimizer.zero_grad()
               total_loss.backward()
               self.policy_optimizer.step()
           else:
               # update actor
               self.actor_optimizer.zero_grad()
               actor_loss_val.backward()
               self.actor_optimizer.step()
               # update critic
               self.critic_optimizer.zero_grad()
               value_loss_val.backward()
               self.critic_optimizer.step()
               
           if log_wandb:
               wandb.log({
                   "train/return": score_monitor[-1],
                   "train/foods_eaten": food_pick_monitor[-1],
                   "train/loss": total_loss.item(),
                   "train/actor_loss": actor_loss_val.item(),
                   "train/value_loss": value_loss_val.item(),
               }, step=ep,)

           if printing and (ep + 1) % 25 == 0:
               print(
                   f"ep={ep+1} | "
                   f"mean return={np.mean(score_monitor[-100:]):.2f} | "
                   f"mean eats={np.mean(food_pick_monitor[-100:]):.2f}"
               )

        return score_monitor, food_pick_monitor

    def _compute_gae(self,rewards, values, gamma=0.99, lam=0.95):
        T = len(rewards)
        advantages = torch.zeros(T)
        returns = torch.zeros(T)

        gae = 0.0
        next_value = 0.0

        for t in reversed(range(T)):
            r = rewards[t]
            v = values[t].squeeze()

            delta = r + gamma * next_value - v
            gae = delta + gamma * lam * gae

            advantages[t] = gae
            returns[t] = gae + v
            next_value = v.detach()

        return advantages, returns

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
            obs["food_embeddings"],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        if self.args["shared"]:
            action_logits, value = self.policy(curr_phy_state, curr_food_state)
        else:
            action_logits = self.actor(curr_phy_state, curr_food_state)
            value = self.critic(curr_phy_state, curr_food_state)

        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
            log_prob = None
        else:
            action, log_prob = choose_action(action_logits)

        return action, log_prob, value, action_logits


    def generate_episode(self, log_wandb=False, episode_idx=None):
        env = self.env

        memory = deque(maxlen=env.max_steps)
        env.reset()
        curr_obs = env._get_obs()
        done = False

        target = np.array(env._target_location)

        actions = []
        rewards = []
        values = []
        action_logits_all = []
        curr_phy_states = []
        distances = []

        while not done:
            action_t, log_prob, value, action_logits = self.act(curr_obs)

            # env requires python int
            action_env = action_t.item()

            next_obs, reward, terminated, done, info = env.step(action_env)

            phy = curr_obs["physiological_state"]

            actions.append(action_env)
            rewards.append(reward)
            values.append(value.item())
            action_logits_all.append(
                action_logits.squeeze(0).detach().cpu().numpy()
            )

            curr_phy_states.append((phy * env.nutrient_norms).tolist())

            distances.append(np.linalg.norm(phy - target, ord=1))

            memory.append(
                EnvData(
                    torch.tensor(curr_obs["physiological_state"]).unsqueeze(0),
                    torch.tensor(curr_obs["food_embeddings"]).unsqueeze(0),
                    action_env,
                    reward,
                    torch.tensor(next_obs["physiological_state"]).unsqueeze(0),
                    torch.tensor(next_obs["food_embeddings"]).unsqueeze(0),
                    action_logits,
                    log_prob,
                    value,
                    0,
                    0,
                    done
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
                values,
                action_logits_all,
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
        values,
        action_logits_all,
        curr_phy_states,
        distances,
        env,
        inference_df
    ):
        actions = np.array(actions)
        rewards = np.array(rewards)
        values = np.array(values)
        action_logits_all = np.array(action_logits_all)
        curr_phy_states = np.array(curr_phy_states)
        distances = np.array(distances)
        steps = np.arange(len(actions))
    
        action_table = wandb.Table(columns=["timestep", "action"])
        for i, a in enumerate(actions):
            action_table.add_data(i, a)

        action_logit_labels = [f"Eat slot {i}" for i in range(env.num_foods)] + ["Skip"]
    
        wandb.log({
            f"inference_{episode_idx}/actions": wandb.plot.scatter(
                action_table,
                x="timestep",
                y="action",
                title="Action selection (slot index, last = skip)",
            ),
            f"inference_{episode_idx}/reward": wandb.plot.line_series(
                xs=steps,
                ys=[rewards],
                keys=["Reward"],
                title="Reward across episode",
                xname="Timestep",
            ),
            f"inference_{episode_idx}/value": wandb.plot.line_series(
                xs=steps,
                ys=[values],
                keys=["V(s)"],
                title="Critic value",
                xname="Timestep",
            ),
            f"inference_{episode_idx}/action_logits": wandb.plot.line_series(
                xs=steps,
                ys=[action_logits_all[:, i] for i in range(action_logits_all.shape[1])],
                keys=action_logit_labels,
                title="Action logits",
                xname="Timestep",
            ),
            f"inference_{episode_idx}/macros": wandb.plot.line_series(
                xs=steps,
                ys=[curr_phy_states[:, i] for i in range(curr_phy_states.shape[1])],
                keys=env.nutrient_names,
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
        })
    
        fig = env.plot_consumption(max_time=50)
        wandb.log({
            f"inference_{episode_idx}/consumption_plot": wandb.Image(fig)
        })
        plt.close(fig)
        # log inference df table
        wandb.log({
            f"inference_{episode_idx}/Companison table": wandb.Table(dataframe=inference_df)
        })
    
    def infer_episode(self, memory):
        """Extracts a single episode from memory and compares final nutrient
        levels against targets, generically over however many nutrients the
        env defines."""
        phy_states = np.array([
            (m.curr_phy_state.squeeze(0).numpy() * self.env.nutrient_norms)
            for m in memory
        ])
        target = (self.env._target_location * self.env.nutrient_norms)
        actual = phy_states[-1]
        df = pd.DataFrame({
            "Ingredients": self.env.nutrient_names,
            "Target": target,
            "Actual": actual,
        })
        return df

    def save_model(self, path, log_wandb = False):
        if self.args["shared"]:
            torch.save(self.policy.state_dict(), path + "_shared.pt")
        else:
            torch.save(self.actor.state_dict(), path + "_actor.pt")
            torch.save(self.critic.state_dict(), path + "_critic.pt")

        if log_wandb and not WANDB_AVAILABLE:
            warnings.warn(
                "wandb logging requested but wandb is not installed. "
                "Continuing without saving artifacts.",
                RuntimeWarning,
            )
        if log_wandb and WANDB_AVAILABLE:
            artifact = wandb.Artifact("MCAgent", type="model")
            if self.args["shared"]:
                artifact.add_file(path+ "_shared.pt")
            else:
                artifact.add_file(path + "_actor.pt")
                artifact.add_file(path + "_critic.pt")
            
            wandb.log_artifact(artifact)

    def load_model(self, path):
        if self.args["shared"]:
            self.policy.load_state_dict(torch.load(path+ "_shared.pt"))
        else:
            self.actor.load_state_dict(torch.load(path+ "_actor.pt"))
            self.critic.load_state_dict(torch.load(path + "_critic.pt"))
