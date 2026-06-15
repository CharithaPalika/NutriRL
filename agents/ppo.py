import torch
import numpy as np
from tqdm.autonotebook import tqdm
from matplotlib import pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import warnings
import copy
from collections import deque, defaultdict, namedtuple
from utils import choose_action, ReplayBuffer, EnvData
from .base_agent import BaseAgent
from models import Actor, Critic, SharedActorCritic
import pandas as pd
from torch.distributions import Categorical


try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False




class PPOAgent(BaseAgent):
    def __init__(self, env, device="cpu", **kwargs):
        super().__init__(env, device=device, **kwargs)
        defaults = dict(
            gamma=0.99,
            lam=0.95,
            clip_eps=0.2,
            value_coeff=0.5,
            entropy_coeff=0.01,
            max_grad_norm=0.5,
            shared=True,
            shared_ac_network=None,
            actor_network=None,
            critic_network=None,
            seed = None
        )

        unknown = set(kwargs) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown init args: {unknown}")

        self.args = {**defaults, **kwargs}
        self.device = device

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

        self.env = env
        # environment specs
        try:
            action_shape = env.action_space.n
            state_shape = env.observation_space["physiological_state"].shape[0]
            food_embed_size = env.observation_space["food_embedding"].shape[0]
        except Exception as e:
            raise ValueError(
                "Environment observation/action space does not match expected format."
            ) from e

        # loading models
        if self.args["shared"]:
            self.policy = (
                self.args["shared_ac_network"]
                if self.args["shared_ac_network"] is not None
                else SharedActorCritic(
                    food_embedding_size=food_embed_size,
                    num_states=state_shape,
                    num_actions=action_shape,
                    seed = self.args["seed"]
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
                    seed = self.args["seed"]
                )
            ).to(self.device)

            self.critic = (
                self.args["critic_network"]
                if self.args["critic_network"] is not None
                else Critic(
                    food_embedding_size=food_embed_size,
                    num_states=state_shape,
                    seed = self.args["seed"]
                )
            ).to(self.device)

            self.policy = None

        # Optimizers
        self.policy_optimizer = None
        self.actor_optimizer = None
        self.critic_optimizer = None

        self.episode_returns = []
        self.food_eaten = []
        self.device = device

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


    def _compute_gae(self, rewards, values, dones):
        advs = np.zeros_like(rewards)
        gae = 0.0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.args["gamma"] * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.args["gamma"] * self.args["lam"] * (1 - dones[t]) * gae
            advs[t] = gae

        returns = advs + values[:-1]
        return advs, returns
    

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


    def train(self, log_wandb = False, printing = True, **kwargs):

        train_defaults = dict(
            num_episodes=10000,
            shared_ac_lr=1e-3,
            actor_lr=1e-3,
            critic_lr=1e-3,
            rollout_steps=512,
            ppo_epochs = 5,
            minibatch_size = 64,
            log_every_episodes = 50,
            rolling_window = 50,
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

        if log_wandb and not WANDB_AVAILABLE:
            warnings.warn(
                "wandb logging requested but wandb is not installed. "
                "Continuing training without logging.",
                RuntimeWarning,
            )
            log_wandb = False

        episode_actor_losses = []
        episode_critic_losses = []
        episode_entropies = []
        episode_total_losses = []

        obs, _ = self.env.reset()
        ep_return = 0
        episode_count = 0
        total_foods_eaten = 0
        while episode_count < train_args["num_episodes"]:
            phy_obs_buf, food_embedding_buff, act_buf, rew_buf, done_buf = [], [], [], [], []
            logp_buf, val_buf = [], []

             # ---------------- Rollout ----------------
            for _ in range(train_args["rollout_steps"]):
                phy_obs = torch.tensor(obs["physiological_state"],dtype=torch.float32,device=self.device,)
                food_embedding = torch.tensor(obs["food_embedding"],dtype=torch.float32,device=self.device,)
                action, logp, value, action_logits = self.act(obs, deterministic=False)
                action = action.squeeze()
                logp = logp.squeeze()
                value = value.squeeze(-1).squeeze()
                
                if action.item() == 1: # eating
                    total_foods_eaten += 1
                next_obs, reward, terminated, done, info = self.env.step(action.item())
                
                phy_obs_buf.append(phy_obs)
                food_embedding_buff.append(food_embedding)
                act_buf.append(action)
                rew_buf.append(reward)
                done_buf.append(done)
                logp_buf.append(logp.detach())
                val_buf.append(value.detach())


                ep_return += reward
                obs = next_obs
                if done:
                    self.episode_returns.append(ep_return)
                    self.food_eaten.append(total_foods_eaten)
                    episode_count += 1
                    ep_return = 0
                    total_foods_eaten = 0
                    obs, _ = self.env.reset()
                    # -------- Episode-based logging --------
                    if episode_count % train_args["log_every_episodes"] == 0:
                        rolling_avg = np.mean(
                            self.episode_returns[-train_args["rolling_window"]:]
                        )
                        last_return = self.episode_returns[-1]
                        last_tot_food_eaten = self.food_eaten[-1]
                        avg_food_eaten = np.mean(
                            self.food_eaten[-train_args["rolling_window"]:]
                        )

                        if printing:
                            print(
                                f"Episode {episode_count:4d} | "
                                f"Last Return: {last_return:7.1f} | "
                                f"Rolling Avg(50): {rolling_avg:7.1f} |"
                                f"Foods eating: {last_tot_food_eaten:7.1f} |"
                                f"Rolling food eaten avg: {avg_food_eaten:7.1f}"
                            )

                        # if log_wandb:
                            # wandb.log({
                                # "train/last_return": last_return,
                                # "train/Avg return": rolling_avg,
                                # "train/last_eaten_foods": last_tot_food_eaten,
                                # "train/avg_food_eaten": avg_food_eaten,
                            # })

                        if log_wandb:
                            wandb.log(
                                {
                                    "train/return": self.episode_returns[-1],
                                    "train/foods_eaten": self.food_eaten[-1],
                                    "train/actor_loss": np.mean(episode_actor_losses),
                                    "train/critic_loss": np.mean(episode_critic_losses),
                                    "train/entropy": np.mean(episode_entropies),
                                    "train/total_loss": np.mean(episode_total_losses),
                                },
                                step=episode_count,
                            )

                    if episode_count >= train_args["num_episodes"]:
                        break
                
            # Bootstrap value
            with torch.no_grad():
                if self.args["shared"]:
                    action_logits, val = self.policy(torch.tensor(obs["physiological_state"], dtype=torch.float32, device=self.device).unsqueeze(0),
                                                    torch.tensor(obs["food_embedding"], dtype=torch.float32, device=self.device).unsqueeze(0))
                else:
                    action_logits = self.actor(torch.tensor(obs["physiological_state"], dtype=torch.float32, device=self.device).unsqueeze(0),
                                                    torch.tensor(obs["food_embedding"], dtype=torch.float32, device=self.device).unsqueeze(0))
                    val = self.critic(torch.tensor(obs["physiological_state"], dtype=torch.float32, device=self.device).unsqueeze(0),
                                                    torch.tensor(obs["food_embedding"], dtype=torch.float32, device=self.device).unsqueeze(0))
                
                
                val_buf.append(val.squeeze(-1).squeeze())
               
            # ---------------- Compute GAE ----------------
            advs, rets = self._compute_gae(
                np.array(rew_buf),
                torch.stack(val_buf).cpu().numpy(),
                np.array(done_buf, dtype=np.float32),
            )
            advs = torch.tensor(advs, dtype=torch.float32, device=self.device)
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            rets = torch.tensor(rets, dtype=torch.float32, device=self.device)
            # print(f'advs shape: {advs.shape}, ret shape: {rets.shape}')
            phy_obs_batch = torch.stack(phy_obs_buf)
            food_embedding_batch = torch.stack(food_embedding_buff)
            act_batch = torch.stack(act_buf)
            logp_old = torch.stack(logp_buf)

            # ---------------- PPO Update ----------------
            for _ in range(train_args["ppo_epochs"]):
                idx = torch.randperm(len(phy_obs_batch))

                for start in range(0, len(phy_obs_batch), train_args["minibatch_size"]):
                    mb = idx[start:start + train_args["minibatch_size"]]

                    # Slice minibatch
                    phy_mb = phy_obs_batch[mb]
                    food_mb = food_embedding_batch[mb]

                    if self.args["shared"]:
                        action_logits , val = self.policy(phy_mb, food_mb) 
                    else:
                        action_logits = self.actor(phy_mb, food_mb)
                        val = self.critic(phy_mb, food_mb)

                    # _, _, val, action_logits = self.act(obs, deterministic=False) #TODO : fix this
                    dist = Categorical(logits = action_logits)
                    logp = dist.log_prob(act_batch[mb])
                    entropy = dist.entropy().mean()
                    
                    ratio = torch.exp(logp - logp_old[mb])
                    surr1 = ratio * advs[mb]
                    surr2 = torch.clamp(
                        ratio, 1 - self.args["clip_eps"], 1 + self.args["clip_eps"]
                    ) * advs[mb]
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = (
                        (val.squeeze() - rets[mb]) ** 2
                    ).mean()                 

                    if self.args["shared"]:
                        actor_loss = (
                            policy_loss
                            - self.args["entropy_coeff"] * entropy)

                        critic_loss = value_loss
                        loss = actor_loss + self.args["value_coeff"] * critic_loss
                        # loss = (
                                # policy_loss
                                # + self.args["value_coeff"] * value_loss
                                # - self.args["entropy_coeff"] * entropy
                            # )
                        episode_actor_losses.append(actor_loss.item())
                        episode_critic_losses.append(critic_loss.item())
                        episode_entropies.append(entropy.item())
                        episode_total_losses.append(loss.item())

                        # if log_wandb:
                            # wandb.log(
                                # {
                                    # "train/total loss": loss,
                                    # "train/actor_loss": actor_loss,
                                    # "train/critic_loss": critic_loss,
                                    # "train/entropy": entropy,
                                # }
                            # )
                        self.policy_optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self.args["max_grad_norm"])
                        self.policy_optimizer.step()
                    
                    else:
                        actor_loss = (
                            policy_loss
                            - self.args["entropy_coeff"] * entropy
                        )
                    
                        # Critic loss (ONLY value loss)
                        critic_loss = value_loss #self.args["value_coeff"] * value_loss
                        episode_actor_losses.append(actor_loss.item())
                        episode_critic_losses.append(critic_loss.item())
                        episode_entropies.append(entropy.item())
                        episode_total_losses.append(0)

                        # if log_wandb:
                            # wandb.log(
                                # {
                                    # "train/actor_loss": actor_loss,
                                    # "train/critic_loss": critic_loss,
                                    # "train/entropy": entropy,
                                # }
                            # )
                        # --- Actor update ---
                        self.actor_optimizer.zero_grad()
                        actor_loss.backward(retain_graph=True)
                        nn.utils.clip_grad_norm_(
                            self.actor.parameters(), self.args["max_grad_norm"]
                        )
                        self.actor_optimizer.step()
                    
                        # --- Critic update ---
                        self.critic_optimizer.zero_grad()
                        critic_loss.backward()
                        nn.utils.clip_grad_norm_(
                            self.critic.parameters(), self.args["max_grad_norm"]
                        )
                        self.critic_optimizer.step()

                    
        return self.episode_returns, self.food_eaten 
    


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
                    action_env,
                    reward,
                    torch.tensor(next_obs["physiological_state"]).unsqueeze(0),
                    torch.tensor(next_obs["food_embedding"]).unsqueeze(0),
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
    
        wandb.log({
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
            f"inference_{episode_idx}/value": wandb.plot.line_series(
                xs=steps,
                ys=[values],
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

    # def save_model(self, path, log_wandb = False):
        # if self.args["shared"]:
            # torch.save(self.policy.state_dict(), path + "_shared.pt")
        # else:
            # torch.save(self.actor.state_dict(), path + "_actor.pt")
            # torch.save(self.critic.state_dict(), path + "_critic.pt")
# 
        # if log_wandb and not WANDB_AVAILABLE:
            # warnings.warn(
                # "wandb logging requested but wandb is not installed. "
                # "Continuing without saving artifacts.",
                # RuntimeWarning,
            # )
        # else:
            # artifact = wandb.Artifact("PPOAgent", type="model")
            # if self.args["shared"]:
                # artifact.add_file(path+ "_shared.pt")
            # else:
                # artifact.add_file(path + "_actor.pt")
                # artifact.add_file(path + "_critic.pt")
            # 
            # wandb.log_artifact(artifact)
# 

    def load_model(self, path):
        if self.args["shared"]:
            self.policy.load_state_dict(torch.load(path+ "_shared.pt"))
        else:
            self.actor.load_state_dict(torch.load(path+ "_actor.pt"))
            self.critic.load_state_dict(torch.load(path + "_critic.pt"))



    def save_model(self, path, log_wandb=False):
        if self.args["shared"]:
            torch.save(self.policy.state_dict(), path + "_shared.pt")
        else:
            torch.save(self.actor.state_dict(), path + "_actor.pt")
            torch.save(self.critic.state_dict(), path + "_critic.pt")

        if not log_wandb:
            return

        if not WANDB_AVAILABLE:
            warnings.warn(
                "wandb logging requested but wandb is not installed. "
                "Continuing without saving artifacts.",
                RuntimeWarning,
            )
            return
        if log_wandb and WANDB_AVAILABLE:
            artifact = wandb.Artifact("PPOAgent", type="model")
            
            if self.args["shared"]:
                artifact.add_file(path + "_shared.pt")
            else:
                artifact.add_file(path + "_actor.pt")
                artifact.add_file(path + "_critic.pt")
    
            wandb.log_artifact(artifact)


