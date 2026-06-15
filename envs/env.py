import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
import random
import os
from typing import NamedTuple, List
from matplotlib import pyplot as plt


# --------------------------------------------------
# Food representation (NO calories)
# --------------------------------------------------
class FoodItem(NamedTuple):
    carbs: float
    fat: float
    protein: float


class NutriRL(gym.Env):
    """
    NutriRL environment with:
    - Per-nutrient digestion (carbs, fat, protein)
    - Nutrient-specific delay and stddev
    - Observable but uncontrollable food availability
    - No calorie scalarization

    Action space:
        0 -> skip current food
        1 -> consume current food
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        file_path,
        **args,
    ):
        super().__init__()
        defaults = dict(
            max_steps=50,
            normalise=True,
            one_hot_embedding=True,
            embed_size=None,
            target_loc=None,
            seed=0,
            target_nutrients=["Carbs", "Fat", "Protein"],
            use_delay = True,
            progress_scale=1.0,
            comfort_scale=1.0,
            comfort_tau=5.0,
            toxicity_scale=1,
        )
        self.args = {**defaults, **args}
        unknown = set(args) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown args: {unknown}")
        
        if self.args["embed_size"] is not None and self.args["one_hot_embedding"]:
            raise ValueError("Cannot use both one-hot embedding and custom embed_size.")

        self.max_steps = self.args["max_steps"]
        self.file_path = file_path
        self.food_df = pd.read_csv(file_path)
        self.one_hot_embedding = self.args["one_hot_embedding"]
        
        if self.one_hot_embedding:
            self.embed_size = len(self.food_df)
        else:
            self.embed_size = self.args["embed_size"]


        self.nutrient_names = ["Carbs", "Fat", "Protein"]

        if self.args["target_nutrients"] is None:
            self.target_nutrients = self.nutrient_names
        else:
            self.target_nutrients = self.args["target_nutrients"]

        # Build mask
        self.nutrient_mask = np.array(
            [1 if n in self.target_nutrients else 0 for n in self.nutrient_names],
            dtype=np.float32,
        )

        self.num_items = len(self.food_df)
        self.item_list = list(self.food_df["Food"])

        self.Carbs = self.food_df["Carbs"].astype(float).values
        self.Fat = self.food_df["Fat"].astype(float).values
        self.Protein = self.food_df["Protein"].astype(float).values


        if self.args["use_delay"]:
            self.Carbs_Delay = self.food_df["Carbs_Delay"].astype(float).values
            self.Carbs_Std = self.food_df["Carbs_StdDev"].astype(float).values

            self.Fat_Delay = self.food_df["Fat_Delay"].astype(float).values
            self.Fat_Std = self.food_df["Fat_StdDev"].astype(float).values

            self.Protein_Delay = self.food_df["Protein_Delay"].astype(float).values
            self.Protein_Std = self.food_df["Protein_StdDev"].astype(float).values

        else:
            self.Carbs_Delay = np.zeros(self.num_items)
            self.Carbs_Std = np.zeros(self.num_items)

            self.Fat_Delay = np.zeros(self.num_items)
            self.Fat_Std = np.zeros(self.num_items)

            self.Protein_Delay = np.zeros(self.num_items)
            self.Protein_Std = np.zeros(self.num_items)


        # -------------------------
        # Optional normalization
        # -------------------------
        if self.args["normalise"]:
            self.Carbs_norm = np.linalg.norm(self.Carbs) or 1.0
            self.Fat_norm = np.linalg.norm(self.Fat) or 1.0
            self.Protein_norm = np.linalg.norm(self.Protein) or 1.0
        else:
            self.Carbs_norm = self.Fat_norm = self.Protein_norm = 1.0

        self.Carbs /= self.Carbs_norm
        self.Fat /= self.Fat_norm
        self.Protein /= self.Protein_norm


        # -------------------------
        # Apply nutrient mask
        # -------------------------
        self.Carbs *= self.nutrient_mask[0]
        self.Fat *= self.nutrient_mask[1]
        self.Protein *= self.nutrient_mask[2]


        # -------------------------
        # Build helpers
        # -------------------------
        self._create_food()
        self._create_food_embedding()

        # -------------------------
        # Spaces
        # -------------------------
        self.observation_space = spaces.Dict(
            {
                "physiological_state": spaces.Box(
                    low=0.0, high=4000.0, shape=(3,), dtype=np.float32
                ),
                # "food_contents": spaces.Box(
                    # low=0.0, high=1000.0, shape=(3,), dtype=np.float32
                # ),
                # "food_item": spaces.Discrete(self.num_items),
                "food_embedding": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.embed_size,), dtype=np.float32
                ),
                # "target": spaces.Box(
                    # low=0.0, high=1000.0, shape=(3,), dtype=np.float32
                # ),
            }
        )

        self.action_space = spaces.Discrete(2)

        # -------------------------
        # Seeding
        # -------------------------
        self._seed = None
        if self.args["seed"] is not None:
            self._set_seed(self.args["seed"])

        # -------------------------
        # Init episode
        # -------------------------
        self.reset(target_loc=self.args["target_loc"],seed = self.args["seed"])

    # --------------------------------------------------
    # Seeding (PUBLIC METHOD)
    # --------------------------------------------------
    def _set_seed(self, seed: int):
        """
        Fully seed the environment.
        Safe to call manually or via reset(seed=...).
        """
        self._seed = int(seed)

        # Python
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

        # NumPy
        np.random.seed(seed)

        # Torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        return seed

    # --------------------------------------------------
    # Food helpers
    # --------------------------------------------------
    def _create_food(self):
        self.food_list = {
            i: FoodItem(self.Carbs[i], self.Fat[i], self.Protein[i])
            for i in range(self.num_items)
        }

    def _create_food_embedding(self):
        self._food_embeddings = {}

        if self.one_hot_embedding:
            eye = torch.eye(self.num_items)
            for i in range(self.num_items):
                self._food_embeddings[i] = eye[i]
        else:
            emb = nn.Embedding(self.num_items, self.embed_size)
            emb.weight.requires_grad_(False)
            for i in range(self.num_items):
                self._food_embeddings[i] = emb(torch.tensor(i))

    
    def _initialise_agent(self):
        self._agent_location = np.zeros(3, dtype=np.float32)
        self._initial_location = np.zeros(3, dtype=np.float32)
        self._food_item_num = self.np_random.integers(self.num_items)
        self._digestion_processes: List[np.ndarray] = []
        self._consumption_log = []
        self._prev_distance = np.linalg.norm(self._agent_location - self._target_location, ord = 1)

    def set_target(self, target_loc=None):
        if target_loc is None:
            t = np.array([250.0, 80.0, 100.0], dtype=float)
        else:
            t = np.array(target_loc, dtype=float)

        # self._target_location = np.array(
            # [
                # t[0] / self.Carbs_norm,
                # t[1] / self.Fat_norm,
                # t[2] / self.Protein_norm,
            # ],
            # dtype=np.float32,
        # )

        self._target_location = np.array(
            [
                (t[0] / self.Carbs_norm) * self.nutrient_mask[0],
                (t[1] / self.Fat_norm) * self.nutrient_mask[1],
                (t[2] / self.Protein_norm) * self.nutrient_mask[2],
            ],
            dtype=np.float32,
        )

    def _start_digestion(self, food_index):
        nutrients = {
            "carbs": (self.Carbs[food_index], self.Carbs_Delay[food_index], self.Carbs_Std[food_index], 0),
            "fat": (self.Fat[food_index], self.Fat_Delay[food_index], self.Fat_Std[food_index], 1),
            "protein": (self.Protein[food_index], self.Protein_Delay[food_index], self.Protein_Std[food_index], 2),
        }

        for name, (amount, delay, std, idx) in nutrients.items():
            if amount <= 0 or self.nutrient_mask[idx] == 0:
                continue

            std = max(std, 0.1)
            mean = self.timepoint + delay
            horizon = int(np.ceil(6 * std)) + 1

            times = np.arange(self.timepoint + 1, self.timepoint + 1 + horizon)
            kernel = np.exp(-0.5 * ((times - mean) / std) ** 2)
            kernel /= kernel.sum()

            schedule = np.zeros((horizon, 3), dtype=np.float32)
            schedule[:, idx] = kernel * amount
            self._digestion_processes.append(schedule)

            self._consumption_log.append(
                {
                    "nutrient": name,
                    "food_index": food_index,
                    "times": times.copy(),
                    "kernel": kernel * amount,
                }
            )

    def _apply_digestion(self):
        total = np.zeros(3, dtype=np.float32)
        remaining = []

        for sched in self._digestion_processes:
            total += sched[0]
            if sched.shape[0] > 1:
                remaining.append(sched[1:])

        self._agent_location += total
        self._digestion_processes = remaining

    # --------------------------------------------------
    # Observation / reward
    # --------------------------------------------------
    def _get_full_obs(self):
        return {
            "physiological_state": self._agent_location.copy(),
            "food_contents": np.array(
                self.food_list[self._food_item_num], dtype=np.float32
            ),
            "food_item": int(self._food_item_num),
            "food_embedding": self._food_embeddings[self._food_item_num]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32),
            "target": self._target_location.copy(),
        }
    
    def _get_obs(self):
        return {
            "physiological_state": self._agent_location.copy(),
            "food_embedding": self._food_embeddings[self._food_item_num]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32),
        }

    def _get_info(self):
        return {
            "distance": float(
                np.linalg.norm(
                    self._agent_location - self._target_location, ord=1
                )
            )
        }

    def _check_overshoot(self):
        a = self._agent_location - self._initial_location
        d = self._target_location - self._initial_location
        return np.dot(a, d) > np.dot(d, d)

    
    def _get_reward(self, prev_dist):
        curr = self._agent_location
        target = self._target_location

        curr_dist = np.linalg.norm(curr - target, ord=1)

        # 1. Progress
        progress = prev_dist - curr_dist

        # 2. Comfort (homeostasis)
        comfort = -np.tanh(curr_dist / self.args["comfort_tau"])

        # 3. Overshoot toxicity
        excess = np.maximum(0.0, curr - target)
        toxicity = np.sum(excess ** 2)
        toxic_penalty = -self.args["toxicity_scale"] * np.tanh(toxicity)

        reward = (
            self.args["progress_scale"] * progress
            + self.args["comfort_scale"] * comfort 
            + toxic_penalty
        )

        return float(reward), curr_dist
    

    
    
    # def _get_reward(self):
        # dist = self._get_info()["distance"]
        # reward = self.args["reward_scale"] * math.exp(-dist) #- self.args["reward_scale"] * 0.5
        # if self._check_overshoot():
            # reward -= self.args["overshoot_penalty_scale"] * math.exp(dist * 0.1)
        # return reward
    

    
    def step(self, action):

        if action == 1:
                if self.args["use_delay"]:
                    self._start_digestion(self._food_item_num)
                else:
                    # direct physiological update without any delay or gaussian dist
                    self._agent_location[0] += self.Carbs[self._food_item_num]
                    self._agent_location[1] += self.Fat[self._food_item_num]
                    self._agent_location[2] += self.Protein[self._food_item_num]

        # Apply digestion only if delays are enabled
        if self.args["use_delay"]:
            self._apply_digestion()
    

        obs = self._get_obs()
        reward, distance = self._get_reward(self._prev_distance)
        
        self._prev_distance = distance

        self.timepoint += 1
        terminated = False
        truncated = self.timepoint >= self.max_steps

        self._food_item_num = self.np_random.integers(self.num_items)

        return obs, reward, terminated, truncated, self._get_info()

    def reset(self, seed=None, target_loc=None):
        super().reset(seed=seed)

        if seed is not None:
            self._set_seed(seed)
        self.set_target(target_loc)
        self._initialise_agent()
        self.timepoint = 0

        return self._get_obs(), {}

    def render(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(*self._target_location, c="red", label="Target")
        ax.scatter(*self._agent_location, c="blue", label="Agent")

        ax.set_xlabel("Carbs")
        ax.set_ylabel("Fat")
        ax.set_zlabel("Protein")
        ax.legend()
        plt.show()

    def plot_consumption(self, max_time=None, figsize=(12, 8)):
        if not self._consumption_log:
            if self.args["use_delay"]:
                print("No foods consumed in this episode.")
                fig  = plt.figure()
                return fig
            else:
                print("No absorptions mechanism used in this episode and ran without delays.")
                fig  = plt.figure()
                return fig

        nutrients = ["carbs", "fat", "protein"]

        if max_time is None:
            max_time = int(
                max(rec["times"][-1] for rec in self._consumption_log)
            )

        x = np.arange(0, max_time + 1)
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

        for ax, nutrient in zip(axes, nutrients):
            total = np.zeros_like(x, dtype=float)

            for rec in self._consumption_log:
                if rec["nutrient"] != nutrient:
                    continue

                arr = np.zeros_like(x, dtype=float)
                valid = rec["times"] <= max_time
                arr[rec["times"][valid].astype(int)] = rec["kernel"][valid]

                total += arr
                ax.plot(x, arr, alpha=0.4)

            ax.plot(x, total, color="black", linewidth=2, label="Total")
            ax.set_ylabel(f"{nutrient.capitalize()} absorbed")
            ax.grid(True)
            ax.legend()

        axes[-1].set_xlabel("Time (steps)")
        fig.suptitle("Per-nutrient digestion dynamics", fontsize=14)
        # plt.tight_layout()
        # plt.show()

        return fig
