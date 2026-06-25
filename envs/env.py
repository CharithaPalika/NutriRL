import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
import os
from typing import List
from matplotlib import pyplot as plt


class NutriRLVec(gym.Env):
    """
    Generalised Food environment with N nutrients and a menu of K foods.

    CSV format
    ----------
    Food, Nutrient_0, ..., Nutrient_{N-1},
          Nutrient_0_Delay, Nutrient_0_StdDev,
          Nutrient_1_Delay, Nutrient_1_StdDev, ...   (only when use_delay=True)

    Action space
    ------------
    Discrete(num_foods + 1)
        0 .. num_foods-1  -> eat the i-th food in the current menu
        num_foods         -> skip (eat nothing this step)

    Observation space
    -----------------
    {
        "physiological_state" : Box(num_nutrients,),
        "food_embeddings"      : Box(num_foods, embed_size),   # current menu
    }

    Decay
    -----
    Every step (after digestion is applied):
        state *= (1 - decay_rate)        element-wise, clipped to >= 0
    decay_rate can be a scalar or a per-nutrient array of length num_nutrients.

    Target nutrient masking
    ------------------------
    `target_nutrients` (list of nutrient names, default: all) restricts which
    nutrients contribute to the target/reward/distance computation. Nutrients
    not in this list are zeroed out of the target and reward calculations,
    mirroring the old NutriRL environment's `target_nutrients` behaviour.
    """

    metadata = {"render_modes": []}

    def __init__(self, file_path, **args):
        super().__init__()

        defaults = dict(
            num_nutrients=3,
            num_foods=5,            # menu size shown to agent each step
            max_steps=50,
            normalise=True,
            one_hot_embedding=True,
            embed_size=None,
            target_loc=None,
            target_nutrients=None,  # list of nutrient names to target; None = all
            seed=0,
            use_delay=True,
            decay_rate=0.02,        # scalar or list/array of length num_nutrients
            progress_scale=1.0,
            comfort_scale=1.0,
            comfort_tau=5.0,
            toxicity_scale=1.0,
        )
        self.args = {**defaults, **args}
        unknown = set(args) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown args: {unknown}")

        if self.args["embed_size"] is not None and self.args["one_hot_embedding"]:
            raise ValueError("Cannot use both one-hot embedding and custom embed_size.")

        # ── Dimensions ────────────────────────────────────────────────────────
        self.num_nutrients  = self.args["num_nutrients"]
        self.num_foods      = self.args["num_foods"]
        self.max_steps      = self.args["max_steps"]
        self.nutrient_names = [f"Nutrient_{i}" for i in range(self.num_nutrients)]

        # ── Target-nutrient mask (restores old NutriRL behaviour) ──────────────
        if self.args["target_nutrients"] is None:
            self.target_nutrients = list(self.nutrient_names)
        else:
            self.target_nutrients = list(self.args["target_nutrients"])
            unknown_targets = set(self.target_nutrients) - set(self.nutrient_names)
            if unknown_targets:
                raise ValueError(
                    f"target_nutrients contains unknown nutrient names: {unknown_targets}. "
                    f"Valid names are {self.nutrient_names}."
                )

        self.nutrient_mask = np.array(
            [1.0 if n in self.target_nutrients else 0.0 for n in self.nutrient_names],
            dtype=np.float32,
        )

        # ── Decay rate vector ─────────────────────────────────────────────────
        dr = np.array(self.args["decay_rate"], dtype=np.float32)
        if dr.ndim == 0:
            dr = np.full(self.num_nutrients, float(dr), dtype=np.float32)
        if dr.shape != (self.num_nutrients,):
            raise ValueError(
                f"decay_rate must be scalar or length {self.num_nutrients}, got {dr.shape}"
            )
        self.decay_rate = dr  # shape (num_nutrients,)

        # ── Load CSV ──────────────────────────────────────────────────────────
        self.file_path = file_path
        self.food_df   = pd.read_csv(file_path)
        self.num_items = len(self.food_df)
        self.item_list = list(self.food_df["Food"])

        if self.num_foods > self.num_items:
            raise ValueError(
                f"num_foods={self.num_foods} exceeds available food items ({self.num_items})."
            )

        # ── Embedding size ────────────────────────────────────────────────────
        self.one_hot_embedding = self.args["one_hot_embedding"]
        self.embed_size = self.num_items if self.one_hot_embedding else self.args["embed_size"]

        # ── Nutrient matrix  (num_items, num_nutrients) ───────────────────────
        self.nutrients = np.stack(
            [self.food_df[name].astype(float).values for name in self.nutrient_names],
            axis=1,
        ).astype(np.float32)

        # ── Delay / stddev matrices ───────────────────────────────────────────
        if self.args["use_delay"]:
            self.nutrient_delays = np.stack(
                [self.food_df[f"{n}_Delay"].astype(float).values for n in self.nutrient_names],
                axis=1,
            ).astype(np.float32)
            self.nutrient_stds = np.stack(
                [self.food_df[f"{n}_StdDev"].astype(float).values for n in self.nutrient_names],
                axis=1,
            ).astype(np.float32)
        else:
            self.nutrient_delays = np.zeros((self.num_items, self.num_nutrients), dtype=np.float32)
            self.nutrient_stds   = np.zeros((self.num_items, self.num_nutrients), dtype=np.float32)

        # ── Normalisation ─────────────────────────────────────────────────────
        if self.args["normalise"]:
            self.nutrient_norms = np.linalg.norm(self.nutrients, axis=0)
            self.nutrient_norms[self.nutrient_norms == 0] = 1.0
        else:
            self.nutrient_norms = np.ones(self.num_nutrients, dtype=np.float32)

        self.nutrients /= self.nutrient_norms

        # ── Apply nutrient mask (zero out non-target nutrients entirely) ───────
        self.nutrients *= self.nutrient_mask[None, :]

        # ── Precompute all food embeddings ────────────────────────────────────
        self._build_food_embeddings()

        # ── Spaces ────────────────────────────────────────────────────────────
        self.observation_space = spaces.Dict(
            {
                "physiological_state": spaces.Box(
                    low=0.0, high=4000.0,
                    shape=(self.num_nutrients,), dtype=np.float32,
                ),
                # Stacked menu embeddings: (num_foods, embed_size)
                "food_embeddings": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(self.num_foods, self.embed_size), dtype=np.float32,
                ),
            }
        )

        # num_foods eat-actions + 1 skip-action
        self.action_space = spaces.Discrete(self.num_foods + 1)

        # ── Seed & initial reset ──────────────────────────────────────────────
        self._seed = None
        if self.args["seed"] is not None:
            self._set_seed(self.args["seed"])

        self.reset(target_loc=self.args["target_loc"], seed=self.args["seed"])

    # ──────────────────────────────────────────────────────────────────────────
    # Seeding
    # ──────────────────────────────────────────────────────────────────────────
    def _set_seed(self, seed: int):
        self._seed = int(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return seed

    # ──────────────────────────────────────────────────────────────────────────
    # Embeddings
    # ──────────────────────────────────────────────────────────────────────────
    def _build_food_embeddings(self):
        """Pre-build a (num_items, embed_size) numpy array for fast index lookup."""
        if self.one_hot_embedding:
            self._all_embeddings = np.eye(self.num_items, dtype=np.float32)
        else:
            emb = nn.Embedding(self.num_items, self.embed_size)
            emb.weight.requires_grad_(False)
            self._all_embeddings = emb.weight.detach().cpu().numpy().astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # Menu sampling
    # ──────────────────────────────────────────────────────────────────────────
    def _sample_menu(self):
        """Draw num_foods unique food indices without replacement."""
        self._menu = self.np_random.choice(self.num_items, size=self.num_foods, replace=False)

    # ──────────────────────────────────────────────────────────────────────────
    # Target
    # ──────────────────────────────────────────────────────────────────────────
    def set_target(self, target_loc=None):
        """
        target_loc : array-like of length num_nutrients in raw (un-normalised) units.
        Defaults to 100.0 per nutrient. Non-target nutrients (per `target_nutrients`)
        are always zeroed out, regardless of target_loc.
        """
        if target_loc is None:
            raw = np.full(self.num_nutrients, 100.0, dtype=float)
        else:
            raw = np.array(target_loc, dtype=float)
            if len(raw) != self.num_nutrients:
                raise ValueError(
                    f"target_loc length {len(raw)} != num_nutrients {self.num_nutrients}"
                )
        self._target_location = (raw / self.nutrient_norms * self.nutrient_mask).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # Agent initialisation
    # ──────────────────────────────────────────────────────────────────────────
    def _initialise_agent(self):
        self._agent_location      = np.zeros(self.num_nutrients, dtype=np.float32)
        self._initial_location    = np.zeros(self.num_nutrients, dtype=np.float32)
        self._digestion_processes: List[np.ndarray] = []
        self._consumption_log     = []
        self._prev_distance       = float(
            np.linalg.norm(self._agent_location - self._target_location, ord=1)
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Digestion
    # ──────────────────────────────────────────────────────────────────────────
    def _start_digestion(self, food_index: int):
        for idx in range(self.num_nutrients):
            if self.nutrient_mask[idx] == 0:
                continue
            amount = float(self.nutrients[food_index, idx])
            if amount <= 0:
                continue

            delay   = float(self.nutrient_delays[food_index, idx])
            std     = max(float(self.nutrient_stds[food_index, idx]), 0.1)
            mean    = self.timepoint + delay
            horizon = int(np.ceil(6 * std)) + 1

            times  = np.arange(self.timepoint + 1, self.timepoint + 1 + horizon)
            kernel = np.exp(-0.5 * ((times - mean) / std) ** 2)
            kernel /= kernel.sum()

            schedule          = np.zeros((horizon, self.num_nutrients), dtype=np.float32)
            schedule[:, idx]  = kernel * amount
            self._digestion_processes.append(schedule)

            self._consumption_log.append(
                {
                    "nutrient":     self.nutrient_names[idx],
                    "nutrient_idx": idx,
                    "food_index":   food_index,
                    "times":        times.copy(),
                    "kernel":       kernel * amount,
                }
            )

    def _apply_digestion(self):
        total     = np.zeros(self.num_nutrients, dtype=np.float32)
        remaining = []
        for sched in self._digestion_processes:
            total += sched[0]
            if sched.shape[0] > 1:
                remaining.append(sched[1:])
        self._agent_location      += total
        self._digestion_processes  = remaining

    # ──────────────────────────────────────────────────────────────────────────
    # Decay
    # ──────────────────────────────────────────────────────────────────────────
    def _apply_decay(self):
        """Multiplicative nutrient burn-off each step: state *= (1 - decay_rate)."""
        self._agent_location *= (1.0 - self.decay_rate)
        self._agent_location  = np.maximum(self._agent_location, 0.0)

    # ──────────────────────────────────────────────────────────────────────────
    # Observation / reward
    # ──────────────────────────────────────────────────────────────────────────
    def _get_obs(self):
        return {
            "physiological_state": self._agent_location.copy(),
            "food_embeddings":     self._all_embeddings[self._menu].copy(),
        }

    def _get_info(self):
        return {
            "distance": float(
                np.linalg.norm(self._agent_location - self._target_location, ord=1)
            ),
            "menu": self._menu.copy(),
        }

    def _get_reward(self, prev_dist: float):
        curr_dist = float(
            np.linalg.norm(self._agent_location - self._target_location, ord=1)
        )
        progress      = prev_dist - curr_dist
        comfort       = -np.tanh(curr_dist / self.args["comfort_tau"])
        excess        = np.maximum(0.0, self._agent_location - self._target_location)
        toxicity      = np.sum(excess ** 2)
        toxic_penalty = -self.args["toxicity_scale"] * np.tanh(toxicity)

        reward = (
            self.args["progress_scale"] * progress
            + self.args["comfort_scale"] * comfort
            + toxic_penalty
        )
        return float(reward), curr_dist

    # ──────────────────────────────────────────────────────────────────────────
    # Core gym API
    # ──────────────────────────────────────────────────────────────────────────
    def step(self, action: int):
        skip_action = self.num_foods  # last index = skip

        # 1. Eat chosen food (if not skip)
        if action != skip_action:
            food_index = int(self._menu[action])
            if self.args["use_delay"]:
                self._start_digestion(food_index)
            else:
                self._agent_location += self.nutrients[food_index]

        # 2. Flush pending digestion schedules
        if self.args["use_delay"]:
            self._apply_digestion()

        # 3. Nutrient burn-off
        self._apply_decay()

        # 4. Reward & bookkeeping
        reward, distance    = self._get_reward(self._prev_distance)
        self._prev_distance = distance
        self.timepoint     += 1

        terminated = False
        truncated  = self.timepoint >= self.max_steps

        # 5. Fresh menu for the next observation
        self._sample_menu()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def reset(self, seed=None, target_loc=None):
        super().reset(seed=seed)
        if seed is not None:
            self._set_seed(seed)
        self.set_target(target_loc)
        self._initialise_agent()
        self._sample_menu()
        self.timepoint = 0
        return self._get_obs(), {}

    # ──────────────────────────────────────────────────────────────────────────
    # Visualisation
    # ──────────────────────────────────────────────────────────────────────────
    def render(self):
        if self.num_nutrients >= 3:
            fig = plt.figure()
            ax  = fig.add_subplot(111, projection="3d")
            ax.scatter(*self._target_location[:3], c="red",  label="Target")
            ax.scatter(*self._agent_location[:3],  c="blue", label="Agent")
            ax.set_xlabel(self.nutrient_names[0])
            ax.set_ylabel(self.nutrient_names[1])
            ax.set_zlabel(self.nutrient_names[2])
            ax.legend()
        elif self.num_nutrients == 2:
            fig, ax = plt.subplots()
            ax.scatter(*self._target_location, c="red",  label="Target")
            ax.scatter(*self._agent_location,  c="blue", label="Agent")
            ax.set_xlabel(self.nutrient_names[0])
            ax.set_ylabel(self.nutrient_names[1])
            ax.legend()
        else:
            fig, ax = plt.subplots()
            ax.bar(self.nutrient_names, self._agent_location,  color="blue", alpha=0.6, label="Agent")
            ax.bar(self.nutrient_names, self._target_location, color="red",  alpha=0.4, label="Target")
            ax.legend()
        plt.show()
        return fig

    def plot_consumption(self, max_time=None, figsize=(12, 8)):
        if not self._consumption_log:
            msg = (
                "No foods consumed in this episode."
                if self.args["use_delay"]
                else "No absorption mechanism used (delays disabled)."
            )
            print(msg)
            return plt.figure()

        if max_time is None:
            max_time = int(max(rec["times"][-1] for rec in self._consumption_log))

        x    = np.arange(0, max_time + 1)
        fig, axes = plt.subplots(self.num_nutrients, 1, figsize=figsize, sharex=True)
        if self.num_nutrients == 1:
            axes = [axes]

        for ax, name, idx in zip(axes, self.nutrient_names, range(self.num_nutrients)):
            total = np.zeros_like(x, dtype=float)
            for rec in self._consumption_log:
                if rec["nutrient_idx"] != idx:
                    continue
                arr   = np.zeros_like(x, dtype=float)
                valid = rec["times"] <= max_time
                arr[rec["times"][valid].astype(int)] = rec["kernel"][valid]
                total += arr
                ax.plot(x, arr, alpha=0.4)
            ax.plot(x, total, color="black", linewidth=2, label="Total absorbed")
            ax.set_ylabel(f"{name} absorbed")
            ax.grid(True)
            ax.legend()

        axes[-1].set_xlabel("Time (steps)")
        fig.suptitle("Per-nutrient digestion dynamics", fontsize=14)
        return fig


# ──────────────────────────────────────────────────────────────────────────────
# Back-compat alias: old code/notebooks import `NutriRL`. FoodEnv is the new
# canonical name (generic N nutrients + menu-of-K), NutriRL is kept as an
# alias so existing scripts that do `from envs import NutriRL` keep working.
# ──────────────────────────────────────────────────────────────────────────────
# NutriRL = FoodEnv
