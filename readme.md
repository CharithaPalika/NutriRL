# NutriRL: A Benchmark for Nutritional Regulation under Delayed State Transitions


> **NutriRL** is a nutrition-aware reinforcement learning project designed to study how different RL agents make food-choice decisions under delayed nutrient absorption and nutrient-targeting objectives. This repository supports the Reinforcement Learning Conference (RLC) 2026 paper workflow and reproducible experiments across multiple algorithms.

---

## Citation

If you use NutriRL in your research, please cite:

```bibtex
@article{khan2026nutrirl,
  title  = {NutriRL: A Benchmark for Nutritional Regulation under Delayed State Transitions},
  author = {Aniket Khan and Charitha Palika and V. Srinivasa Chakravarthy},
  year   = {2026},
  note   = {Preprint}
}
```
---

## Overview

This repository contains:

- a custom Gym-style environment for nutrient-based decision making under delayed nutrient absorption,
- implementations of multiple RL algorithms and supporting code to compare behavior and reproduce results,
- food datasets and experiment notebooks for training, evaluation, and analysis.

At a high level, NutriRL is a simple but powerful idea: train an agent to make food-choice decisions in a simulated body where the consequences of eating are not immediate. Instead of seeing a reward right away, the agent must reason about how today’s meal will affect future nutrient levels, health balance, and long-term goals.

---

## What is included

### Environment
The main environment models:
- nutrient intake for Carbs, Fat, and Protein,
- delayed digestion effects,
- target-based reward signals,

![Environment framework](figures/env_framework.png)

The environment is designed to feel intuitive: at each step, the agent observes its current physiological state and the available food item, then chooses whether to skip it or consume it. If it chooses to eat, the body absorbs nutrients over time rather than instantly, which makes the task more realistic and challenging. This introduces delayed consequences, uncertainty, and the need for planning rather than short-sighted reward chasing.

### Agents
The repository includes several RL agents for comparison, including:
- PPO
- DDQN
- SAC
- AC-GAE

### Models
The neural network architectures used by the agents are implemented for policy and value learning.

### Utilities
Supporting utilities include:
- action-selection helpers,
- replay-buffer logic,
- environment data structures.

---

## Dataset and experiments

The food datasets are stored in the dataset folder, and the experiment matrix is provided in the experiment table used for the paper pipeline.

This setup supports:
- delay enabled / disabled settings,
- toxicity penalty settings,
- target nutrient conditions,
- delay mean and standard deviation,
- multiple random seeds for repeated runs.

---

## Experiment notebooks

The main experiment pipelines are provided in the root notebooks:
- run_ppo_all.ipynb
- run_ddqn_all.ipynb
- run_sac_all.ipynb
- run_ac_gae_all.ipynb

These notebooks are the recommended entry points for reproducing the reported experimental runs.

---

## Why this environment matters

NutriRL is more than just a toy RL task. It is a structured benchmark for studying how reinforcement learning agents behave when the environment has:

- delayed effects,
- partial observability of internal states,
- long-horizon decision making,
- and reward signals that depend on future physiological outcomes.

This makes the repository valuable for researchers working at the intersection of reinforcement learning, computational biology, and health-inspired decision systems. It also provides a reproducible testbed for comparing different algorithms under the same nutritional setting.

---

## Environment idea

At each decision step:

1. The agent observes its current physiological state.
2. It also sees the current food embedding.
3. It chooses one of two actions:
   - 0: skip the food
   - 1: consume the food
4. The environment updates nutrient digestion and computes a reward based on how well the agent approaches the target nutrient balance.

This setup is useful for studying:
- delayed reward effects,
- nutrient-sensitive planning,
- long-horizon dietary decision making,
- algorithm comparison in a structured RL benchmark.

---

## Dependencies

Typical dependencies include:
- Python 3.9+
- PyTorch
- NumPy
- pandas
- gymnasium
- matplotlib
- tqdm

---

## Suggested workflow

1. Install the required Python packages.
2. Open one of the experiment notebooks.
3. Select an agent and an experiment setting from the experiment table.
4. Run the training/evaluation pipeline.
5. Record the results for comparison plots, ablations, and paper analysis.

---

## Relevance to the RLC 2026 paper

This repository is intended to support the paper’s focus on:
- reinforcement learning in nutrition-inspired tasks,
- delayed nutrient effects,
- comparison of policy-learning and value-learning methods,
- reproducible experimental evaluation.

It provides a practical implementation base for reporting algorithm performance and experimental trends in the RLC study.

---

## Summary

NutriRL is a compact but complete RL benchmark for studying food-choice decisions under delayed nutrient effects. It is designed to be easy to run, extend, and compare across multiple RL methods for paper-ready experiments.