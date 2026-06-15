# NutriRL: A Benchmark for Nutritional Regulation under Delayed State Transitions

NutriRL is a nutrition-aware reinforcement learning project designed to study how different RL agents make food-choice decisions under delayed nutrient absorption and nutrient-targeting objectives. This repository supports the Reinforcement Learning Conference (RLC) 2026 paper workflow and reproducible experiments across multiple algorithms.
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

- a custom Gym-style environment for nutrient-based decision making,
- multiple RL agents for comparison,
- food datasets with varying delay and uncertainty settings,
- experiment notebooks for paper-style evaluation.

The core objective is to model an agent that observes its current physiological state and the current food item, then decides whether to consume it in order to move toward a target nutrient profile.

---

## What is included

### Environment
The main environment models:
- nutrient intake for Carbs, Fat, and Protein,
- delayed digestion effects,
- stochastic nutrient absorption,
- target-based reward signals,
- food embeddings used by learning agents.

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