# NutriRL Extension

This repository is a small extension of NutriRL for running experiment notebooks on the vectorised `NutriRLVec` environment.

## Environment

![Environment framework](figures/env_framework.png)

`NutriRLVec` is a vectorised Gym-style environment. At each step, the agent sees a physiological state and a menu of foods, then chooses one action from `num_foods + 1` possibilities: eat one food or skip.

## Inputs

The dataset CSVs in `foods_dataset/` define the nutrient inputs for the environment. You can use any nutrient setup by editing these files, so the model is not limited to carbs, fats, and protein.

## Requirements

Install the packages listed in [requirements.txt](requirements.txt).

## Citation

```bibtex
@article{khan2026nutrirl,
	title  = {NutriRL: A Benchmark for Dimension Regulation under Delayed State Transitions},
	author = {Aniket Khan and Charitha Palika and V. Srinivasa Chakravarthy},
	year   = {2026},
	note   = {Preprint}
}
```
