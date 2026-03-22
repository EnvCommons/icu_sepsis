# ICUSepsis

[![⭐ OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/ICUSepsis)

## Description

ICU-Sepsis is an environment for evaluating agents on a tabular Markov Decision Process (MDP) that models sepsis treatment in the intensive care unit. Agents select treatment actions representing combinations of vasopressor and IV fluid doses to maximize patient survival probability. The MDP has 716 discrete states and 25 discrete actions, with transition dynamics derived from the MIMIC-III clinical dataset.

## Capabilities

- Sequential clinical treatment decision-making under uncertainty
- Balancing vasopressor and IV fluid dosing across 25 action combinations
- Optimizing sparse binary rewards (survival vs. death)
- Reasoning about admissible actions observed in real clinical data
- Monitoring patient severity via SOFA scores

## Compute Requirements

Minimal. The environment is a tabular MDP with no GPU or significant memory requirements.

## License

[MIT License](https://github.com/icu-sepsis/icu-sepsis/blob/main/packages/icu_sepsis/LICENSE.txt) (original ICU-Sepsis package).

## Tasks

There is one split:

- **train**: 1,000 tasks (seeds 0-999)

Each task uses a unique random seed that determines the initial patient state sampled from the learned initial state distribution. All tasks share the same underlying MDP dynamics (transition probabilities, reward structure).

## Reward Structure

This is a sparse, verifiable reward environment. All intermediate steps yield zero reward. Terminal rewards are:

- Patient survival (state 714): **+1.0**
- Patient death (state 713): **0.0**

The discount factor is 1.0 (undiscounted returns). We do not use LLM graders for this environment.

## Data

MDP parameters (transition matrix, reward matrix, initial state distribution, expert policy) ship with the [`icu-sepsis`](https://pypi.org/project/icu-sepsis/) pip package as a compressed NumPy archive (`dynamics.npz`). These parameters were derived from the MIMIC-III clinical dataset using the methodology of Komorowski et al. (2018).

## Tools

Agents are given two tools:

- **`treat(vasopressor_level, iv_fluid_level)`**: Administer treatment by choosing vasopressor dose (0-4) and IV fluid volume (0-4) independently. Returns the new patient state, SOFA score, list of admissible treatments, and current step count.
- **`info()`**: Display a reference of the state space, treatment parameters, reward structure, and other environment details.

## Time Horizon

ICU-Sepsis is a multi-turn environment. Episodes terminate when the patient reaches a survival or death state, or when the maximum step limit (20 steps) is reached. Based on baseline evaluations from the original paper, episodes typically last 9-11 steps.

## Environment Difficulty

From the original paper (Choudhary et al., 2024):

| Policy | Avg. Return | Avg. Episode Length |
|--------|-------------|---------------------|
| Random | 0.78 | 9.45 |
| Expert (clinician-derived) | 0.78 | 9.22 |
| Optimal (value iteration) | 0.88 | 10.99 |

The gap between expert/random (~0.78) and optimal (~0.88) performance indicates room for improvement, while the high random baseline reflects that most patients survive regardless of treatment in the underlying data.

## Other Environment Requirements

There are no further environment requirements; ICU-Sepsis works out of the box without any secrets or API keys.

## Safety

Agents interact only with a tabular MDP simulation derived from anonymized clinical records (MIMIC-III). There is no access to real patient data, external systems, or the internet during task execution.

## Citations

```bibtex
@inproceedings{choudhary2024icusepsis,
  title={{ICU-Sepsis}: A Benchmark {MDP} Built from Real Medical Data},
  author={Kartik Choudhary and Dhawal Gupta and Philip S. Thomas},
  booktitle={Reinforcement Learning Conference},
  year={2024},
  url={https://arxiv.org/abs/2406.05646}
}
```
