"""ICU-Sepsis OpenReward environment.

A tabular MDP modeling sepsis treatment in the ICU. The agent selects one of
25 treatment actions (5 vasopressor levels x 5 IV fluid levels) at each step.
Episodes terminate when the patient reaches a survival state (reward +1) or
death state (reward 0). Built on the icu-sepsis gymnasium environment which
derives its transition dynamics from the MIMIC-III clinical dataset.
"""

from pathlib import Path

import gymnasium as gym
import icu_sepsis  # noqa: F401 - registers ICU-Sepsis environments
import icu_sepsis.envs.sepsis as _icu_sepsis_env
from icu_sepsis.utils.io import MDPParameters
from typing import List
from pydantic import BaseModel, field_validator

from openreward.environments import (
    Environment, JSONObject, ToolOutput, tool, TextBlock,
)

# The ICU-Sepsis transition + reward matrices ((716, 25, 716) float64, ~98 MiB
# each) are identical for every session — only the seed/RNG differs. gym.make()
# loads a fresh ~196 MiB copy of them per session, so under concurrent sessions a
# pod duplicated them to tens of GiB and exhausted node RAM. Load them ONCE at
# import and share the read-only arrays across all sessions (see setup()).
_SHARED_MDP = MDPParameters(Path(_icu_sepsis_env.__file__).parent / "assets")

NUM_TASKS = 1000
NUM_ACTIONS = 25
ACTION_LEVELS = 5
STATE_DEATH = 713
STATE_SURVIVAL = 714


class TaskSpec(BaseModel):
    id: str
    seed: int


class TreatParams(BaseModel, extra="forbid"):
    vasopressor_level: int
    iv_fluid_level: int

    @field_validator("vasopressor_level", "iv_fluid_level")
    @classmethod
    def validate_level(cls, v: int) -> int:
        if v < 0 or v >= ACTION_LEVELS:
            raise ValueError(
                f"Level must be between 0 and {ACTION_LEVELS - 1}, got {v}"
            )
        return v


class InfoParams(BaseModel, extra="forbid"):
    pass


class ICUSepsisEnvironment(Environment):

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.config = TaskSpec.model_validate(task_spec)
        self.env = None
        self.current_state = None
        self.current_info = None
        self.step_count = 0
        self.episode_done = False

    async def setup(self):
        self.env = gym.make("Sepsis/ICU-Sepsis-v2")
        # Repoint this session's env at the shared transition/reward matrices so
        # the fresh ~196 MiB copies gym.make() just loaded are freed. NB: we do
        # NOT pass these via gym.make(params=...) — gymnasium deep-copies make
        # kwargs into the env's cached spec, which would reintroduce a full
        # per-session copy. The matrices are read-only during stepping (only the
        # per-env RNG is mutated), so sharing across sessions is safe.
        unwrapped = self.env.unwrapped
        unwrapped._tx_mat = _SHARED_MDP.tx_mat
        unwrapped._r_mat = _SHARED_MDP.r_mat
        obs, info = self.env.reset(seed=self.config.seed)
        self.current_state = int(obs)
        self.current_info = info
        self.step_count = 0
        self.episode_done = False

    async def teardown(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split == "train":
            return [
                {"id": f"icu_sepsis_{i}", "seed": i}
                for i in range(NUM_TASKS)
            ]
        return []

    @staticmethod
    def _action_to_levels(action: int) -> tuple[int, int]:
        return action // ACTION_LEVELS, action % ACTION_LEVELS

    def _format_observation(self, state: int, info: dict, step: int) -> str:
        parts = []
        parts.append(f"State: {state}")
        if "sofa_score" in info:
            parts.append(f"SOFA Score: {info['sofa_score']:.1f}")
        if "admissible_actions" in info:
            pairs = [
                self._action_to_levels(a) for a in info["admissible_actions"]
            ]
            formatted = [f"(vaso={v}, fluid={f})" for v, f in pairs]
            parts.append(f"Admissible Treatments: {', '.join(formatted)}")
        parts.append(f"Step: {step}")
        return "\n".join(parts)

    async def get_prompt(self) -> List[TextBlock]:
        obs_text = self._format_observation(
            self.current_state, self.current_info, self.step_count
        )
        prompt = f"""You are an AI clinician treating a sepsis patient in the ICU. Your goal is to choose treatments that maximize the patient's chance of survival.

ENVIRONMENT:
- The patient's condition is represented as a discrete state (0-715).
- States 0-712 represent different patient conditions.
- State {STATE_DEATH} is death (episode ends, reward = 0).
- State {STATE_SURVIVAL} is survival (episode ends, reward = 1).
- The episode also ends if the maximum number of steps is reached.

TREATMENT:
Use the `treat` tool to administer treatment. You choose two parameters independently:
- `vasopressor_level` (0-4): vasopressor dose (0 = none, 1 = low, 2 = medium, 3 = high, 4 = maximum)
- `iv_fluid_level` (0-4): IV fluid volume (0 = none, 1 = low, 2 = medium, 3 = high, 4 = maximum)

OBSERVATIONS:
After each treatment, you will see:
- The new patient state ID
- The SOFA score (Sequential Organ Failure Assessment; higher = more severe)
- The list of admissible treatments (treatments clinicians actually used for patients in similar states)
- The current step number

STRATEGY TIPS:
- Pay attention to admissible treatments — these reflect real clinical practice for patients in similar states.
- The SOFA score indicates organ dysfunction severity. Monitor how it changes with your treatment choices.
- This is a stochastic environment — the same treatment in the same state can lead to different outcomes.

Use the `treat` tool to administer treatment. Use the `info` tool to review the environment details.

CURRENT PATIENT STATE:
{obs_text}

Begin treating the patient."""
        return [TextBlock(text=prompt)]

    @tool
    async def treat(self, params: TreatParams) -> ToolOutput:
        """Administer treatment to the sepsis patient by choosing vasopressor
        and IV fluid levels independently. vasopressor_level and iv_fluid_level
        each range from 0 (none) to 4 (maximum dose). Returns the new patient
        state, SOFA score, and admissible treatments."""
        if self.episode_done:
            return ToolOutput(
                blocks=[TextBlock(text="The episode is already over.")],
                metadata={"error": "episode_finished"},
                reward=0.0,
                finished=True,
            )

        action = params.vasopressor_level * ACTION_LEVELS + params.iv_fluid_level
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        self.current_state = int(obs)
        self.current_info = info

        finished = terminated or truncated
        if finished:
            self.episode_done = True

        obs_text = self._format_observation(
            self.current_state, self.current_info, self.step_count
        )
        display_parts = [obs_text]

        if finished:
            if self.current_state == STATE_SURVIVAL:
                display_parts.append(
                    f"\n=== PATIENT SURVIVED === Reward: {float(reward):.1f}"
                )
            elif self.current_state == STATE_DEATH:
                display_parts.append(
                    f"\n=== PATIENT DIED === Reward: {float(reward):.1f}"
                )
            else:
                display_parts.append(
                    f"\n=== MAX STEPS REACHED === Reward: {float(reward):.1f}"
                )

        display_text = "\n".join(display_parts)

        return ToolOutput(
            blocks=[TextBlock(text=display_text)],
            metadata={
                "state": self.current_state,
                "reward": float(reward),
                "step": self.step_count,
                "terminated": terminated,
                "truncated": truncated,
                "sofa_score": float(info.get("sofa_score", 0.0)),
            },
            reward=float(reward),
            finished=finished,
        )

    @tool
    async def info(self, params: InfoParams) -> ToolOutput:
        """Show a reference of the ICU-Sepsis action space and environment details."""
        reference = f"""ICU-SEPSIS ENVIRONMENT REFERENCE

STATES:
- 716 discrete states (0-715)
- States 0-712: patient conditions derived from clinical data clusters
- State {STATE_DEATH}: patient death (terminal)
- State {STATE_SURVIVAL}: patient survival (terminal)
- State 715: absorbing state (s_inf)

TREATMENT PARAMETERS:
Use the `treat` tool with two parameters:
- vasopressor_level (0-4): 0=none, 1=low, 2=medium, 3=high, 4=maximum
- iv_fluid_level (0-4): 0=none, 1=low, 2=medium, 3=high, 4=maximum

This gives 25 possible treatment combinations (5 x 5).

REWARDS:
- Survival (state {STATE_SURVIVAL}): +1.0
- Death (state {STATE_DEATH}): 0.0
- All intermediate steps: 0.0

SOFA SCORE:
The Sequential Organ Failure Assessment score measures organ dysfunction.
Higher values indicate more severe organ failure (range typically 0-24).

ADMISSIBLE TREATMENTS:
Each state has a set of admissible treatments — combinations actually observed
in clinical data for patients in similar conditions. Non-admissible treatments
use mean transition probabilities as a fallback."""
        return ToolOutput(
            blocks=[TextBlock(text=reference)],
            metadata={},
            reward=0.0,
            finished=False,
        )
