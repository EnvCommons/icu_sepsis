import asyncio
import json
import os

from openai import AsyncOpenAI
from icu_sepsis_env import ICUSepsisEnvironment, TreatParams, InfoParams


def get_secrets():
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    secrets = {}
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    secrets[key.strip().lower()] = val.strip()
    return secrets


async def run_agent_test(max_turns=30):
    secrets = get_secrets()
    oai_client = AsyncOpenAI(api_key=secrets.get("openai_api_key"))

    tasks = ICUSepsisEnvironment.list_tasks(split="train")

    task = tasks[0]

    print(f"=== Agent Test: ICU-Sepsis ===")
    print(f"Task: {task['id']}")

    env = ICUSepsisEnvironment(task_spec=task, secrets=secrets)
    await env.setup()
    prompt = await env.get_prompt()

    tools = [
        {
            "type": "function",
            "name": "treat",
            "description": (
                "Administer treatment to the sepsis patient by choosing "
                "vasopressor and IV fluid levels independently. "
                "Returns the new patient state, SOFA score, and admissible treatments."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "vasopressor_level": {
                        "type": "integer",
                        "description": "Vasopressor dose level (0=none, 1=low, 2=medium, 3=high, 4=maximum)",
                    },
                    "iv_fluid_level": {
                        "type": "integer",
                        "description": "IV fluid volume level (0=none, 1=low, 2=medium, 3=high, 4=maximum)",
                    },
                },
                "required": ["vasopressor_level", "iv_fluid_level"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "info",
            "description": "Show a reference of the ICU-Sepsis environment details.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    ]

    input_list = [{"role": "user", "content": prompt[0].text}]
    finished = False
    turn = 0

    while not finished and turn < max_turns:
        turn += 1
        response = await oai_client.responses.create(
            model="gpt-5.2",
            tools=tools,
            input=input_list,
        )

        input_list += response.output

        for item in response.output:
            if item.type == "function_call":
                args = json.loads(str(item.arguments))

                if item.name == "treat":
                    result = await env.treat(TreatParams(**args))
                elif item.name == "info":
                    result = await env.info(InfoParams())
                else:
                    continue

                finished = result.finished
                reward = result.reward

                input_list.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": result.blocks[0].text,
                })
                print(input_list[-1]["output"])

                print(f"  Turn {turn}: {item.name}({args}) reward={reward:.2f} finished={finished}")

                if finished:
                    state = result.metadata.get("state", "?")
                    print(f"\n=== FINISHED! Final State: {state}, Reward: {reward} ===")
                    break

    if not finished:
        print(f"\n=== Hit max turns ({max_turns}) without finishing ===")

    await env.teardown()


if __name__ == "__main__":
    asyncio.run(run_agent_test())
