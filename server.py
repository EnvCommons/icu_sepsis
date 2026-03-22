from openreward.environments import Server

from icu_sepsis_env import ICUSepsisEnvironment

if __name__ == "__main__":
    server = Server([ICUSepsisEnvironment])
    server.run()
