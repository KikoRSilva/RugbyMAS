import argparse
import numpy as np

from environment.rugby_env import RugbyEnv
from agents.agent import Agent
from agents.opponent import Opponent

NUM_EPISODES=10000

def run_multi_agent(environment, agents, n_episodes):

  results = np.zeros(n_episodes)

  for episode in range(n_episodes):
    
    steps = 0
    terminals = [False for _ in range(len(agents))]
    observations = environment.reset()

    while not all(terminals):
        steps += 1
        # TODO - main loop

    results[episode] = steps

    environment.close()

  return results
  


if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  parser.add_argument("--episodes", type=int, default=100)
  opt = parser.parse_args()

  # Setup the environment
  env = RugbyEnv()

  # Setup the teams
  teams = {
     "Random Team": [
        
     ]
  }

  # Create agents
  agent = Agent(env, team="home")
  opponent = Opponent(env, team="away")


