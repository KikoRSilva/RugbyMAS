import argparse
import random
import time
import numpy as np

from environment.utils import compare_results
from environment.rugby_env import RugbyEnv

from agents.random_agent import RandomAgent
from agents.dummy_greedy_agent import DummyGreedyAgent
from agents.greedy_agent import GreedyAgent

ROLES = 5
BALL_CARRIER, ATTACKER, TACKLER, FORWARD_DEFENSE, BACK_DEFENSE = range(ROLES)

def run_multi_agent(environment, agents, n_episodes):

  results = np.zeros(n_episodes)

  for episode in range(n_episodes):
    
    steps = 0
    terminals = [False for _ in range(len(agents))]
    observations = environment.reset()

    while not all(terminals):
        steps += 1
        # TODO - main loop
        for observations, agent in zip(observations, agents):
          agent.see(observations)
        actions = [agent.action() for agent in agents]
        next_observations, rewards, terminals, info = environment.step(actions)

        environment.render()
        time.sleep(opt.render_sleep_time)

        observations = next_observations

    results[episode] = steps

    environment.close()

  return results
  


if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  parser.add_argument("--episodes", type=int, default=10)
  parser.add_argument("--n_agents", type=int, default=7)
  parser.add_argument("--n_opponents", type=int, default=7)
  parser.add_argument("--render-sleep-time", type=float, default=0.5)
  opt = parser.parse_args()

  # Setup the environment
  env = RugbyEnv(grid_shape=(21,11), n_agents=opt.n_agents, n_opponents=opt.n_opponents, max_steps=1000)

  # Set seeds.
  random.seed(3)
  np.random.seed(3)
  env.seed(3)

  ACTIONS = 6
  DOWN, LEFT, UP, RIGHT, STAY, PASS = range(ACTIONS)
  conventions = {
     'attack': [[0, 1, 2, 3, 4, 5, 6], []],
     'defense': [[0, 1, 2, 3, 4, 5, 6], []]
  }

  roles = [BALL_CARRIER, ATTACKER, TACKLER, FORWARD_DEFENSE, BACK_DEFENSE]

  random_games = {
    "Random Team vs Random Team": [
           # Attacker team
           RandomAgent(id=0, n_actions=env.action_space[0].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=0),
           RandomAgent(id=1, n_actions=env.action_space[1].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=0),
           RandomAgent(id=2, n_actions=env.action_space[2].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=0),
           RandomAgent(id=3, n_actions=env.action_space[3].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=0),
           RandomAgent(id=4, n_actions=env.action_space[4].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=0),
           RandomAgent(id=5, n_actions=env.action_space[5].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=0),
           RandomAgent(id=6, n_actions=env.action_space[6].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=0),

           # Defensive team
           RandomAgent(id=7, n_actions=env.action_space[7].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
           RandomAgent(id=8, n_actions=env.action_space[8].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
           RandomAgent(id=9, n_actions=env.action_space[9].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
           RandomAgent(id=10, n_actions=env.action_space[10].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
           RandomAgent(id=11, n_actions=env.action_space[11].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
           RandomAgent(id=12, n_actions=env.action_space[12].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
           RandomAgent(id=13, n_actions=env.action_space[13].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
      ],
      "Dummy Greedy Team vs Random Team": [
          # Attacker team
          DummyGreedyAgent(id=0, n_actions=env.action_space[0].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
          DummyGreedyAgent(id=1, n_actions=env.action_space[1].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
          DummyGreedyAgent(id=2, n_actions=env.action_space[2].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
          DummyGreedyAgent(id=3, n_actions=env.action_space[3].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
          DummyGreedyAgent(id=4, n_actions=env.action_space[4].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
          DummyGreedyAgent(id=5, n_actions=env.action_space[5].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
          DummyGreedyAgent(id=6, n_actions=env.action_space[6].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),

          # Defensive team
          RandomAgent(id=7, n_actions=env.action_space[7].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
          RandomAgent(id=8, n_actions=env.action_space[8].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
          RandomAgent(id=9, n_actions=env.action_space[9].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
          RandomAgent(id=10, n_actions=env.action_space[10].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
          RandomAgent(id=11, n_actions=env.action_space[11].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
          RandomAgent(id=12, n_actions=env.action_space[12].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
          RandomAgent(id=13, n_actions=env.action_space[13].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
      ],
      "Greedy Team vs Random Team": [
          # Attacker team
          GreedyAgent(id=0, n_actions=env.action_space[0].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
          GreedyAgent(id=1, n_actions=env.action_space[1].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
          GreedyAgent(id=2, n_actions=env.action_space[2].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
          GreedyAgent(id=3, n_actions=env.action_space[3].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
          GreedyAgent(id=4, n_actions=env.action_space[4].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
          GreedyAgent(id=5, n_actions=env.action_space[5].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
          GreedyAgent(id=6, n_actions=env.action_space[6].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),

          # Defensive team
          RandomAgent(id=7, n_actions=env.action_space[7].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
          RandomAgent(id=8, n_actions=env.action_space[8].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
          RandomAgent(id=9, n_actions=env.action_space[9].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
          RandomAgent(id=10, n_actions=env.action_space[10].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
          RandomAgent(id=11, n_actions=env.action_space[11].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
          RandomAgent(id=12, n_actions=env.action_space[12].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
          RandomAgent(id=13, n_actions=env.action_space[13].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
      ]
  }

  dummy_greedy_games = {
      "Random Team vs Dummy Greedy Team": [
           # Attacker team
           RandomAgent(id=0, n_actions=env.action_space[0].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=0),
           RandomAgent(id=1, n_actions=env.action_space[1].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=0),
           RandomAgent(id=2, n_actions=env.action_space[2].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=0),
           RandomAgent(id=3, n_actions=env.action_space[3].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=0),
           RandomAgent(id=4, n_actions=env.action_space[4].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=0),
           RandomAgent(id=5, n_actions=env.action_space[5].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=0),
           RandomAgent(id=6, n_actions=env.action_space[6].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=0),

           # Defensive team
           DummyGreedyAgent(id=7, n_actions=env.action_space[7].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
           DummyGreedyAgent(id=8, n_actions=env.action_space[8].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
           DummyGreedyAgent(id=9, n_actions=env.action_space[9].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
           DummyGreedyAgent(id=10, n_actions=env.action_space[10].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
           DummyGreedyAgent(id=11, n_actions=env.action_space[11].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
           DummyGreedyAgent(id=12, n_actions=env.action_space[12].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
           DummyGreedyAgent(id=13, n_actions=env.action_space[13].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
      ],

      "Dummy Greedy Team vs Dummy Greedy Team": [
           # Attacker team
           DummyGreedyAgent(id=0, n_actions=env.action_space[0].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
           DummyGreedyAgent(id=1, n_actions=env.action_space[1].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
           DummyGreedyAgent(id=2, n_actions=env.action_space[2].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
           DummyGreedyAgent(id=3, n_actions=env.action_space[3].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
           DummyGreedyAgent(id=4, n_actions=env.action_space[4].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
           DummyGreedyAgent(id=5, n_actions=env.action_space[5].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
           DummyGreedyAgent(id=6, n_actions=env.action_space[6].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),

           # Defensive team
           DummyGreedyAgent(id=7, n_actions=env.action_space[7].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
           DummyGreedyAgent(id=8, n_actions=env.action_space[8].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
           DummyGreedyAgent(id=9, n_actions=env.action_space[9].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
           DummyGreedyAgent(id=10, n_actions=env.action_space[10].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
           DummyGreedyAgent(id=11, n_actions=env.action_space[11].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
           DummyGreedyAgent(id=12, n_actions=env.action_space[12].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
           DummyGreedyAgent(id=13, n_actions=env.action_space[13].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
      ],

      "Greedy Team vs Dummy Greedy Team": [
         # Attacker team
          GreedyAgent(id=0, n_actions=env.action_space[0].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
          GreedyAgent(id=1, n_actions=env.action_space[1].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
          GreedyAgent(id=2, n_actions=env.action_space[2].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
          GreedyAgent(id=3, n_actions=env.action_space[3].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
          GreedyAgent(id=4, n_actions=env.action_space[4].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
          GreedyAgent(id=5, n_actions=env.action_space[5].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
          GreedyAgent(id=6, n_actions=env.action_space[6].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),

          # Defensive team
          DummyGreedyAgent(id=7, n_actions=env.action_space[7].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
          DummyGreedyAgent(id=8, n_actions=env.action_space[8].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
          DummyGreedyAgent(id=9, n_actions=env.action_space[9].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
          DummyGreedyAgent(id=10, n_actions=env.action_space[10].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
          DummyGreedyAgent(id=11, n_actions=env.action_space[11].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
          DummyGreedyAgent(id=12, n_actions=env.action_space[12].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
          DummyGreedyAgent(id=13, n_actions=env.action_space[13].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
      ]
  }

  greedy_games = {
      "Random Team vs Greedy Team": [
          # Attacker team
          RandomAgent(id=0, n_actions=env.action_space[0].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=0),
          RandomAgent(id=1, n_actions=env.action_space[1].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=0),
          RandomAgent(id=2, n_actions=env.action_space[2].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=0),
          RandomAgent(id=3, n_actions=env.action_space[3].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=0),
          RandomAgent(id=4, n_actions=env.action_space[4].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=0),
          RandomAgent(id=5, n_actions=env.action_space[5].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=0),
          RandomAgent(id=6, n_actions=env.action_space[6].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=0),

          # Defensive team
          GreedyAgent(id=7, n_actions=env.action_space[7].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
          GreedyAgent(id=8, n_actions=env.action_space[8].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
          GreedyAgent(id=9, n_actions=env.action_space[9].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
          GreedyAgent(id=10, n_actions=env.action_space[10].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
          GreedyAgent(id=11, n_actions=env.action_space[11].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
          GreedyAgent(id=12, n_actions=env.action_space[12].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
          GreedyAgent(id=13, n_actions=env.action_space[13].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
      ],

      "Dummy Greedy Team vs Greedy Team": [
          # Attacker team
          DummyGreedyAgent(id=0, n_actions=env.action_space[0].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
          DummyGreedyAgent(id=1, n_actions=env.action_space[1].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
          DummyGreedyAgent(id=2, n_actions=env.action_space[2].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
          DummyGreedyAgent(id=3, n_actions=env.action_space[3].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
          DummyGreedyAgent(id=4, n_actions=env.action_space[4].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
          DummyGreedyAgent(id=5, n_actions=env.action_space[5].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
          DummyGreedyAgent(id=6, n_actions=env.action_space[6].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),

          # Defensive team
          GreedyAgent(id=7, n_actions=env.action_space[7].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
          GreedyAgent(id=8, n_actions=env.action_space[8].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
          GreedyAgent(id=9, n_actions=env.action_space[9].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
          GreedyAgent(id=10, n_actions=env.action_space[10].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
          GreedyAgent(id=11, n_actions=env.action_space[11].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
          GreedyAgent(id=12, n_actions=env.action_space[12].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
          GreedyAgent(id=13, n_actions=env.action_space[13].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
      ],

      "Greedy Team vs  Greedy Team": [
         # Attacker team
          GreedyAgent(id=0, n_actions=env.action_space[0].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
          GreedyAgent(id=1, n_actions=env.action_space[1].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
          GreedyAgent(id=2, n_actions=env.action_space[2].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
          GreedyAgent(id=3, n_actions=env.action_space[3].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
          GreedyAgent(id=4, n_actions=env.action_space[4].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
          GreedyAgent(id=5, n_actions=env.action_space[5].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
          GreedyAgent(id=6, n_actions=env.action_space[6].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),

          # Defensive team
          GreedyAgent(id=7, n_actions=env.action_space[7].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
          GreedyAgent(id=8, n_actions=env.action_space[8].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
          GreedyAgent(id=9, n_actions=env.action_space[9].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
          GreedyAgent(id=10, n_actions=env.action_space[10].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
          GreedyAgent(id=11, n_actions=env.action_space[11].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
          GreedyAgent(id=12, n_actions=env.action_space[12].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
          GreedyAgent(id=13, n_actions=env.action_space[13].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
      ]
  }

  # Evaluate teams
  results = {}

  results_random = {}
  for game, agents in random_games.items():
      print(f'Running {game}.')
      result = run_multi_agent(env, agents, opt.episodes)
      results_random[game] = result
  results['Random Team Tournament Performance'] = results_random

  results_dummy_greedy = {}
  for game, agents in dummy_greedy_games.items():
        print(f'Running {game}.')
        result = run_multi_agent(env, agents, opt.episodes)
        results_dummy_greedy[game] = result
  results['Dummy Greedy Team Tournament Performance'] = results_dummy_greedy

  results_greedy = {}
  for game, agents in greedy_games.items():
        print(f'Running {game}.')
        result = run_multi_agent(env, agents, opt.episodes)
        results_greedy[game] = result
  results['Greedy Team Tournament Performance'] = results_greedy

  # Compare results
  for name, res in results.items():
    compare_results(res, title=name, colors=["orange", "blue", "green"])