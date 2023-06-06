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

  parser.add_argument("--episodes", type=int, default=100)
  parser.add_argument("--n_agents", type=int, default=7)
  parser.add_argument("--n_opponents", type=int, default=7)
  parser.add_argument("--render-sleep-time", type=float, default=0.5)
  opt = parser.parse_args()

  # Setup the environment
  env = RugbyEnv(grid_shape=(21,11), n_agents=opt.n_agents, n_opponents=opt.n_opponents, max_steps=100)

  # Set seeds.
  random.seed(3)
  np.random.seed(3)
  env.seed(3)

  # Setup the teams
  conventions = [[0, 1, 2, 3, 5, 6], [BALL_CARRIER, ATTACKER, TACKLER, FORWARD_DEFENSE, BACK_DEFENSE]]

  roles = [BALL_CARRIER, ATTACKER, TACKLER, FORWARD_DEFENSE, BACK_DEFENSE]
  teams = {
    #  "Random Team vs Random Team": [
    #       # Attacker team
    #       RandomAgent(id=0, n_actions=env.action_space[0].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=0),
    #       RandomAgent(id=1, n_actions=env.action_space[1].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=0),
    #       RandomAgent(id=2, n_actions=env.action_space[2].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=0),
    #       RandomAgent(id=3, n_actions=env.action_space[3].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=0),
    #       RandomAgent(id=4, n_actions=env.action_space[4].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=0),
    #       RandomAgent(id=5, n_actions=env.action_space[5].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=0),
    #       RandomAgent(id=6, n_actions=env.action_space[6].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=0),

    #       # Defensive team
    #       RandomAgent(id=7, n_actions=env.action_space[7].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
    #       RandomAgent(id=8, n_actions=env.action_space[8].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
    #       RandomAgent(id=9, n_actions=env.action_space[9].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
    #       RandomAgent(id=10, n_actions=env.action_space[10].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
    #       RandomAgent(id=11, n_actions=env.action_space[11].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
    #       RandomAgent(id=12, n_actions=env.action_space[12].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
    #       RandomAgent(id=13, n_actions=env.action_space[13].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
    #  ],

      # "Dummy Greedy Team vs Random Team": [
      #     # Attacker team
      #     DummyGreedyAgent(id=0, n_actions=env.action_space[0].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
      #     DummyGreedyAgent(id=1, n_actions=env.action_space[1].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
      #     DummyGreedyAgent(id=2, n_actions=env.action_space[2].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
      #     DummyGreedyAgent(id=3, n_actions=env.action_space[3].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
      #     DummyGreedyAgent(id=4, n_actions=env.action_space[4].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
      #     DummyGreedyAgent(id=5, n_actions=env.action_space[5].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
      #     DummyGreedyAgent(id=6, n_actions=env.action_space[6].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),

      #     # Defensive team
      #     RandomAgent(id=7, n_actions=env.action_space[7].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
      #     RandomAgent(id=8, n_actions=env.action_space[8].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
      #     RandomAgent(id=9, n_actions=env.action_space[9].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
      #     RandomAgent(id=10, n_actions=env.action_space[10].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
      #     RandomAgent(id=11, n_actions=env.action_space[11].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
      #     RandomAgent(id=12, n_actions=env.action_space[12].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
      #     RandomAgent(id=13, n_actions=env.action_space[13].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team=1),
      # ],

      # "Dummy Greedy Team vs Dummy Greedy Team": [
      #    # Attacker team
      #     DummyGreedyAgent(id=0, n_actions=env.action_space[0].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
      #     DummyGreedyAgent(id=1, n_actions=env.action_space[1].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
      #     DummyGreedyAgent(id=2, n_actions=env.action_space[2].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
      #     DummyGreedyAgent(id=3, n_actions=env.action_space[3].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
      #     DummyGreedyAgent(id=4, n_actions=env.action_space[4].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
      #     DummyGreedyAgent(id=5, n_actions=env.action_space[5].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),
      #     DummyGreedyAgent(id=6, n_actions=env.action_space[6].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 0),

      #     # Defensive team
      #     DummyGreedyAgent(id=7, n_actions=env.action_space[0].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 1),
      #     DummyGreedyAgent(id=8, n_actions=env.action_space[1].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 1),
      #     DummyGreedyAgent(id=9, n_actions=env.action_space[2].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 1),
      #     DummyGreedyAgent(id=10, n_actions=env.action_space[3].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 1),
      #     DummyGreedyAgent(id=11, n_actions=env.action_space[4].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 1),
      #     DummyGreedyAgent(id=12, n_actions=env.action_space[5].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 1),
      #     DummyGreedyAgent(id=13, n_actions=env.action_space[6].n, n_agents=opt.n_agents, n_opponents=opt.n_opponents, team = 1),
      # ],

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

  # Evaluate teams
  results = {}
  for team, agents in teams.items():
      print(f'Running {team}.')
      result = run_multi_agent(env, agents, opt.episodes)
      results[team] = result

  compare_results(results, title="Tournament Performance", colors=["orange","blue","green","gray"])

  # 4 - Compare results
  # compare_results(
  #     results,
  #     title="Teams Comparison on 'Predator Prey' Environment",
  #     colors=["orange", "green", "blue", "gray"]
  # )


