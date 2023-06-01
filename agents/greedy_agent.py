import numpy as np

from base_agent import Agent

class GreedyAgent(Agent):

    def __init__(self, id: int, n_actions: int, n_agents: int):
        super(GreedyAgent, self).__init__("Greedy Agent")
        self.id = id
        self.n_agents = n_agents
        self.n_actions = n_actions

    def action(self) -> tuple:
        a = np.random.randint(self.n_actions)
        if a == 5:
            # pass the ball
            ag = np.random.randint(self.n_agents) 
            while ag == self.id:
               ag = np.random.randint(self.n_agents)
            return (0, ag)
        return (a, None)