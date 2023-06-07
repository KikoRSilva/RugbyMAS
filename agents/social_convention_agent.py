import numpy as np
from typing import List, Tuple

from .base_agent import Agent

AGENT_TEAM = 0
OPPONENT_TEAM = 1
ACTIONS = 6
DOWN, LEFT, UP, RIGHT, STAY, PASS = range(ACTIONS)


class ConventionAgent(Agent):

    def __init__(self, id: int, n_actions: int, n_agents: int, n_opponents: int, team: int, social_conventions: List):
        super(ConventionAgent, self).__init__("Convention Agent")
        self.id = id
        self.n_agents = n_agents
        self.n_opponents = n_opponents
        self.n_actions = n_actions
        self.team = team
        self.conventions = social_conventions

    def action(self) -> tuple:

        my_position = self.observation[0]
        ball_position = self.observation[1]
        score = self.observation[2]
        agents_position = self.observation[2:2+self.n_agents]
        opponents_position = self.observation[2+self.n_agents:2+self.n_agents+self.n_opponents]

        agent_order = self.conventions[0]
        action_order = self.conventions[1]

    def _step_convention(self):
        my_position = self.observation[0]
        ball_position = self.observation[1]
        agents_position = self.observation[2:2+self.n_agents]
        opponents_position = self.observation[2+self.n_agents:2+self.n_agents+self.n_opponents]



        return {
            'attack': [[0, 1, 2, 3, 4, 5, 6], []],
            'defense': [[0, 1, 2, 3, 4, 5, 6], []]
        }
 