import numpy as np

from .base_agent import Agent

AGENT_TEAM = 0
OPPONENT_TEAM = 1
ACTIONS = 6
DOWN, LEFT, UP, RIGHT, STAY, PASS = range(ACTIONS)

class RandomAgent(Agent):

    def __init__(self, id: int, n_actions: int, n_agents: int, n_opponents: int, team: int):
        super(RandomAgent, self).__init__("Random Agent")
        self.id = id
        self.n_agents = n_agents
        self.n_opponents = n_opponents
        self.n_actions = n_actions
        self.team = team

    def action(self) -> tuple:

        my_position = self.observation[0]
        ball_position = self.observation[1]
        score = self.observation[2]
        agents_position = self.observation[2:2+self.n_agents]
        opponents_position = self.observation[2+self.n_agents:2+self.n_agents+self.n_opponents]

        a = np.random.randint(self.n_actions)

        if a == PASS and my_position[0] == ball_position[0] and my_position[1] == ball_position[1]:
            # pass the ball
            ag = None
            valid_agent = False

            while not valid_agent:
                if self.team == AGENT_TEAM:
                    ag = np.random.randint(self.n_agents)
                    if ag != self.id and agents_position[ag][0] <= my_position[0] :
                        valid_agent = True
                    else:
                        break
                else: 
                    ag = np.random.randint(self.n_opponents)
                    if ag != (self.id - self.n_agents):
                        if opponents_position[ag][0] >= my_position[0] :
                            valid_agent = True
                        else:
                            break
                    

            if valid_agent:
                return (PASS, ag)
        
        
        a = np.random.randint(self.n_actions-1)
        return (a, None)
 