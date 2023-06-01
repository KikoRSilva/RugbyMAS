import math
import random
from scipy.spatial.distance import cityblock
from .base_agent import Agent

AGENT_TEAM = 0
OPPONENT_TEAM = 1
ACTIONS = 6
DOWN, LEFT, UP, RIGHT, STAY, PASS = range(ACTIONS)

class DummyGreedyAgent(Agent):

    def __init__(self, id: int, n_actions: int, n_agents: int, n_opponents: int, team: int):
        super(DummyGreedyAgent, self).__init__("Dummy Greedy Agent")
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
        opponents_position = self.observation[2+self.n_agents:2+self.n_agents+self.n_opponents:]
        print("My position"+my_position.__str__()+"   |   Ball position"+ball_position.__str__())
        if my_position.all() == ball_position.all():
            if self.team == AGENT_TEAM:
                closest_opponent, closest_opponent_i = self.find_closest_player(my_position, opponents_position)
                closest_bro, closest_bro_i = self.find_closest_player(my_position, agents_position)
            else:
                closest_opponent, closest_opponent_i = self.find_closest_player(my_position, agents_position)
                closest_bro, closest_bro_i = self.find_closest_player(my_position, opponents_position)
            bro_found = closest_bro is not None
            opponent_found = closest_opponent is not None
            if opponent_found and bro_found:
                action = self._decide(my_position, closest_opponent, closest_bro_i)
            else:
                action = (DOWN, None) if AGENT_TEAM == self.team else (UP, None)
        else:
            action = (random.randrange(ACTIONS-1), None)
        return action
    def _decide(self, my_position, closest_opponent, closest_bro):
        if abs(my_position[0]-closest_opponent[0]) + abs(my_position[1]-closest_opponent[1]) <= 2:
            return (PASS, closest_bro)
        else:
            return (DOWN, None) if AGENT_TEAM == self.team else (UP, None)
    def find_closest_player(self, my_position, opponent_positions):
        """
        Calculates the number of agents that are nearest to my position

        Args:
            my_position (tuple): The agent that wants to get the nearest agents
            agent_positions list[tuple]: position of the each agent in the list
            number_of_agents (int): The number of agents that we want

        Returns:
            list[tuple]: The positions of the agents that are nearer
        """
        min = math.inf
        closest_opponent_position = None
        closest_opponent_index = None
        for p in range(self.n_opponents):
            opponent_position = opponent_positions[p]
            distance = cityblock(my_position, opponent_position)
            if distance < min:
                min = distance
                closest_opponent_position = opponent_position
                closest_opponent_index = p
        return closest_opponent_position, closest_opponent_index


    def calculate_angle(my_position, agent1_position, agent2_position):
        """
        Calculate the angles between you and two agents.

        Args:
            my_position (tuple): Your position as a tuple of (x, y) coordinates.
            agent1_position (tuple): The position of the first agent as a tuple of (x, y) coordinates.
            agent2_position (tuple): The position of the second agent as a tuple of (x, y) coordinates.

        Returns:
            tuple: The angles between you and the two agents in degrees.
        """
        angle1 = math.degrees(math.atan2(agent1_position[1] - my_position[1], agent1_position[0] - my_position[0]))
        angle2 = math.degrees(math.atan2(agent2_position[1] - my_position[1], agent2_position[0] - my_position[0]))

        return angle1, angle2