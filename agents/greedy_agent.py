import math
import random
from scipy.spatial.distance import cityblock
from .base_agent import Agent

AGENT_TEAM = 0
OPPONENT_TEAM = 1
ACTIONS = 6
DOWN, LEFT, UP, RIGHT, STAY, PASS = range(ACTIONS)

class GreedyAgent(Agent):

    def __init__(self, id: int, n_actions: int, n_agents: int, n_opponents: int, team: int):
        super(GreedyAgent, self).__init__("Greedy Agent")
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

        if my_position.all() == ball_position.all():
            if self.team == AGENT_TEAM:
                closest_opponents = self.find_closest_players(my_position, opponents_position, 1)
                if self.player_nearby(my_position=my_position, closest_opponent=closest_opponents[0], distance=2):
                    agent_i = self.find_best_player(agents_position, opponents_position)
                    return (PASS, agent_i)
                else:
                    return (DOWN, None)
            else:
                closest_agents = self.find_closest_players(my_position, agents_position, 1)
                if self.player_nearby(my_position=my_position, closest_opponent=closest_agents[0], distance=2):
                    opponent_i = self.find_best_player(opponents_position, agents_position)
                    return (PASS, opponent_i)
                else:
                    return (UP, None)
        else:
            action = (random.randrange(ACTIONS-1), None)
        return action
    
    def player_nearby(self, my_position, closest_opponent, distance):
        return abs(my_position[0]-closest_opponent[0]) + abs(my_position[1]-closest_opponent[1]) <= distance

    def find_closest_players(self, my_position, opponent_positions, n_players):
        distances = {}

        for i, opponent_pos in enumerate(opponent_positions):
            distances[i] = cityblock(my_position, opponent_pos)

        sorted_opponents = sorted(distances.items(), key=lambda x: x[1])
        closest_opponents = [opponent_positions[i] for i, _ in sorted_opponents[:n_players]]

        return closest_opponents
            
    def find_best_player(self, teammates, opponents):
        angles = {}

        for teammate_i, teammate_pos in enumerate(teammates):
            closest_two_opponents = self.find_closest_players(teammate_pos, opponents, 2)
            angles[teammate_i] = self.calculate_angle(my_position=teammate_pos, agent1_position=closest_two_opponents[0], agent2_position=closest_two_opponents[1])

        sorted_angles = sorted(angles.items(), key=lambda x: x[1])
        teammate_i, _ = sorted_angles[0]

        return teammate_i

    
    def calculate_angle(self, my_position, agent1_position, agent2_position):
        angle1 = math.degrees(math.atan2(agent1_position[1] - my_position[1], agent1_position[0] - my_position[0]))
        angle2 = math.degrees(math.atan2(agent2_position[1] - my_position[1], agent2_position[0] - my_position[0]))

        return angle1, angle2


    