import math
import random
import numpy as np
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

        if my_position[0] == ball_position[0] and my_position[1] == ball_position[1]:
            # Temos a bola
            if self.team == AGENT_TEAM:
                closest_opponents = self.find_closest_players(my_position, opponents_position, 1)
                if self.player_nearby(my_position=my_position, closest_opponent=closest_opponents[0], distance=2):
                    agent_i = self.find_best_player(agents_position, opponents_position, my_position, AGENT_TEAM)
                    if agent_i is not None:
                        action = (PASS, agent_i)
                    else:
                        action = (DOWN, None)
                else:
                    action = (DOWN, None)

            # Não temos a bola
            else:
                closest_agents = self.find_closest_players(my_position, agents_position, 1)
                if self.player_nearby(my_position=my_position, closest_opponent=closest_agents[0], distance=2):
                    opponent_i = self.find_best_player(opponents_position, agents_position,my_position,OPPONENT_TEAM)
                    if opponent_i is not None:
                        action = (PASS, opponent_i)
                    else:
                        action = (UP, None)
                else:
                    action = (UP, None)

        elif self.my_team_has_ball(agents_position if self.team == AGENT_TEAM else opponents_position, ball_position):
                # Somos apoiantes
                if self.team == AGENT_TEAM:
                    if my_position[0] <= ball_position[0]:
                        # Estamos ao lado ou atras do portador da bola -> avançar
                        action = (DOWN, None)
                    else:
                        # Recuar para poder receber a bola
                        action = (UP, None)
                else:
                    if my_position[0] >= ball_position[0]:
                        action = (UP, None)
                    else:
                        action = (DOWN, None)
        else:
            # Vamos defender
            action = self.go_toward_ball_carrier(my_position=my_position, ball_carrier_position=ball_position)
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
            
    def find_best_player(self, teammates, opponents, my_position, team):
        angles = {}
        teammate_i = None

        for teammate_i, teammate_pos in enumerate(teammates):
            closest_two_opponents = self.find_closest_players(teammate_pos, opponents, 2)
            angles[teammate_i] = self.calculate_angle(my_position=teammate_pos, agent1_position=closest_two_opponents[0], agent2_position=closest_two_opponents[1])

                
        sorted_angles = sorted(angles.items(), key=lambda x: x[1])
        if team == AGENT_TEAM:
            for i, _ in sorted_angles:
                if teammates[i][0] <= my_position[0]:
                    return i
            return None

        else:
            for i,_ in sorted_angles:
                if teammates[i][0] >=  my_position[0]:
                    return i
            return None


    
    def calculate_angle(self, my_position, agent1_position, agent2_position):
        angle1 = math.degrees(math.atan2(agent1_position[1] - my_position[1], agent1_position[0] - my_position[0]))
        angle2 = math.degrees(math.atan2(agent2_position[1] - my_position[1], agent2_position[0] - my_position[0]))
        return angle1, angle2
    
    def my_team_has_ball(self, team, ball_pos):
        for teammate_pos in team:
            if teammate_pos[0] == ball_pos[0] and teammate_pos[1] == ball_pos[1]:
                return True
        return False
    
    def go_toward_ball_carrier(self, my_position, ball_carrier_position):
        my_pos_x, my_pos_y = my_position
        ball_carrier_x, ball_carrier_y = ball_carrier_position

        if my_pos_x == ball_carrier_x and my_pos_y == ball_carrier_y:
            return (STAY, None)
        
        dx = ball_carrier_x - my_pos_x
        dy = ball_carrier_y - my_pos_y

        if abs(dx) > abs(dy):
            if dy > 0:
                return (UP, None)
            
            return (DOWN, None)
        else:
            if dx > 0:
                return (LEFT, None)
            return (RIGHT, None)
            
    