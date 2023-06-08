import math
import numpy as np
from typing import List, Tuple
from scipy.spatial.distance import cityblock

from .base_agent import Agent

AGENT_TEAM = 0
OPPONENT_TEAM = 1

ATTACKING = 0
DEFENDING = 1

ACTIONS = 6
DOWN, LEFT, UP, RIGHT, STAY, PASS = range(ACTIONS)

ROLES = 7
BALL_CARRIER, RIGHT_SUPPORTER, LEFT_SUPPORTER, RIGHT_SUB_SUPPORTER, LEFT_SUB_SUPPORTER, RIGH_WING, LEFT_WING = range(ROLES)

FORWARD_DEFENDER_1, FORWARD_DEFENDER_2, FORWARD_DEFENDER_3, FORWARD_DEFENDER_4, BACK_DEFENDER_1, BACK_DEFENDER_2, BACK_DEFENDER_3 = range(ROLES)

DISTANCE_BETWEEN_LINES = 5

class RoleAgent(Agent):

    def __init__(self, id: int, n_agents: int, n_opponents: int, team: int, attack_roles: List, defense_roles: List, role_assign_period: int = 1):
        super(RoleAgent, self).__init__(f"Role-based Agent")
        self.id = id
        self.n_agents = n_agents
        self.n_opponents = n_opponents
        self.role_assign_period = role_assign_period
        self.curr_role = None
        self.steps_counter = 0
        self.team = team
        self.attack_roles= attack_roles
        self.defense_roles=defense_roles

    def potential_function(self, agent_pos: Tuple, ball_position: Tuple, role: int, strategy: int):
        if strategy == ATTACKING:
            diamond_positions = self.get_diamond_positions(ball_position)
            role_target_pos = diamond_positions[role]
            return cityblock(agent_pos, role_target_pos)
        elif strategy == DEFENDING:
            return cityblock(agent_pos, ball_position)

    def role_assignment(self, teammates, roles, ball_position, strategy):
        roles_potentials = []
        for role in roles:
            role_potentials = []
            teammates_length=len(teammates)
            for teammate_i in range(teammates_length):
                teammate_pos = teammates[teammate_i]
                potential = self.potential_function(teammate_pos, ball_position, role, strategy)
                role_potentials.append((teammate_i, potential))
            role_potentials.sort(key=lambda x: x[1])
            roles_potentials.append(role_potentials)

        assigned_roles = [-1] * teammates_length
        for role_id, role_potentials in enumerate(roles_potentials):
            for agent_id, _ in role_potentials:
                if assigned_roles[agent_id] == -1:
                    assigned_roles[agent_id] = role_id
                    break

        return assigned_roles

    def action(self) -> int:

        my_position = self.observation[0]
        ball_position = self.observation[1]
        score = self.observation[2]
        agents_position = self.observation[2:2+self.n_agents]
        opponents_position = self.observation[2+self.n_agents:2+self.n_agents+self.n_opponents]

        print('Ball position: ' + str(ball_position))

        # Compute potential-based role assignment every `role_assign_period` steps.
        if self.curr_role is None or self.steps_counter % self.role_assign_period == 0:
           
            if self.id == 5:
                print('My Current ROle is: ' + str(self.curr_role))

            if self.my_team_has_ball(agents_position if self.team == AGENT_TEAM else opponents_position, ball_position):
                role_assignments = self.role_assignment(agents_position if self.team == AGENT_TEAM else opponents_position, self.attack_roles, ball_position, ATTACKING)
                self.curr_role = role_assignments[self.id if self.team == AGENT_TEAM else self.id - self.n_agents]
                print('CURRENT ROLE: ' + str(self.curr_role) + ' and ball carrier = ' + str(BALL_CARRIER))
                if self.curr_role == BALL_CARRIER:
                    print('I AM THE BOSS')
                    closest_opponents = self.find_closest_players(my_position, opponents_position if self.team == AGENT_TEAM else agents_position, 1)
                    if self.player_nearby(my_position=my_position, closest_opponent=closest_opponents[0], distance=2):
                        agent_i = self.find_best_player(agents_position if self.team == AGENT_TEAM else opponents_position, opponents_position if self.team == AGENT_TEAM else agents_position, my_position, AGENT_TEAM if self.team == AGENT_TEAM else OPPONENT_TEAM)
                        if agent_i is not None:
                            action = (PASS, agent_i)
                        else:
                            action = (DOWN if self.team == AGENT_TEAM else UP, None)
                    else:
                        action = (DOWN if self.team == AGENT_TEAM else UP, None)
                else:
                    # is teammate
                    action = self.advance_to_pos(my_position, ball_position, self.curr_role, ATTACKING)
            
            else:
                role_assignments = self.role_assignment(agents_position if self.team == AGENT_TEAM else opponents_position, self.defense_roles, ball_position, DEFENDING)
                self.curr_role = role_assignments[self.id if self.team == AGENT_TEAM else self.id - self.n_agents]
                # is defender or teammate
                action = self.advance_to_pos(my_position, ball_position, self.curr_role, DEFENDING)

        self.steps_counter += 1
        print('My id: ' + str(self.id) + ' ACTION' + action.__str__())
        print('--------------------------------------')

        return action

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

    def get_diamond_positions(self, ball_position: Tuple) -> List[Tuple]:

        def _validate_vertical_position(x, dx):
            next_x = x + dx
            if next_x < 0:
                return next_x + 1
            elif next_x > 10:
                return next_x - 1
            else:
                return next_x
            
        def _validate_horizontal_position(y, dy):
            next_y = y + dy
            if next_y < 0:
                return next_y + 1
            elif next_y > 20:
                return next_y - 1
            else:
                return next_y

        ball_x = ball_position[0]
        ball_y = ball_position[1]

        ball_carrier = ball_position
        supporter_left = (_validate_vertical_position(ball_x, -1), _validate_horizontal_position(ball_y, -1))
        supporter_right = (_validate_vertical_position(ball_x, -1), _validate_horizontal_position(ball_y, 1))
        sub_supporter_left = (_validate_vertical_position(ball_x, -2), _validate_horizontal_position(ball_y, -2))
        sub_supporter_right = (_validate_vertical_position(ball_x, -2), _validate_horizontal_position(ball_y, 2))
        winger_left = (_validate_vertical_position(ball_x, -3), _validate_horizontal_position(ball_y, -3))
        winger_right = (_validate_vertical_position(ball_x, -3), _validate_horizontal_position(ball_y, 3))

        if self.team == AGENT_TEAM:
            return [ball_carrier, supporter_left, supporter_right, sub_supporter_left, sub_supporter_right, winger_left, winger_right]
        else:
            return [(ball_x, ball_y), (ball_x + 1 , ball_y - 1), (ball_x + 1, ball_y + 1), (ball_x + 2, ball_y - 2), (ball_x + 2, ball_y + 2), (ball_x + 3, ball_y - 3), (ball_x + 3, ball_y + 3)]
    
    def player_nearby(self, my_position, closest_opponent, distance):
        return abs(my_position[0]-closest_opponent[0]) + abs(my_position[1]-closest_opponent[1]) <= distance


    def find_closest_players(self, my_position, opponent_positions, n_players):
        distances = {}

        for i, opponent_pos in enumerate(opponent_positions):
            distances[i] = cityblock(my_position, opponent_pos)

        sorted_opponents = sorted(distances.items(), key=lambda x: x[1])
        closest_opponents = [opponent_positions[i] for i, _ in sorted_opponents[:n_players]]

        return closest_opponents

    def advance_to_pos(self, my_position: Tuple, ball_position: Tuple, role: int, strategy: int) -> Tuple:
            
        if strategy == ATTACKING:

            diamond_positions = self.get_diamond_positions(ball_position)
            return self.go_toward_position(my_position, diamond_positions[role])
            
        elif strategy == DEFENDING:

            if role <= 3:
                return self.go_toward_position(my_position, ball_position)
            else:
                # stay 4 block behind the forwards line
                if self.team == AGENT_TEAM:
                    return self.go_toward_position(my_position, (ball_position[0] - DISTANCE_BETWEEN_LINES, ball_position[1]))
                elif self.team == OPPONENT_TEAM:
                    return self.go_toward_position(my_position, (ball_position[0] + DISTANCE_BETWEEN_LINES, ball_position[1]))

    def my_team_has_ball(self, team, ball_pos):
        for teammate_pos in team:
            if teammate_pos[0] == ball_pos[0] and teammate_pos[1] == ball_pos[1]:
                return True
        return False
    
    def go_toward_position(self, my_position, dest_position):
        my_pos_x, my_pos_y = my_position
        dest_x, dest_y = dest_position
        print('My role is: ' + str(self.curr_role) + ' and I want to go to: ' + str(dest_position))
        print('My position: x=' + my_pos_x.__str__() + ';y=' + my_pos_y.__str__())
        print('Dest position: x=' + dest_x.__str__() + ';y=' + dest_y.__str__())

        if my_pos_x == dest_x and my_pos_y == dest_y:
            return (STAY, None)
        
        dx = dest_x - my_pos_x
        dy = dest_y - my_pos_y

        if abs(dx) >= abs(dy):
            if dx > 0:
                return (DOWN, None)
            elif dy == 0:
                return (STAY, None)
            else:
                return (UP, None)
        else:
            if dy > 0:
                return (RIGHT, None)
            elif dy == 0:
                return (STAY, None)
            else:
                return (LEFT, None)


