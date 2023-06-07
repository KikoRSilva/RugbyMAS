import numpy as np
from typing import List, Tuple

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


class RoleAgent(Agent):

    def __init__(self, agent_id: int, n_agents: int, roles: List, team: int, attack_roles: List, defense_roles: List, role_assign_period: int = 1):
        super(RoleAgent, self).__init__(f"Role-based Agent")
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.role_assign_period = role_assign_period
        self.curr_role = None
        self.steps_counter = 0
        self.team = team
        self.attack_roles= attack_roles
        self.defense_roles=defense_roles

    def nearest_to_the_left(self, my_position, colleagues_position):
        """
        We are x2 and we may be in "p" and just in 1 of these ps.
        x1 corresponds to our nearest colleague
        | p | p | p |  p |
        ------------------
        | O | O | x1 | O |
        ------------------
        | O | O | O  | O |

        """
        distances = {}
        colleagues_at_the_left = []
        for i, colleague_position in enumerate(colleagues_position):
            if colleague_position[1]<= my_position[1]:
                return None

        #for i, coleague_pos in enumerate(colleagues_position):
        #    distances[i] = cityblock(my_position, coleague_pos)

        #sorted_colleagues = sorted(distances.items(), key=lambda x: x[1])
        
        return closest_colleagues[0]

    def potential_function(self, agent_pos: Tuple, ball_position: Tuple, role: int, strategy: int):
        if strategy == ATTACKING:
            diamond_positions = self.get_diamond_positions(ball_position)
            role_target_pos = diamond_positions[role]
            return cityblock(agent_pos, role_target_pos)
        elif strategy == DEFENDING:
            return cityblock(agent_pos, ball_position)

    def role_assignment(self, teammates, roles, ball_position):
        roles_potentials = []
        for role in roles:
            role_potentials = []
            num_of_teammates=len(teammates)
            for agent_id in range(num_of_teammates):
                agent_pos = teammates[agent_id]
                potential = self.potential_function(agent_pos, ball_position, role, ATTACKING)
                role_potentials.append((agent_id, potential))
            role_potentials.sort(key=lambda x: x[1])
            roles_potentials.append(role_potentials)

        assigned_roles = [-1] * self.n_agents
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

        # Compute potential-based role assignment every `role_assign_period` steps.
        if self.curr_role is None or self.steps_counter % self.role_assign_period == 0:
            if self.team == AGENT_TEAM:
                if self.my_team_has_ball(agents_position, ball_pos):
                    role_assignments = self.role_assignment(agents_position, self.attack_roles, ball_position)
                    self.curr_role = role_assignments[self.agent_id]

        prey_pos = self.observation[self.n_agents * 2:]
        agent_pos = (self.observation[self.agent_id * 2], self.observation[self.agent_id * 2 + 1])
        self.steps_counter += 1

        return self.advance_to_pos(agent_pos, prey_pos, self.curr_role)

    def get_diamond_positions(self, ball_position: Tuple) -> List[Tuple]:
        ball_x = ball_position[0]
        ball_y = ball_position[1]
        return [(ball_x-1 , ball_y - 1), (ball_x-1, ball_y + 1), (ball_x - 2, ball_y-2), (ball_x -2, ball_y+2),(ball_x-3, ball_y-3), (ball_x-3, ball_y+3)]

    def advance_to_pos(self, agent_pos: Tuple, prey_pos: Tuple, agent_dest: int) -> int:
        """
        Choose movement action to advance agent towards the destination around prey

        :param agent_pos: current agent position
        :param prey_pos: prey position
        :param agent_dest: agent destination in relation to prey (0 for NORTH, 1 for SOUTH,
                            2 for WEST, and 3 for EAST)

        :return: movement index
        """

        def _move_vertically(distances) -> int:
            if distances[1] > 0:
                return DOWN
            elif distances[1] < 0:
                return UP
            else:
                return STAY

        def _move_horizontally(distances) -> int:
            if distances[0] > 0:
                return RIGHT
            elif distances[0] < 0:
                return LEFT
            else:
                return STAY

        prey_adj_locs = self.get_prey_adj_locs(prey_pos)
        distance_dest = np.array(prey_adj_locs[agent_dest]) - np.array(agent_pos)
        abs_distances = np.absolute(distance_dest)
        if abs_distances[0] > abs_distances[1]:
            return _move_horizontally(distance_dest)
        elif abs_distances[0] < abs_distances[1]:
            return _move_vertically(distance_dest)
        else:
            roll = np.random.uniform(0, 1)
            return _move_horizontally(distance_dest) if roll > 0.5 else _move_vertically(d)

    def find_closest_colleagues(self, my_position, colleagues_position, n_players):
        distances = {}

        for i, coleague_pos in enumerate(colleagues_position):
            distances[i] = cityblock(my_position, coleague_pos)

        sorted_colleagues = sorted(distances.items(), key=lambda x: x[1])
        closest_colleagues = [colleagues_position[i] for i, _ in sorted_opponents[:n_players]]

        return closest_colleagues

    def my_team_has_ball(self, team, ball_pos):
        for teammate_pos in team:
            if teammate_pos[0] == ball_pos[0] and teammate_pos[1] == ball_pos[1]:
                return True
        return False


