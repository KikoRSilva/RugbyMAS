import numpy as np
from typing import List, Tuple

from .base_agent import Agent

AGENT_TEAM = 0
OPPONENT_TEAM = 1

ACTIONS = 6
DOWN, LEFT, UP, RIGHT, STAY, PASS = range(ACTIONS)

ROLES = 7
BALL_CARRIER, RIGHT_SUPPORTER, LEFT_SUPPORTER, RIGHT_SUB_SUPPORTER, LEFT_SUB_SUPPORTER, RIGH_WING, LEFT_WING = range(ROLES)


class RoleAgent(Agent):

    def __init__(self, agent_id: int, n_agents: int, roles: List, role_assign_period: int = 1)):
        super(RoleAgent, self).__init__(f"Role-based Agent")
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.roles = roles
        self.role_assign_period = role_assign_period
        self.curr_role = None
        self.steps_counter = 0


    def potential_function(self, agent_pos: Tuple, ball_pos: Tuple,oponents_positions: List, role: int):
        """
        Calculates the potential function used for role assignment.
        The potential function consists of the negative Manhattan distance between the
        `agent_pos` and the target position of the given `role` (which corresponds
        to a position that is adjacent to the position of the prey).

        :param agent_pos: agent position
        :param prey_pos: prey position
        :param role: role

        :return: (float) potential value
        """
        # 2 prox da bola -> supporters;
        # 1 prox do supporters -> sub-supporters
        # 1 mais próximo das primeiras 3 linhas -> winger left/right
        prey_adj_locs = self.get_prey_adj_locs(prey_pos)
        role_target_pos = prey_adj_locs[role]
        return cityblock(agent_pos, role_target_pos)

    def role_assignment(self):
        """
        Given the observation vector containing the positions of all predators
        and the prey(s), compute the role-assignment for each of the agents.

        :return: a list with the role assignment for each of the agents
        """
        prey_pos = self.observation[self.n_agents * 2:]
        agent_positions = self.observation[:self.n_agents * 2]

        roles_potentials = []
        for role in self.roles:
            role_potentials = []
            for agent_id in range(self.n_agents):
                agent_pos = agent_positions[agent_id * 2], agent_positions[agent_id * 2 + 1]
                potential = self.potential_function(agent_pos, prey_pos, role)
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

        # Compute potential-based role assignment every `role_assign_period` steps.
        if self.curr_role is None or self.steps_counter % self.role_assign_period == 0:
            role_assignments = self.role_assignment()
            self.curr_role = role_assignments[self.agent_id]

        prey_pos = self.observation[self.n_agents * 2:]
        agent_pos = (self.observation[self.agent_id * 2], self.observation[self.agent_id * 2 + 1])
        self.steps_counter += 1

        return self.advance_to_pos(agent_pos, prey_pos, self.curr_role)

    def get_prey_adj_locs(self, loc: Tuple) -> List[Tuple]:
        prey_x = loc[0]
        prey_y = loc[1]
        return [(prey_x, prey_y - 1), (prey_x, prey_y + 1), (prey_x - 1, prey_y), (prey_x + 1, prey_y)]

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

