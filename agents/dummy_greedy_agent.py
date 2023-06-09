import math
import random
from scipy.spatial.distance import cityblock
from .base_agent import Agent

AGENT_TEAM = 0
OPPONENT_TEAM = 1
ACTIONS = 6
DOWN, LEFT, UP, RIGHT, STAY, PASS = range(ACTIONS)


class DummyGreedyAgent(Agent):
    def __init__(
        self, id: int, n_actions: int, n_agents: int, n_opponents: int, team: int
    ):
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
        agents_position = self.observation[2 : 2 + self.n_agents]
        opponents_position = self.observation[
            2 + self.n_agents : 2 + self.n_agents + self.n_opponents
        ]

        if my_position[0] == ball_position[0] and my_position[1] == ball_position[1]:
            if self.team == AGENT_TEAM:
                closest_opponent, _ = self.find_closest_player(
                    my_position, opponents_position
                )
                closest_bro, closest_bro_i = self.find_closest_teammate(
                    my_position, agents_position, AGENT_TEAM
                )
            else:
                closest_opponent, _ = self.find_closest_player(
                    my_position, agents_position
                )
                closest_bro, closest_bro_i = self.find_closest_teammate(
                    my_position, opponents_position, OPPONENT_TEAM
                )

            bro_found = closest_bro is not None
            opponent_found = closest_opponent is not None

            if opponent_found and bro_found:
                action = self._decide(my_position, closest_opponent, closest_bro_i)
            else:
                action = (DOWN, None) if AGENT_TEAM == self.team else (UP, None)
        else:
            action = (random.randrange(ACTIONS - 1), None)

        return action

    def _decide(self, my_position, closest_opponent, closest_bro):
        if (
            abs(my_position[0] - closest_opponent[0])
            + abs(my_position[1] - closest_opponent[1])
            <= 2
        ):
            return (PASS, closest_bro)
        else:
            return (DOWN, None) if AGENT_TEAM == self.team else (UP, None)

    def find_closest_teammate(self, my_position, teammates, team):
        min = math.inf
        closest_teammate_position = None
        closest_teammate_index = None
        teammates_length = len(teammates)

        for p in range(teammates_length):
            teammate = teammates[p]
            if AGENT_TEAM == team:
                if (
                    not (
                        teammate[0] == my_position[0] and teammate[1] == my_position[1]
                    )
                    and my_position[0] >= teammate[0]
                ):
                    distance = cityblock(my_position, teammate)
                    if distance < min:
                        min = distance
                        closest_teammate_position = teammate
                        closest_teammate_index = p
            else:
                if (
                    not (
                        teammate[0] == my_position[0] and teammate[1] == my_position[1]
                    )
                    and my_position[0] <= teammate[0]
                ):
                    distance = cityblock(my_position, teammate)
                    if distance < min:
                        min = distance
                        closest_teammate_position = teammate
                        closest_teammate_index = p

        return closest_teammate_position, closest_teammate_index

    def find_closest_player(self, my_position, players_position):
        min = math.inf
        closest_player_position = None
        closest_player_index = None
        players_length = len(players_position)

        for p in range(players_length):
            opponent_position = players_position[p]

            if not (
                opponent_position[0] == my_position[0]
                and opponent_position[1] == my_position[1]
            ):
                distance = cityblock(my_position, opponent_position)

                if distance < min:
                    min = distance
                    closest_player_position = opponent_position
                    closest_player_index = p

        return closest_player_position, closest_player_index
