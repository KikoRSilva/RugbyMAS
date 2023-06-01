import copy
import logging
import time

import numpy as np

from PIL import ImageColor
import gym
from gym import spaces
from gym.utils import seeding

from ma_gym.envs.utils.action_space import MultiAgentActionSpace
from ma_gym.envs.utils.observation_space import MultiAgentObservationSpace

AGENT_TEAM = 0
OPPONENT_TEAM = 1


class RugbyEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, grid_shape=(21, 11), n_agents=7, n_opponents=7, max_steps=100, penalty=-0.5, step_cost=-0.01, score_try_reward=5, opponent_move_probs=(0.175, 0.175, 0.175, 0.175, 0.3, 0), try_area=2):
        self._grid_shape = grid_shape
        self.n_agents = n_agents
        self.n_opponents = n_opponents
        self._max_steps = max_steps
        self._steps_count = None
        self._penalty = penalty
        self._step_cost = step_cost
        self._score_try_reward = score_try_reward
        self.team_scores = (0, 0)

        self.action_space = MultiAgentActionSpace([spaces.Discrete(6) for _ in range(self.n_agents + self.n_opponents)])
        self.agent_pos = {_: None for _ in range(self.n_agents)}
        self.opponents_pos = {_: None for _ in range(self.n_opponents)}
        self.ball_pos = None
        self._players_done = [False for _ in range(self.n_agents + self.n_opponents)]

        self._try_area = try_area
        self._base_grid = self.__create_grid()
        self._full_obs = self.__create_grid()
        self._agent_dones = [False for _ in range(self.n_agents)]
        self._opponent_move_probs = opponent_move_probs
        self.viewer = None

        # agent pos, ball pos, score, players pos
        self.observation_size = 1 + 1 + 1 + self.n_agents + self.n_opponents
        self.observation_space = MultiAgentObservationSpace(
            [spaces.Box(0.0, 1.0, shape=(self.observation_size,)) for _ in range(self.n_agents + self.n_opponents)])
        
        self._total_episode_reward = None
        self.seed()

    def __create_grid(self):
        _grid = []
        for col in range(self._grid_shape[0]):
            _row = []
            for row in range(self._grid_shape[1]):
                if col == self._try_area - 1 or col == self._grid_shape[0] - self._try_area:
                    _row.append(PRE_IDS['wall'])
                else:
                    _row.append(PRE_IDS['empty'])
            _grid.append(_row)

        return _grid
    
    def seed(self, n=None):
        self.np_random, seed = seeding.np_random(n)
        return [seed]
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def reset(self):
        # Reset player and ball positions, scores, and step count
        self._total_episode_reward = [0 for _ in range(self.n_agents + self.n_opponents)]
        self._base_grid = self.__create_grid()
        self._full_obs = self.__create_grid()
        self._players_done = [False for _ in range(self.n_agents + self.n_opponents)]
        self._team_scores = (0, 0)
        self._step_count = 0

        # Place players and ball on the grid
        self.__place_agents()
        self.__place_opponents()
        self.__place_ball()

        # Set initial observations
        observations = []
        for agent_i in range(self.n_agents):
            observation = self.__get_observation(AGENT_TEAM, agent_i)
            observations.append(observation)
            
        for opponent_i in range(self.n_opponents):
            observation = self.__get_observation(OPPONENT_TEAM, opponent_i)
            observations.append(observation)

        return observations

    def __place_agents(self):
        x = (self._grid_shape[0] - 1) // 4 # center left midfield
        # x = 17
        offset = (self._grid_shape[1] % self.n_agents) / 2
        for agent_i in range(self.n_agents):
            y = int(agent_i + offset)
            self.agent_pos[agent_i] = (x, y)
            self._full_obs[x][y] = PRE_IDS['agent'] + str(agent_i + 1)

    def __place_opponents(self):
        x = self._grid_shape[0] - 1 - self._grid_shape[0] // 4 # center right midfield
        offset = (self._grid_shape[1] % self.n_opponents) / 2
        for opponent_i in range(self.n_opponents):
            y = int(opponent_i + offset)
            self.opponents_pos[opponent_i] = (x, y)
            self._full_obs[x][y] = PRE_IDS['opponent'] + str(opponent_i + 1)

    def __place_ball(self):
        agent_i = np.random.randint(0, self.n_agents) # choose random agent
        self.ball_pos = self.agent_pos[agent_i]
        self._full_obs[self.ball_pos[0]][self.ball_pos[1]] = 'A' + str(agent_i + 1) + PRE_IDS['ball']
        
    def __get_observation(self, team, id):
        # agent pos, ball pos, score, players pos
        observation = np.array([(0, 0) for _ in range(1 + 1 + 1 + self.n_agents + self.n_opponents)])
        observation[0] = self.agent_pos[id] if team == AGENT_TEAM else self.opponents_pos[id]
        observation[1] = self.ball_pos
        observation[2] = self._team_scores
        observation[2:2+self.n_agents] = list(self.agent_pos.values())
        observation[2+self.n_agents:2+self.n_agents+self.n_opponents] = list(self.opponents_pos.values())
        return observation
    
    def step(self, actions):
        self._step_count += 1
        rewards = [self._step_cost for _ in range(self.n_agents + self.n_opponents)]

        for player_i, action in enumerate(actions):
           if not (self._players_done[player_i]):
                if player_i <= 6:
                    #print("Agent " + str(player_i) + " choose action " + str(action))
                    self.__update_agent_pos(player_i, action)    # agent ids [0 - 6]
                else:
                    #print("Opponent " + str(player_i - self.n_opponents) + " choose action " + str(action))
                    self.__update_opponent_pos(player_i - self.n_opponents, action)  # opponent ids [7 - 13]

        if (self._step_count >= self._max_steps):
            for i in range(self.n_agents + self.n_opponents):
                self._players_done[i] = True

        for i in range(self.n_agents + self.n_opponents):
            self._total_episode_reward[i] += rewards[i]

        observations = []
        for agent_i in range(self.n_agents):
            observation = self.__get_observation(AGENT_TEAM, agent_i)
            observations.append(observation)
            
        for opponent_i in range(self.n_opponents):
            observation = self.__get_observation(OPPONENT_TEAM, opponent_i)
            observations.append(observation)

        # time.sleep(0.5) 
        self.print_field()
        print("\n")
        print("Score: " + self._team_scores.__str__())
        print("Step: " + str(self._step_count))
        print("\n")
            
        return observations, rewards, self._players_done, {'score': self._team_scores}

    def __update_agent_pos(self, agent_i, move):

        curr_pos = copy.copy(self.agent_pos[agent_i])
        next_pos = None
        pass_the_ball = False

        if move[0] == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move[0] == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move[0] == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move[0] == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move[0] == 4:  # stay
            pass
        elif move[0] == 5:  # pass theball
            # TODO: Check if has the ball or any teammate
            pass_the_ball = True
            pass
        else:
            raise Exception('Action Not found!')

        # If move and there is no one in cell
        # else:
        #   TODO: Pass the ball
        if next_pos is not None and self._is_cell_vacant(next_pos):

            # is ball carrier
            if self.ball_pos == curr_pos:

                self.ball_pos = next_pos
                self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
                self.agent_pos[agent_i] = next_pos
                self.__update_agent_view(agent_i, has_ball=True)


                # can score try ?
                if next_pos[0] >= self._grid_shape[0] - self._try_area:
                    time.sleep(3)
                    self._score_try(AGENT_TEAM)
                    # TODO: Score Try Reward
 
            # is teammate
            elif self.team_has_ball(AGENT_TEAM):

                if next_pos[1] <= self.ball_pos[1]:
                    self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
                    self.agent_pos[agent_i] = next_pos
                    self.__update_agent_view(agent_i)
                else:
                    print("CANNOT MOVE TOWARDS THE BALL CARRIER")

            else:
                # is defender
                # TODO: advance towards the ball carrier
                pass
        # pass or stay
        else:
            if pass_the_ball:
                # has the ball
                if self.ball_pos == curr_pos:
                    if self.agent_pos[move[1]][1] <= curr_pos[1]:
                        # pass it
                        self.ball_pos = self.agent_pos[move[1]]
                        self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['agent'] + str(agent_i + 1)
                        self._full_obs[self.agent_pos[move[1]][0]][self.agent_pos[move[1]][1]] = PRE_IDS['agent'] + str(move[1] + 1) + 'B'  
                    else:
                        print("CANNOT PASS FORWARD THE BALL")   

    def __update_opponent_pos(self, opponent_i, move):

        curr_pos = copy.copy(self.opponents_pos[opponent_i])
        next_pos = None
        pass_the_ball = False

        if move[0] == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move[0] == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move[0] == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move[0] == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move[0] == 4:  # stay
            pass
        elif move[0] == 5:  # pass theball
            # TODO: Check if has the ball or any teammate
            pass_the_ball = True
            pass
        else:
            raise Exception('Action Not found!')

        # If move and there is no one in cell
        # else:
        #   TODO: Pass the ball
        if next_pos is not None and self._is_cell_vacant(next_pos):

            # is ball carrier
            if self.ball_pos == curr_pos:

                self.ball_pos = next_pos
                self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
                self.opponents_pos[opponent_i] = next_pos
                self.__update_opponent_view(opponent_i, has_ball=True)

                # can score try ?
                if next_pos[0] >= (self._try_area - 1):
                    time.sleep(3)
                    self._score_try(OPPONENT_TEAM)
                    # TODO: Score Try Reward
 
            # is teammate
            elif self.team_has_ball(OPPONENT_TEAM):

                if next_pos[1] <= self.ball_pos[1]:
                    self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
                    self.opponents_pos[opponent_i] = next_pos
                    self.__update_opponent_view(opponent_i)
                else:
                    print("CANNOT MOVE TOWARDS THE BALL CARRIER")

            else:
                # is defender
                self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
                self.opponents_pos[opponent_i] = next_pos
                self.__update_opponent_view(opponent_i)
                # TODO: advance towards the ball carrier
               
        # pass or stay
        else:
            if pass_the_ball:
                # has the ball
                if self.ball_pos == curr_pos:
                    # pass it
                    self.ball_pos = self.opponents_pos[move[1]]
                    self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['opponent'] + str(opponent_i + 1)
                    self._full_obs[self.opponents_pos[move[1]][0]][self.opponents_pos[move[1]][1]] = PRE_IDS['opponent'] + str(move[1] + 1) + 'B'   

    def __update_agent_view(self, agent_i, has_ball=False):
        self._full_obs[self.agent_pos[agent_i][0]][self.agent_pos[agent_i][1]] = PRE_IDS['agent'] + str(agent_i + 1) + ('B' if has_ball else '')

    def __update_opponent_view(self, opponent_i, has_ball=False):
        self._full_obs[self.opponents_pos[opponent_i][0]][self.opponents_pos[opponent_i][1]] = PRE_IDS['opponent'] + str(opponent_i + 1) + ('B' if has_ball else '')

    def _is_cell_vacant(self, pos):
        return self.is_valid(pos) and (self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty'] or self._full_obs[pos[0]][pos[1]] == PRE_IDS['wall'])
    
    def is_valid(self, pos):
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])
    
    def _score_try(self, team):
        agent_score, opponent_score = self._team_scores
        if team == AGENT_TEAM:
            self._team_scores = (agent_score + 5, opponent_score)
        else:
            self._team_scores = (agent_score, opponent_score + 5)
    
    def team_has_ball(self, team):
        if team == AGENT_TEAM:
            for pos in self.agent_pos.values():
                if pos == self.ball_pos:
                    return True
            return False
        else:
            for pos in self.opponents_pos.values():
                if pos == self.ball_pos:
                    return True
            return False
    
    def print_field(self):
        
        for col in range(self._grid_shape[1]):
            column_values = [row[col] for row in self._full_obs]
            print(column_values)
    
    def render(self, mode='human'):
        # render the game state
        # ...
        pass

AGENT_COLOR = ImageColor.getcolor('blue', mode='RGB')
AGENT_NEIGHBORHOOD_COLOR = (186, 238, 247)
OPPONENT_COLOR = 'red'

CELL_SIZE = 35

WALL_COLOR = 'black'

ACTION_MEANING = {
    0: "DOWN",
    1: "LEFT",
    2: "UP",
    3: "RIGHT",
    4: "STAY",
    5: "PASS"
}

PRE_IDS = {
    'agent': 'A',
    'opponent': 'O',
    'empty': ' ',
    'wall': '|',
    'ball': 'B'
}