import numpy as np
import gym
from gym import spaces
from rugby_field import RugbyField
from ball import Ball
from ..agents.agent import Agent

RUGBY_FIELD_WIDTH = 21
RUGBY_FIELD_HEIGHT = 11
RUGBY_FIELD_TRY_AREA_WIDTH = 2

class RugbyEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.field = RugbyField(RUGBY_FIELD_HEIGHT, RUGBY_FIELD_WIDTH, RUGBY_FIELD_TRY_AREA_WIDTH)
        self.ball = Ball()
        self.teams = {"x": [], "o": []}
        self.team_with_ball = None
        self.opponent_team = None
        self.reset()

        # define action and observation spaces
        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Discrete(21 * 11 + 1)

    def reset(self):
        # initialize teams
        for team in self.teams:
            for i in range(7):
                if i < 3:
                    self.teams[team].append(Agent(team, i, "forward"))
                else:
                    self.teams[team].append(Agent(team, i, "back"))

        # reset the RugbyField
        self.field.reset()
        
        # place agents at the center of the midfield
        for i in range(7):
            self.teams["x"][i].position = (6, i + 2)
            self.teams["o"][i].position = (14, i + 2)

        # reset the score board
        self.try_scores = {'x': 0, 'o': 0}

        # set the team with the ball possession
        self.team_with_ball = "x"
        self.opponent_team = "o"

        # place the ball at the center of the midfield
        self.ball.position = (np.random.randint(0, RUGBY_FIELD_WIDTH), np.random.randint(0, RUGBY_FIELD_HEIGHT))
        self.team_with_ball = 'x' if np.random.rand() < 0.5 else 'o'
        if self.team_with_ball == 'x':
            self.field.field_matrix[self.ball.position] = 3
        else:
            self.field.field_matrix[self.ball.position] = 4
        

        # return the initial state
        obs = {"field": self.field.get_matrix(),
               "ball": self.ball.position,
               "team_with_ball": self.team_with_ball,
               "opponent_team": self.opponent_team}
        
        for team in self.teams:
            for agent in self.teams[team]:
                obs[f"{team}_agent_{agent.id}"] = agent.position
        return obs

    def step(self, action):
        # implement the game logic
        reward = 0
        done = False
        
        # decode the action
        row = action // 11
        col = action % 11
        
        # check if the ball carrier is performing a valid action
        if self.field.field_matrix[self.ball.position] == 3:
            if (row, col) not in self.get_valid_actions(self.ball.position):
                reward = -1
                return obs, reward, done, {}
            
        # move the ball carrier or the opponents
        if self.field.field_matrix[self.ball.position] == 3:
            # move the ball carrier
            self.field.field_matrix[self.ball.position] = 0
            self.ball.position = (row, col)
            self.field.field_matrix[self.ball.position] = 3
        else:
            # move the opponents
            for player_pos in self.players['o']:
                self.field.field_matrix[player_pos] = 0
                player_row, player_col = player_pos
                if row > player_row:
                    player_row += 1
                elif row < player_row:
                    player_row -= 1
                if col > player_col:
                    player_col += 1
                elif col < player_col:
                    player_col -= 1
                self.field.field_matrix[(player_row, player_col)] = 2
                player_pos = (player_row, player_col)
        
        # check if ball carrier reaches try area
        if self.field.field_matrix[self.ball.position] == 3 and self.ball.position in self.try_areas['x']:
            reward = 1
            done = True
        
        obs = {
            "field": self.field.get_matrix(),
            "ball": self.ball.position,
            "team_with_ball": self.team_with_ball,
            "opponent_team": self.opponent_team
        }

        for team in self.teams:
            for agent in self.teams[team]:
                obs[f"{team}_agent_{agent.id}"] = agent.position
                
        return obs, reward, done, {}
    
    def get_valid_actions(self, position):
        row, col = position
        valid_actions = []
        if row > 0:
            valid_actions.append((row - 1) * 11 + col)
        if row < 20:
            valid_actions.append((row + 1) * 11 + col)
        if col > 0:
            valid_actions.append(row * 11 + col - 1)
        if col < 10:
            valid_actions.append(row * 11 + col + 1)
        return valid_actions
    
    def render(self, mode='human'):
        # render the game state
        # ...
        pass
