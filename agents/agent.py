class Agent:
    def __init__(self, id, team):
        self.id = id
        self.team = team
        self.position = None
        self.observation = None

    def set_position(self, position):
        self.position = position

    def move(self, direction):
        # Logic to move the agent in the specified direction
        if direction == "up":
            self.position[0] -= 1
        elif direction == "down":
            self.position[0] += 1
        elif direction == "left":
            self.position[1] -= 1
        elif direction == "right":
            self.position[1] += 1

        # Ensure the agent stays within the field boundaries
        self.position[0] = max(0, min(self.position[0], self.field_height - 1))
        self.position[1] = max(0, min(self.position[1], self.field_width - 1))

    def is_beside_or_behind(self, position):
        row_diff = position[0] - self.position[0]
        col_diff = position[1] - self.position[1]

        # Check if the teammate is either in the same row or behind the agent
        if self.is_same_row(position) and abs(col_diff) <= 1:
            return True

        # Check if the teammate is in the same column and behind the agent
        if self.is_same_column(position) and row_diff >= 0:
            return True

        return False

    def pass_ball(self, teammate):
        # Check if the teammate is beside or behind the agent
        if self.is_beside_or_behind(teammate.position):
            return teammate

        # If no valid passing option is found, return None
        return None

    def score_try(self):
        # Check if the agent has the ball
        if self.has_ball:
            # Check if the agent is in the opponent's try area
            if self.position[0] <= 1:
                return "score_try"

        return None

    def choose_action(self, observation):
        self.observation = observation

    def observe(self, observation):
        # Logic to process the observation
        pass

    def update(self, reward):
        # Logic to update the agent based on the received reward
        pass
