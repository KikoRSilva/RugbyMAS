import random

class Opponent:
    def __init__(self, env, team):
        self.team = team
        self.env = env
        self.position = self.random_position()
        
    def random_position(self):
        # Generate a random position within the opponent's half of the field
        x = random.uniform(self.env.field_length/2, self.env.field_length)
        y = random.uniform(0, self.env.field_width)
        return (x, y)
    
    def move_towards_ball_carrier(self):
        # Move towards the ball carrier
        ball_carrier = self.env.get_ball_carrier()
        if ball_carrier is not None:
            dx = ball_carrier.position[0] - self.position[0]
            dy = ball_carrier.position[1] - self.position[1]
            norm = (dx**2 + dy**2)**0.5
            if norm > 0:
                dx /= norm
                dy /= norm
                self.position = (self.position[0]+dx*self.env.player_speed, self.position[1]+dy*self.env.player_speed)
                
    def tackle(self):
        # Tackle the ball carrier if within range
        ball_carrier = self.env.get_ball_carrier()
        if ball_carrier is not None:
            dx = ball_carrier.position[0] - self.position[0]
            dy = ball_carrier.position[1] - self.position[1]
            dist = (dx**2 + dy**2)**0.5
            if dist < self.env.tackle_distance:
                ball_carrier.tackled = True
                self.env.ball.possession = None
                
    def update(self):
        # Move towards the ball carrier and attempt to tackle
        self.move_towards_ball_carrier()
        self.tackle()
