class Ball:
    def __init__(self, position=(0, 0)):
        self.position = position
    
    def move(self, new_position):
        self.position = new_position
