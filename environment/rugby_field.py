import numpy as np

class RugbyField:
    def __init__(self, field_width = 11, field_height = 21, try_area_width = 2):
        self.field_width = field_width
        self.field_height = field_height
        self.try_area_width = try_area_width
        self.midfield = self.field_width // 2

        self.field_matrix = np.zeros((self.field_height, self.field_width))

        # Set the try areas for each team
        left_try_area = slice(0, self.field_height), slice(0, self.try_area_width)
        right_try_area = slice(0, self.field_height), slice(self.field_width - self.try_area_width, self.field_width)
        self.field_matrix[left_try_area] = 1
        self.field_matrix[right_try_area] = 1

    def reset(self):
        self.field_matrix = np.zeros((self.field_height, self.field_width))
        # place agents at the center of the midfield
        for i in range(7):
            self.field_matrix[6, i+2] = 'x'
            self.field_matrix[14, i+2] = 'o'

    def print_field(self):
        print(self.field_matrix)

    def get_matrix(self):
        return self.field_matrix

