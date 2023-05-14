print("Hello representation line")
class Line:
    def __init__(self, m, b):
        self.m = m  # slope of the line
        self.b = b  # y-intercept of the line

    def __repr__(self):
        return f"y = {self.m}x + {self.b}"

# create a line object with slope = 2 and y-intercept = 3
line = Line(2, 3)

# print the line representation
print(line)
