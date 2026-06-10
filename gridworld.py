class GridWorld:
    def __init__(self):
        self.rows = 5
        self.cols = 5

        self.start = (0, 0)
        self.goal = (4, 4)

        self.agent_pos = self.start

    def reset(self):
        self.agent_pos = self.start

    def print_grid(self):
        for r in range(self.rows):
            for c in range(self.cols):

                if (r, c) == self.agent_pos:
                    print("A", end=" ")

                elif (r, c) == self.start:
                    print("S", end=" ")

                elif (r, c) == self.goal:
                    print("G", end=" ")

                else:
                    print(".", end=" ")

            print()

        print()


env = GridWorld()
env.print_grid()
