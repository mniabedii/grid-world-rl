def print_policy(env, agent):
    ARROWS = {0: "↑", 1: "↓", 2: "←", 3: "→"}

    for r in range(env.rows):
        for c in range(env.cols):

            pos = (r, c)

            # wall
            if pos in env.walls:
                print("X", end=" ")

            # goal
            elif pos == env.goal:
                print("G", end=" ")

            # start (optional highlight)
            elif pos == env.start:
                print("S", end=" ")

            else:
                best_action = np.argmax(agent.q_table[r, c])
                print(ARROWS[best_action], end=" ")

        print()

    print()
