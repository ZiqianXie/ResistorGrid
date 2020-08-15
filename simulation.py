from ResistorNetwork import ResistorNetwork
import numpy as np


N_resistor_horizontal = 40
N_resistor_vertical = 60
r_horizontal = N_resistor_vertical / N_resistor_horizontal
r_vertical_1 = 1
r_vertical_2 = 0.5


def criteria_circle(pos):
    center = (N_resistor_horizontal//2, N_resistor_vertical//2)
    radius_sq = 15 ** 2
    if (pos[0] - center[0]) ** 2 + (pos[1] - center[1]) ** 2 <= radius_sq:
        return True
    return False


def criteria_sq(pos):
    x, y = pos
    if 10 <= x <= 30 and 20 <= y <= 40:
        return True
    return False


def simulation(criteria, r_horizontal=r_horizontal, r_vertical_1=r_vertical_1, r_vertical_2=r_vertical_2):
    grid = ResistorNetwork(N_resistor_horizontal + 1, (0, 0))
    r = []
    direction = (1, 0)
    pos = (0, 0)
    resume = False
    for i in range((N_resistor_horizontal + 1) * (N_resistor_vertical + 1) - 1):
        new_pos = (pos[0] + direction[0], pos[1] + direction[1])
        if criteria(new_pos):
            r_horizontal_origin, r_vertical_1_origin, r_vertical_2_origin = r_horizontal, r_vertical_1, r_vertical_2
            r_horizontal, r_vertical_1, r_vertical_2 = [1.2, 1, np.inf]
            resume = True
        if direction == (0, 1):
            grid.connect(pos, new_pos, r_vertical_1)
        else:
            grid.connect(pos, new_pos, r_horizontal)
        pos = new_pos
        if pos[1] > 0:
            node_below = (pos[0], pos[1] - 1)
            if direction != (0, 1):
                grid.connect(pos, node_below, r_vertical_2)
            if node_below != (0, 0):
                grid.free(node_below)
        if pos[0] == N_resistor_horizontal and direction == (1, 0):
            direction = (0, 1)
        elif pos[0] == N_resistor_horizontal and direction == (0, 1):
            direction = (-1, 0)
        elif pos[0] == 0 and direction == (-1, 0):
            direction = (0, 1)
        elif pos[0] == 0 and direction == (0, 1):
            direction = (1, 0)
        if resume:
            r_horizontal, r_vertical_1, r_vertical_2 = r_horizontal_origin, r_vertical_1_origin, r_vertical_2_origin
            resume = False
        r.append(grid.get_resistance((0, 0), pos))
    return r


r1 = simulation(criteria_circle)
r2 = simulation(criteria_sq)
