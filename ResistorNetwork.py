import numpy as np
np.seterr(invalid="ignore")


class ResistorNetwork:
    def __init__(self, max_num_node, root_name=0):
        self._current_node = {root_name: 0}
        self._unused_index = set(range(1, max_num_node+2))
        self._pairwise_r = np.ones((max_num_node+2, max_num_node+2)) * np.inf
        self._pairwise_r[0, 0] = 0

    def connect(self, node_i, node_j, r):
        assert node_i in self._current_node or node_j in self._current_node, "neither node is in the network"
        if node_i not in self._current_node:
            return self._connect_new(node_i, node_j, r)
        if node_j not in self._current_node:
            return self._connect_new(node_j, node_i, r)
        return self._connect(node_i, node_j, r)

    def _connect_new(self, new_node, node, r):
        new_node_index = self._unused_index.pop()
        self._current_node[new_node] = new_node_index
        node_index = self._current_node[node]
        self._pairwise_r[new_node_index, :] = self._pairwise_r[node_index, :] + r
        self._pairwise_r[new_node_index, new_node_index] = 0
        self._pairwise_r[:, new_node_index] = self._pairwise_r[new_node_index, :]

    def _connect(self, node_i, node_j, r):
        i_index = self._current_node[node_i]
        j_index = self._current_node[node_j]
        self._pairwise_r -= (((self._pairwise_r[j_index, :] - self._pairwise_r[i_index, :]).reshape(-1, 1) -
                              (self._pairwise_r[:, j_index] - self._pairwise_r[:, i_index]).reshape(1, -1)) ** 2 /
                             (r + self._pairwise_r[i_index, j_index])/4)

    def free(self, node):  # if a node will no longer make further connections, it can be freed.
        self._unused_index.add(self._current_node[node])
        del self._current_node[node]

    def get_resistance(self, node_i, node_j):
        assert node_i in self._current_node, f"{node_i} not in the network"
        assert node_j in self._current_node, f"{node_j} not in the network"
        i_index = self._current_node[node_i]
        j_index = self._current_node[node_j]
        return self._pairwise_r[i_index, j_index]


if __name__ == "__main__":
    N_resistor_horizontal = 24
    N_resistor_vertical = 24
    r_horizontal = N_resistor_vertical/N_resistor_horizontal
    r_vertical_1 = 1 # overlapping resistance
    r_vertical_2 = 0.5
    defect_indices = {40, 41, 42, 43, 44, 45}
    grid = ResistorNetwork(N_resistor_horizontal+1, (0, 0))
    r = []
    direction = (1, 0)
    pos = (0, 0)
    for i in range((N_resistor_horizontal+1)*(N_resistor_vertical+1)-1):
        new_pos = (pos[0] + direction[0], pos[1] + direction[1])
        if i in defect_indices:
            r_horizontal_origin, r_vertical_1_origin, r_vertical_2_origin = r_horizontal, r_vertical_1, r_vertical_2
            r_horizontal, r_vertical_1, r_vertical_2 = [2, 1, np.inf]
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
        if i in defect_indices:
            r_horizontal, r_vertical_1, r_vertical_2 = r_horizontal_origin, r_vertical_1_origin, r_vertical_2_origin
        r.append(grid.get_resistance((0, 0), pos))




