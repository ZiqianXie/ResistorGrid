from ResistorNetwork import ResistorNetwork
import numpy as np
from matplotlib import pyplot as plt


n_resistor_x = 20
n_resistor_y = 20
n_resistor_z = 10
nx = n_resistor_x + 1
ny = n_resistor_y + 1
nz = n_resistor_z + 1


class CubeTrajectory:
    def __init__(self, start_pos=(0, 0, 0), nx=nx,
                  ny=ny, nz=nz, dir_long=(1, 0, 0), dir_short=(0, 1, 0)):
        self.n = (nx, ny, nz)
        self.area = nx * ny
        self.vol = nx * ny * nz
        self.pos = np.array(start_pos)
        self.dir_cur = np.array(dir_long)
        self.dir_short = np.array(dir_short)
        self.dir_long = np.array(dir_long)
        self.cnt = 1

    @staticmethod
    def vec(ind, val):
        n = np.zeros(3, dtype='i')
        n[ind] = val
        return n

    def step(self):
        if self.cnt == self.vol:
            return self.pos, self.pos
        prev_pos = self.pos.copy()
        self.pos += self.dir_cur
        self.cnt += 1
        layer = self.pos[2] % 2
        if self.cnt % self.area == 0:
            self.dir_cur = np.array([0, 0, 1])
            self.dir_long = self.vec(1 - layer, 2 * (self.pos[1 - layer] == 0) - 1)
            self.dir_short = self.vec(layer, 2 * (self.pos[layer] == 0) - 1)
            return tuple(self.pos), tuple(prev_pos)
        proj = self.cnt % self.n[layer]
        if proj == 0:
            self.dir_cur = self.dir_short
            self.dir_long = -1 * self.dir_long
        elif proj == 1:
            self.dir_cur = self.dir_long
        return tuple(self.pos), tuple(prev_pos)


def simulation(start, grid, trajectory, defect_pos, neighbor_con):
    r = []
    for i in range(trajectory.vol - 1):
        r.append(neighbor_con(start, grid, trajectory, defect_pos))
    return r


if __name__ == "__main__":
    r_u = 1  # resistance along the printed line
    r_o = 2  # within layer overlapping resistance
    r_z = 1  # between layer contacting resistance
    r_u_def = 2  # resistance along the printed line when defect
    r_o_def = np.inf  # within layer overlapping resistance when defect
    r_z_def = 2  # between layer contacting resistance when defect
    start = (0, 0, 0)
    grid = ResistorNetwork(nx * ny, start)
    traj = CubeTrajectory(start)

    def defect_pos(pos):
        return False

    def neighbor_con(start, grid, trajectory, defect_pos):
        # print(len(grid._unused_index))
        def within_boundary(pos):
            x, y, z = pos
            if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
                return True
            return False

        pos, prev_pos = trajectory.step()
        if defect_pos(pos):
            r1 = r_u_def
            r2 = r_o_def
            r3 = r_z_def
        else:
            r1 = r_u
            r2 = r_o
            r3 = r_z
        grid.connect(pos, prev_pos, r1)
        pos_overlap = tuple(np.array(pos) - traj.dir_short)
        if within_boundary(pos_overlap) and pos_overlap != prev_pos:
            grid.connect(pos, pos_overlap, r2)
        pos_below = tuple(np.array(pos) - np.array([0, 0, 1]))
        if within_boundary(pos_below):
            if pos_below != prev_pos:
                grid.connect(pos, pos_below, r3)
            if pos_below != start:
                grid.free(pos_below)
        # print(grid.get_resistance(pos, start), pos, len(grid._unused_index))
        return grid.get_resistance(pos, start), pos
    r = simulation(start, grid, traj, defect_pos, neighbor_con)
plt.plot(list(map(lambda x: x[0], r)))


