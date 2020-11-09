from ResistorNetwork import ResistorNetwork
import numpy as np
from matplotlib import pyplot as plt


n_resistor_x = 40
n_resistor_y = 40
n_resistor_z = 10
nx = n_resistor_x + 1
ny = n_resistor_y + 1
nz = n_resistor_z + 1


class CubeTrajectoryWithJump:
    def __init__(self, start_pos=(0, 0, 0), nx=nx,
                  ny=ny, nz=nz, dir_long=(1, 0, 0)):
        self.n = (nx, ny, nz)
        self.area = nx * ny
        self.vol = nx * ny * nz
        self.pos = np.array(start_pos)
        self.dir_cur = np.array(dir_long)
        self.dir_long = np.array(dir_long)
        self.dir_short = np.array([1, 1, 0]) - self.dir_long
        self.cnt = 1
        self.jump = False

    def step(self):
        self.jump = False
        prev_pos = self.pos.copy()
        if self.cnt == self.vol:
            return self.pos, self.pos
        if self.cnt % self.area == 0:
            self.jump = True
            self.dir_short = np.abs(self.dir_long)
            self.dir_long[:2] = np.ones(2) - self.dir_short[:2]
            self.dir_cur = self.dir_long
            self.pos = np.array([0, 0, self.pos[2] + 1]) - self.dir_long
        elif self.cnt % np.array(self.n)[np.abs(self.dir_long).astype('bool')][0] == 0:
            self.dir_cur = self.dir_short
            self.dir_long *= -1
        elif self.cnt % np.array(self.n)[np.abs(self.dir_long).astype('bool')][0] == 1:
            self.dir_cur = self.dir_long
        self.cnt += 1
        self.pos += self.dir_cur
        return tuple(self.pos), tuple(prev_pos)


def simulation(start, grid, trajectory, defect_pos, neighbor_con):
    r = []
    for i in range(trajectory.vol - 1):
        print(i)
        r.append(neighbor_con(start, grid, trajectory, defect_pos))
    return r

if __name__ == "__main__":
    r_u_x = 1  # resistance along the printed line, x axis
    r_u_y = 1  # resistance along the printed line, y axis
    r_o_x = 2  # within layer overlapping resistance, x axis
    r_o_y = 2  # within layer overlapping resistance, y axis
    r_z = 1  # between layer contacting resistance
    r_u_def_x = 2  # resistance along the printed line when defect, x axis
    r_u_def_y = 2  # resistance along the printed line when defect, y axis
    r_o_def_x = np.inf  # within layer overlapping resistance when defect, x axis
    r_o_def_y = np.inf  # within layer overlapping resistance when defect, y axis
    r_z_def = 2  # between layer contacting resistance when defect
    start = (0, 0, 0)
    grid = ResistorNetwork(nx * ny, start)
    traj = CubeTrajectoryWithJump(start)

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
            r1 = [r_u_def_x, r_u_def_y, r_z_def]
            r2 = [r_o_def_x, r_o_def_y]
            r3 = r_z_def
        else:
            r1 = [r_u_x, r_u_y, r_z]
            r2 = [r_o_x, r_o_y]
            r3 = r_z
        d = np.argmax(np.abs(np.array(pos) - np.array(prev_pos)))
        if not traj.jump:
            grid.connect(pos, prev_pos, r1[d])
        pos_overlap = tuple(np.array(pos) - trajectory.dir_short)
        d1 = np.argmax(np.abs(trajectory.dir_short))
        if within_boundary(pos_overlap) and pos_overlap != prev_pos:
            grid.connect(pos, pos_overlap, r2[d1])
        pos_below = tuple(np.array(pos) - np.array([0, 0, 1]))
        if within_boundary(pos_below):
            if pos_below != prev_pos:
                grid.connect(pos, pos_below, r3)
            if pos_below != start:
                grid.free(pos_below)
        # print(grid.get_resistance(pos, start), pos, len(grid._unused_index))
        return grid.get_resistance(pos, start), pos
    r = simulation(start, grid, traj, defect_pos, neighbor_con)
    spatial_r = np.zeros((nx, ny, nz))
    for resistance, (i, j, k) in r:
        spatial_r[i, j, k] = resistance
    for i in range(nz):
        plt.figure()
        plt.imshow(spatial_r[..., i])
        plt.show()