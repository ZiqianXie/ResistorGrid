import pandas as pd
from ResistorNetwork import ResistorNetwork
import numpy as np
from scipy import interpolate


df = pd.read_excel("Data.xlsx").set_index("time")
peak_annotation = [[154, 252], [171, 271], [147, 248]]
N_resistor_horizontal = 50
N_resistor_vertical = 24
data_len = (N_resistor_horizontal + 1) * (N_resistor_vertical + 1) - 1
data = [df[[c]].dropna() for c in df.columns if "Nowall" in c]
for i, datum in enumerate(data):
    f = interpolate.interp1d(datum.index, datum.values.reshape(-1))
    peak0, peak1 = peak_annotation[i]
    time_unit = (datum.index[peak1] - datum.index[peak0])/(2*N_resistor_horizontal+2)
    start_time = datum.index[peak0] - time_unit * N_resistor_horizontal
    r_horizontal = (f(datum.index[peak0] - time_unit) - f(datum.index[peak0] -
                                                          time_unit * N_resistor_horizontal))/N_resistor_horizontal
    r_vertical_1 = datum.iloc[peak0].values - f(datum.index[peak0] - time_unit)
    gt = f(start_time + np.arange(data_len) * time_unit) - f(start_time)
    best_loss = np.inf
    best_r_vertical_2 = None  # 1.9e-5,
    for j in range(1, 500):
        r_vertical_2 = 1e-6 * j
        grid = ResistorNetwork(N_resistor_horizontal + 1, (0, 0))
        r = []
        direction = (1, 0)
        pos = (0, 0)
        for i in range(data_len):
            new_pos = (pos[0] + direction[0], pos[1] + direction[1])
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
            r.append(grid.get_resistance((0, 0), pos))
        r = np.array(r)
        loss = np.abs(r - gt).sum()
        if loss < best_loss:
            best_r_vertical_2 = r_vertical_2
            best_loss = loss
            best_r = r
    figure()
    plot(gt)
    plot(best_r)
