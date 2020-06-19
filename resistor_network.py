import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


N = 24  # grid resolution
t = N  # current time
target = []
peak_r = [74, 97, 118, 140, 161, 183, 205, 226, 248, 269]
r = pd.read_table("90-100-1.txt", sep='\t').iloc[:, 1].values
r -= r[peak_r[0]]
peaks = [24, 49, 74, 99, 124, 149, 174, 199, 224, 249]

for i in range(len(peaks)-1):
    sig_start = peak_r[i]
    sig_end = peak_r[i+1]
    start = peaks[i]
    end = peaks[i+1]
    target.extend(list(np.interp(np.arange(end - start)*(sig_end - sig_start)/(end - start)+sig_start,
                                 np.arange(sig_start, sig_end), r[sig_start: sig_end])))

def pos_at(t):
    period = 2*(N+1)
    p = t % period
    return p * (p <= N) + (period - 1 - p) * (p > N)

hr = 5.5e-4
vr1 = 5.5e-4
vr2 = 1.2e-4
omega = np.zeros((N+2, N+2))  # pairwise resistance
omega[1:-1, 1:-1] = np.abs(np.arange(N).reshape(-1, 1) - np.arange(N).reshape(1, -1)) * hr  # initialize as a series of resistors
omega[0, :] = omega[1, :]  # make extra copy of the first and last positions
omega[:, 0] = omega[:, 1]
omega[-1, :] = omega[-2, :]
omega[:, -1] = omega[:, -2]
l = list(map(lambda x: x*hr, range(N)))
for i in range(t+1, (N+1)**2-1):
    pt = pos_at(i)
    ptm1 = pos_at(i-1)
    if pt != ptm1:
        omega[-1, :] = omega[ptm1 + 1, :] + hr
        omega[-1, -1] = 0
        omega[:, -1] = omega[-1, :]
        omega -= ((omega[:, -1] - omega[:, pt+1]).reshape(-1, 1)
                  - (omega[-1, :] - omega[pt+1, :]).reshape(1, -1))**2 / (4*vr2 + 4*omega[pt+1, -1])
    else:
        omega[-1, :] = omega[ptm1 + 1, :] + vr1
        omega[-1, -1] = 0
        omega[:, -1] = omega[-1, :]
    omega[pt+1, :] = omega[-1, :]
    omega[:, pt+1] = omega[:, -1]
    l.append(omega[0, pt+1])
# plt.plot(l)
maps = {}
peaks = [24, 49, 74, 99, 124, 149, 174, 199, 224, 249]
r = pd.read_table("90-100-1.txt", sep='\t').iloc[:, 1].values
peak = [74, 97, 118, 140, 161, 183, 205, 226, 248, 269]
for i in range(len(peaks)-1):
    sig_start = peak[i]
    sig_end = peak[i+1]
    start = peaks[i]
    end = peaks[i+1]
    maps.update(zip(np.arange(start, end),
                    np.interp(np.arange(end - start)*(sig_end - sig_start)/(end - start)+sig_start,
                              np.arange(sig_start, sig_end), r[sig_start: sig_end])))
plot(list(map(lambda x: x - l[24], l[peaks[0]: peaks[-1]])))
plot(target)
np.corrcoef(np.array(list(map(lambda x: x - l[24], l[peaks[0]: peaks[-1]]))), target)[0, 1]