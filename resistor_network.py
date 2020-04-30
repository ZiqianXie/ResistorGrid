import numpy as np
from matplotlib import pyplot as plt


N = 24  # grid resolution
t = N  # current time


def pos_at(t):
    period = 2*(N+1)
    p = t % period
    return p * (p <= N) + (period - 1 - p) * (p > N)

hr = 1
vr1 = 1
vr2 = 1
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
plt.plot(l)
