import torch
from matplotlib import pyplot as plt
import pandas as pd
from torch.optim import Adam as Optim


r = pd.read_table("90-100-1.txt", sep='\t').iloc[:, 1].values
# plt.plot(r)
# np.where(((r[1:-1]-r[:-2]) > 0)&((r[1:-1]-r[2:])> 0))[0] + 1 peak [74, 97, 118, ...]
# np.where(((r[1:-1]-r[:-2]) < 0)&((r[1:-1]-r[2:])< 0))[0] + 1 trough [95, 98, 138, ...]
r -= r[74]
N = 24  # grid resolution
t = N-1  # current time


def pos_at(t):
    period = 2*(N+1)
    p = t % period
    return p * (p <= N) + (period - 1 - p) * (p > N)

hr = torch.tensor(1., requires_grad=True)  # horizontal_resistance_per_step
vr1 = torch.tensor(1., requires_grad=True)  # vertical_resistance_per_step
vr2 = torch.tensor(2., requires_grad=True)
lr = 1e-3
optim = Optim([hr, vr1, vr2], lr)

for _ in range(10000):
    optim.zero_grad()
    omega = torch.zeros((N+2, N+2))  # pairwise resistance
    omega[1:-1, 1:-1] = torch.abs(torch.arange(N).reshape(-1, 1) - torch.arange(N).reshape(1, -1)).float() * hr  # initialize as a series of resistors
    omega[0, :] = omega[1, :]  # make extra copy of the first and last positions
    omega[:, 0] = omega[:, 1]
    omega[-1, :] = omega[-2, :]
    omega[:, -1] = omega[:, -2]
    l = list(map(lambda x: x*hr, range(N)))
    omega_list = [omega]
    for i in range(t+1, 4*N+4):
        pt = pos_at(i)
        ptm1 = pos_at(i-1)
        omega_list.append(omega_list[-1].clone())
        if pt != ptm1:
            omega_list[-1][-1, :] = omega_list[-2][ptm1 + 1, :] + hr
            omega_list[-1][-1, -1] = 0
            omega_list[-1][:, -1] = omega_list[-1][-1, :]
            omega_list.append(omega_list[-1].clone())
            omega_list[-1] = omega_list[-2] - ((omega_list[-2][:, -1] - omega_list[-2][:, pt+1]).reshape(-1, 1)
                      - (omega_list[-2][-1, :] - omega_list[-2][pt+1, :]).reshape(1, -1))**2 / (4*vr2 + 4*omega_list[-2][pt+1, -1])
        else:
            omega_list[-1][-1, :] = omega_list[-2][ptm1 + 1, :] + vr1
            omega_list[-1][-1, -1] = 0
            omega_list[-1][:, -1] = omega_list[-1][-1, :]
        omega_list.append(omega_list[-1].clone())
        omega_list[-1][pt+1, :] = omega_list[-2][-1, :]
        omega_list[-1][:, pt+1] = omega_list[-1][:, -1]
        l.append(omega_list[-1][0, pt+1].clone())
    peak_1 = l[N+1]
    trough_1 = l[2*N+1] - peak_1
    peak_2 = l[2*N+2] - peak_1
    trough_2 = l[2*N+3] - peak_1
    peak_3 = l[3*N+3] - peak_1
    trough_3 = l[4*N+3] - peak_1
    L = torch.abs(trough_1 - r[95]) + torch.abs(trough_2 - r[98]) + torch.abs(trough_3 - r[138]) + torch.abs(peak_2 - r[97]) + torch.abs(peak_3 - r[118])
    print(L)
    L.backward()
    optim.step()