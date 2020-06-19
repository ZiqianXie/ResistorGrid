import torch
import pandas as pd
from torch.optim import Adam as Optim
import numpy as np


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

N = 24  # grid resolution
t = N  # current time


def pos_at(t):
    period = 2*(N+1)
    p = t % period
    return p * (p <= N) + (period - 1 - p) * (p > N)

hr = torch.tensor(5e-4, requires_grad=True)  # horizontal_resistance_per_step
vr1 = torch.tensor(5e-4, requires_grad=True)  # vertical_resistance_per_step
vr2 = torch.tensor(5e-4, requires_grad=True)
b = torch.zeros(peaks[-1] - peaks[0] + 1, requires_grad=True)
lr = 1e-6
optim = Optim([hr, vr1, vr2, b], lr)

best_corr = -np.inf
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
    for i in range(t+1, t+1+peaks[-1]):
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
    L = 0
    for i in range(peaks[0], peaks[-1]):
        j = i-peaks[0]
        L += torch.abs(l[i] - l[peaks[0]] - target[j] - b[j]) + b[j]**2 + (b[j+1] - b[j])**2
    corr = np.corrcoef(np.array(list(map(lambda x: x.detach().numpy() - l[24].detach().numpy(),
                                         l[peaks[0]: peaks[-1]]))), target)[0, 1]
    if corr > best_corr:
        best_corr = corr
        best_params = [hr.clone(), vr1.clone(), vr2.clone(), b.clone()]
        print("best_corr:", best_corr)
    print(L)
    L.backward()
    optim.step()

plot(list(map(lambda x: x - l[24], l[peaks[0]: peaks[-1]])))
plot(target-b[:-1].detach().numpy())