from model.utils import *

a_auler = [[70, -45, 30]]
pitch = a_auler[0][0]
yaw = a_auler[0][1]
roll = a_auler[0][2]

pitch = pitch /180 +0.5
yaw = yaw /360 + 0.5
roll = roll /180 +0.5

a = [[pitch, yaw, roll]]
a_tensor = torch.tensor(a, dtype=torch.float32)

b = compute_rotation_matrix_from_euler(a_tensor, normalized=True)
c = compute_euler_angles_from_rotation_matrices(b)
import torch
deg = torch.rad2deg(c)

print(f'a_auler = {a_auler}')
print(f'a = {a}')
print(f'b = {b}')
print(f'c = {c}')
print(f'deg = {deg}')


