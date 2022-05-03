from typing import List

import numpy as np
import torch
import torch.nn as nn
from scipy import signal

np.random.seed(0)


class FurnaceSimulator:
    def __init__(self,
                 num_grid=100,
                 delta_t=0.000025,
                 diff_constant=1.0,
                 domain=(0.0, 1.0),
                 timestep=0.01,
                 init_cond=None,
                 boundary_cond=None):
        # Asserting numerical diffusivity condition

        self.num_x = num_grid + 1
        self.num_y = num_grid + 1
        self.delta_t = delta_t
        self.diff_constant = diff_constant
        self.domain = domain
        self.timestep = timestep
        self.init_cond = init_cond
        self.boundary_cond = boundary_cond

        self.state_dim = self.num_x * self.num_y
        self.action_dim = self.num_x * self.num_y
        self.state_shape = (self.num_x, self.num_y)
        self.action_shape = (self.num_x, self.num_y)
        self.x_pos = np.linspace(domain[0], domain[1], self.num_x)
        self.y_pos = np.linspace(domain[0], domain[1], self.num_y)
        self.num_t = int(timestep / delta_t)
        self.u = np.zeros(shape=(self.num_x, self.num_y))

        # masking for known values
        # known values such as Initial condition
        # and Boundary conditions are masked as 0
        mask = np.ones(shape=(self.num_x, self.num_y))

        # boundary values
        mask[0, :] = 0
        mask[-1, :] = 0
        mask[:, 0] = 0
        mask[:, -1] = 0
        self.mask = mask
        # source terms

        # imposing initial conditions
        self.xx, self.yy = np.meshgrid(self.x_pos, self.y_pos)
        self.reset()

        # imposing boundary conditions
        if boundary_cond is None:
            pass  # zero Dirichlet boundary conditions
        else:
            raise NotImplementedError

        self.conv_filter = [[0.0, 1.0, 0.0],
                            [1.0, -4.0, 1.0],
                            [0.0, 1.0, 0.0]]

        self.conv_filter = np.array(self.conv_filter)

        self.t = 0
        x_length = domain[1] - domain[0]
        self.step_param = diff_constant * delta_t * (num_grid ** 2) / (x_length ** 2)

    def reset(self):
        self.u = self.init_cond(self.xx, self.yy)
        return

    def step(self, a):
        a = np.reshape(a, self.action_shape)
        for t in range(self.num_t):
            conv = signal.correlate2d(self.u, self.conv_filter, mode='same')
            self.u = self.u + self.step_param * conv + self.delta_t * a
            self.u = self.mask * self.u
        self.t += self.delta_t
        return self.u

    def set_state(self, u):
        assert len(u) == self.state_dim
        self.u = np.reshape(u, self.state_shape)
        return


class FurnaceEnvironment:
    def __init__(self,
                 num_grid: int,
                 diff_constant: float,
                 delta_t: float,
                 domain: List,
                 timestep: float,
                 state_pos: np.array,  # [I x 2] numpy integer array
                 action_pos: np.array):  # [J x 2] numpy integer array
        # Furnace Setting
        self.num_grid = num_grid  # 100 x 100 mesh
        self.diff_constant = diff_constant
        self.delta_t = delta_t  # dt <= dx * dx / 4k
        self.domain = domain
        self.timestep = timestep  # 400 iterations per each step

        def zero_cond(xx, yy):
            return np.zeros_like(xx)

        self.sim = FurnaceSimulator(num_grid=self.num_grid, delta_t=self.delta_t, diff_constant=self.diff_constant, domain=self.domain, timestep=self.timestep,
                                    init_cond=zero_cond)
        self.state_trajectory = []
        self.action_trajectory = []
        self.state_dim = state_pos.shape[0]
        self.action_dim = action_pos.shape[0]
        self.state_pos = state_pos
        self.action_pos = action_pos
        self.reset()

    def reset(self):
        self.sim.reset()
        self.state_trajectory = []
        self.action_trajectory = []
        self.state_trajectory.append(self.sim.u[self.state_pos[:, 0], self.state_pos[:, 1]])

    def step(self, a):
        """
        :param a: numpy array, [self.action_dim] shape
        :return:
        """
        action = np.zeros((self.num_grid + 1, self.num_grid + 1))
        # NEW
        for i, pos in enumerate(self.action_pos):
            action[pos[0], pos[1]] += a[i]
        # OLD
        # action[self.action_pos[:, 0], self.action_pos[:, 1]] = a
        state = self.sim.step(action)
        self.state_trajectory.append(state[self.state_pos[:, 0], self.state_pos[:, 1]])
        self.action_trajectory.append(a)
        return state[self.state_pos[:, 0], self.state_pos[:, 1]]

    def get_trajectory(self):
        if len(self.state_trajecotry) > 0:
            self.state_trajectory = np.stack(self.state_trajectory, axis=0)  # [T+1 x self.state_dim] numpy array
        if len(self.action_trajectory) > 0:
            self.action_trajectory = np.stack(self.action_trajectory, axis=0)  # [T x self.action_dim] numpy array
        return self.state_trajectory, self.action_trajectory


class TorchFurnaceSimulator(nn.Module):
    def __init__(self,
                 num_grid=100,
                 delta_t=0.000025,
                 diff_constant=1.0,
                 domain=(0.0, 1.0),
                 timestep=0.01,
                 init_cond=None,
                 boundary_cond=None,
                 device='cpu'):
        # Asserting numerical diffusivity condition
        super(TorchFurnaceSimulator, self).__init__()
        self.num_x = num_grid + 1
        self.num_y = num_grid + 1
        self.delta_t = delta_t
        self.diff_constant = diff_constant
        self.domain = domain
        self.timestep = timestep
        self.init_cond = init_cond
        self.boundary_cond = boundary_cond
        self.device = device

        self.state_dim = self.num_x * self.num_y
        self.action_dim = self.num_x * self.num_y
        self.state_shape = (self.num_x, self.num_y)
        self.action_shape = (self.num_x, self.num_y)
        self.x_pos = torch.linspace(domain[0], domain[1], self.num_x).to(self.device)
        self.y_pos = torch.linspace(domain[0], domain[1], self.num_y).to(self.device)
        self.num_t = int(timestep / delta_t)
        self.u = torch.zeros((self.num_x, self.num_y)).to(self.device)

        # masking for known values
        # known values such as Initial condition
        # and Boundary conditions are masked as 0
        mask = torch.ones((self.num_x, self.num_y)).to(self.device)

        # boundary values
        mask[0, :] = 0.0
        mask[-1, :] = 0.0
        mask[:, 0] = 0.0
        mask[:, -1] = 0.0
        self.mask = mask
        # source terms

        # imposing initial conditions
        self.xx, self.yy = torch.meshgrid(self.x_pos, self.y_pos)
        self.reset()

        # imposing boundary conditions
        if boundary_cond is None:
            pass  # zero Dirichlet boundary conditions
        else:
            raise NotImplementedError

        self.conv_filter_value = [[0.0, 1.0, 0.0],
                                  [1.0, -4.0, 1.0],
                                  [0.0, 1.0, 0.0]]
        self.conv_filter = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False).to(self.device)
        self.conv_filter.weight.data = torch.tensor(self.conv_filter_value).to(self.device).unsqueeze(
            dim=0).unsqueeze(dim=0)

        self.t = 0
        x_length = domain[1] - domain[0]
        self.step_param = self.diff_constant * delta_t * (num_grid ** 2) / (x_length ** 2)

    def reset(self):
        self.u = self.init_cond(self.xx, self.yy).to(self.device).unsqueeze(dim=0).unsqueeze(dim=0)
        return

    def step(self, a):
        a = torch.reshape(a, self.action_shape).to(self.device).unsqueeze(dim=0).unsqueeze(dim=0)
        for t in range(self.num_t):

            self.u = self.u + self.step_param * self.conv_filter(self.u) + self.delta_t * a
            # conv = nn.functional.conv2d(input=self.u, weight=self.conv_filter, padding=1)
            # self.u = self.u + self.step_param * conv + self.dt * a
            self.u = self.mask * self.u
        self.t += self.delta_t
        return self.u[0, 0, :, :]


class TorchFurnaceEnvironment(nn.Module):
    def __init__(self,
                 num_grid,
                 diff_constant,
                 delta_t,
                 domain,
                 timestep,
                 state_pos: torch.tensor,  # [I x 2] torch tensor array
                 action_pos: torch.tensor,  # [J x 2] torch tensor array
                 device='cpu'):
        super(TorchFurnaceEnvironment, self).__init__()
        # Furnace Setting
        self.num_grid = num_grid  # 100 x 100 mesh
        self.diff_constant = diff_constant
        self.delta_t = delta_t  # dt <= dx * dx / 4k
        self.domain = domain
        self.timestep = timestep  # 400 iterations per each step
        self.device = device

        def zero_cond(xx, yy):
            return torch.zeros_like(xx)

        self.sim = TorchFurnaceSimulator(num_grid=self.num_grid, delta_t=self.delta_t, diff_constant=self.diff_constant, domain=self.domain, timestep=self.timestep,
                                         init_cond=zero_cond, device=device)
        self.state_trajectory = []
        self.action_trajectory = []
        self.state_dim = state_pos.shape[0]
        self.action_dim = action_pos.shape[0]
        self.state_pos = state_pos
        self.action_pos = action_pos
        self.reset()

    def reset(self):
        self.sim.reset()
        self.state_trajectory = []
        self.action_trajectory = []
        self.state_trajectory.append(self.sim.u[0, 0, self.state_pos[:, 0], self.state_pos[:, 1]])

    def step(self, a):
        """
        :param a: torch.tensor, [self.action_dim] shape
        :return:
        """
        action = torch.zeros((self.num_grid + 1, self.num_grid + 1)).to(self.device)
        for i, pos in enumerate(self.action_pos):
            action[pos[0], pos[1]] += a[i]
        state = self.sim.step(action)
        self.state_trajectory.append(state[self.state_pos[:, 0], self.state_pos[:, 1]])
        self.action_trajectory.append(a)
        return state[self.state_pos[:, 0], self.state_pos[:, 1]]

    def get_trajectory(self):
        self.state_trajectory = torch.stack(self.state_trajectory, dim=0)  # [T+1 x self.state_dim] numpy array
        self.action_trajectory = torch.stack(self.action_trajectory, dim=0)  # [T x self.action_dim] numpy array
        return self.state_trajectory, self.action_trajectory
