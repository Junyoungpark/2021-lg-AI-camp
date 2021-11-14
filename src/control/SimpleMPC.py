import numpy as np
import torch
import torch.nn as nn


class Action(nn.Module):
    """
    A nn.Module wrapper for action
    """

    def __init__(self,
                 H,
                 action_dim,
                 action_min,
                 action_max):
        super(Action, self).__init__()
        us = np.random.uniform(low=action_min, high=action_max, size=(1, H, action_dim))
        self.us = torch.nn.Parameter(torch.from_numpy(us).float())
        self.action_min = action_min
        self.action_max = action_max

    def forward(self):
        return self.us

    def clamp_action(self):
        self.us.data = self.us.data.clamp(min=self.action_min, max=self.action_max)


class MPC(nn.Module):
    """
    Minimal MPC implementation utilizing arbitrary torch.nn.Module as the dynamic model
    """

    def __init__(self,
                 model: nn.Module,
                 state_dim: int,
                 action_dim: int,
                 H: int,  # receding horizon
                 action_min: float = -1.0,
                 action_max: float = 1.0,
                 gamma: float = 1.0):
        super(MPC, self).__init__()

        self.model = model
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.H = H
        self.action_min = action_min
        self.action_max = action_max
        self.gamma = gamma

    def solve(self, x0, target, max_iter: int, tol=1e-5):
        crit = torch.nn.MSELoss()

        us = Action(self.H, self.action_dim, self.action_min, self.action_max).to(target.device)
        opt = torch.optim.Adam(us.parameters(), lr=1e-1)  # Large LR (step size) start heuristics
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)

        info = dict()
        solve = False
        for i in range(max_iter):
            opt.zero_grad()
            prediction = self.roll_out(x0, us())
            loss = crit(prediction, target)
            loss.backward()
            opt.step()
            scheduler.step(loss)

            # Projection heuristics
            us.clamp_action()
            if loss <= tol:
                solve = True
                break
        info['loss'] = loss.item()
        info['solve'] = solve
        info['iters'] = i
        return us.us.data, info

    def roll_out(self, x0, us):
        """
        :param x0: initial state. expected to get 'torch.tensor' with dimension of [batch x state_dim]
        :param us: action sequences [batch x time stamps x  action_dim]
        """
        xs = []
        x = x0

        for u in us.unbind(dim=1):  # iterating over time stamps
            x = self.model(x, u)
            xs.append(x)
        return torch.stack(xs, dim=1)  # [batch x time stamps x state_dim]
