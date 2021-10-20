## MAE-598 Project 1
## Sebastian Garcia Peralta
#1213082648


# overhead

import logging
import math
import random
import numpy as np
import time
import torch as t
import torch.nn as nn
from torch import optim
from torch.nn import utils
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# environment parameters

FRAME_TIME = 0.1  # time interval
GRAVITY_ACCEL = 0.12  # gravity constant
BOOST_ACCEL = 0.18  # thrust constant
DRAG_CON = 0.05 #acts as the whole force

# # the following parameters are not being used in the sample code
#PLATFORM_WIDTH = 0.25  # landing platform width
#PLATFORM_HEIGHT = 0.06  # landing platform height
# ROTATION_ACCEL = 20  # rotation constant

# define system dynamics
# Notes:
# 0. You only need to modify the "forward" function
# 1. All variables in "forward" need to be PyTorch tensors.
# 2. All math operations in "forward" has to be differentiable, e.g., default PyTorch functions.
# 3. Do not use inplace operations, e.g., x += 1. Please see the following section for an example that does not work.

class Dynamics(nn.Module):

    def __init__(self):
        super(Dynamics, self).__init__()

    @staticmethod
    def forward(state, action):
        """
        action: thrust or no thrust
        state[0] = x
        state[1] = y
        state[2] = x_dot
        state[3] = y_dot
        """

# Initialize a matrix for new delta states
        matrix_deltastate = []
        for i in range(len(state)):

            #Consider gravity
            dstate_grav = t.tensor([0.,0.,0.,-GRAVITY_ACCEL*FRAME_TIME])

            #Add Drag to the problem
            dstate_drag = t.tensor([0.,0.,0.,-DRAG_CON*FRAME_TIME])

            #Consider thrust
            dstate_trust = BOOST_ACCEL * FRAME_TIME * t.tensor([0., 0., 1., 1.]) * t.cat((t.zeros(2), action[i])) #need to implement action variable into problem in a better way.


            dstate = dstate_trust + dstate_drag + dstate_grav
            matrix_deltastate.append(dstate)
        dstate_mat_t = t.stack(matrix_deltastate)

        #Velocity
        state = state+dstate_mat_t

        #Position
        step_mat = t.tensor([[1., 0., 0., 0.],
                            [0., 1., 0., 0.],
                            [FRAME_TIME, 0., 1., 0.],
                            [0., FRAME_TIME, 0., 1]])
        state = t.matmul(state,step_mat)

        return state

# a deterministic controller
# Note:
# 0. You only need to change the network architecture in "__init__"
# 1. nn.Sigmoid outputs values from 0 to 1, nn.Tanh from -1 to 1
# 2. You have all the freedom to make the network wider (by increasing "dim_hidden") or deeper (by adding more lines to nn.Sequential)
# 3. Always start with something simple


class Controller(nn.Module):

    def __init__(self, dim_input, dim_hidden, dim_output):
        """
        dim_input: # of system states
        dim_output: # of actions
        dim_hidden: up to you
        """
        super(Controller, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim_output),
            # You can add more layers here
            nn.Sigmoid()
        )

    def forward(self, state):
        action = self.network(state)
        return action

# the simulator that rolls out x(1), x(2), ..., x(T)
# Note:
# 0. Need to change "initialize_state" to optimize the controller over a distribution of initial states
# 1. self.action_trajectory and self.state_trajectory stores the action and state trajectories along time

class Simulation(nn.Module):

    def __init__(self, controller, dynamics, T):
        super(Simulation, self).__init__()
        self.state = self.initialize_state()
        self.controller = controller
        self.dynamics = dynamics
        self.T = T
        self.action_trajectory = []
        self.state_trajectory = []

    def forward(self, state):
        self.action_trajectory = []
        self.state_trajectory = []
        for _ in range(T):
            action = self.controller.forward(state)
            state = self.dynamics.forward(state, action)
            self.action_trajectory.append(action)
            self.state_trajectory.append(state)
        return self.error(state)

    @staticmethod
    def initialize_state():
        state = [0.1, 1.,0.,0.]  # TODO: need batch of initial states
        return t.tensor(state, requires_grad=False).float()

    def error(self, state):
        return state[0] ** 2 + state[1] ** 2 +state[2] ** 2+ state[3] ** 2

    # @staticmethod
    # def initialize_state():
    #     num = 2
    #     randlimit = 0.25
    #     limitdev = randlimit/3
    #     state = [[1-random.gauss(0,limitdev), 0]]  # TODO: need batch of initial states
    #     for i in range(num-1):
    #         state.append([1-random.gauss(0,limitdev), 0])
    #     return t.tensor(state, requires_grad=False).float()
    #
    # def error(self, state):
    #     errorsum = 0
    #     num = 2
    #     for i in range(num):
    #         errorsum+=state[i,0]**2+state[i,1]**2
    #     return errorsum/len(state)


# set up the optimizer
# Note:
# 0. LBFGS is a good choice if you don't have a large batch size (i.e., a lot of initial states to consider simultaneously)
# 1. You can also try SGD and other momentum-based methods implemented in PyTorch
# 2. You will need to customize "visualize"
# 3. loss.backward is where the gradient is calculated (d_loss/d_variables)
# 4. self.optimizer.step(closure) is where gradient descent is done

class Optimize:
    def __init__(self, simulation):
        self.simulation = simulation
        self.parameters = simulation.controller.parameters()
        self.optimizer = optim.LBFGS(self.parameters, lr=0.01)

    def step(self):
        def closure():
            loss = self.simulation(self.simulation.state)
            self.optimizer.zero_grad()
            loss.backward()
            return loss
        self.optimizer.step(closure)
        return closure()

    def train(self, epochs):
        for epoch in range(epochs):
            loss = self.step()
            self.loss = loss
            print('[%d] loss: %.3f' % (epoch + 1, loss))
            self.visualize()

    # def visualize(self):
    #     data = np.array([self.simulation.state_trajectory[i].detach().numpy() for i in range(self.simulation.T)])
    #     x = data[:, 0]
    #     y = data[:, 1]
    #     #plt.plot(x, y)
    #     #plt.show()

# Now it's time to run the code!

T = 100  # number of time steps
dim_input = 4  # state space dimensions
dim_hidden = 6  # latent dimensions
dim_output = 2  # action space dimensions
d = Dynamics()  # define dynamics
c = Controller(dim_input, dim_hidden, dim_output)  # define controller
s = Simulation(c, d, T)  # define simulation
o = Optimize(s)  # define optimizer
o.train(40)  # solve the optimization problem





