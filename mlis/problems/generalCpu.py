# You need to learn a function with n inputs.
# For given number of inputs, we will generate random function.
# Your task is to learn it
import time
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
sys.path.append('./../utils')
import solutionmanager as sm
from gridsearch import GridSearch
import pandas as pd
import numpy as np


LOSSES = {
        'MSELoss': nn.MSELoss(reduction='sum'),
        'BCELoss': nn.BCELoss(),
        }

ACTIVATIONS_GRID = {
        'sigmoid': nn.Sigmoid(),
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'hardshrink': nn.Hardshrink(),
        'htang1': nn.Hardtanh(-1, 1),
    }
class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size, params):
        super(SolutionModel, self).__init__()
        self.params = params
        self.input_size = input_size
        
        if params.grid_search.enabled:
            torch.manual_seed(params.random)

        sizes = (input_size, *params.hidden_sizes, output_size)
        
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for idx in range(len(sizes) - 1):
            linear = nn.Linear(sizes[idx], sizes[idx + 1])
            if params.batch_norm:
                bn = nn.BatchNorm1d(sizes[idx + 1], track_running_stats=False)
            self.layers.append(linear)
            self.batch_norms.append(bn)
        
    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx != len(self.layers) - 1 and self.params.batch_norm:
                x = self.batch_norms[idx](x)
            x = ACTIVATIONS_GRID[self.params.activations[idx]](x)
        return x

    def calc_loss(self, output, target):
        loss = LOSSES[self.params.loss](output, target)
        return loss

    def calc_predict(self, output):
        predict = output.round()
        return predict

class Solution():
    def __init__(self):
        self.sols = {}
        self.solsSum = {}

        self.n_hidden = 5

        self.hidden_sizes = [47] * self.n_hidden
        self.activations = ['relu'] + (['relu'] * (self.n_hidden - 1)) + ['sigmoid']
        self.loss = 'MSELoss'
        self.lr = 0.00774
        self.momentum = 0.91079
        
        self.random = 0
        self.batch_norm = True
        
        self.hidden_sizes_grid = [
            [i] * self.n_hidden
            for i in [47]
#            for i in np.linspace(3, 66, 40, dtype=int)

        ]

        self.activations_grid = [
           [i] * self.n_hidden + [j]
           for i in ['relu']
           for j in ['sigmoid']
        ]
        
        self.loss_grid = list(LOSSES)
#        self.activation_grid = list(ACTIVATIONS_GRID)
        self.lr_grid = np.round(np.geomspace(0.001, 10, 10), 5)
#        self.hidden_size_grid = np.linspace(17, 35, 12, dtype=int)
#        self.lr_grid = [0.001]
#        self.lr_grid = np.linspace(0.001, 0.2, 11)
#        1 - 10**(- np.random.rand(5) - 1)
#        self.momentum_grid = np.round(np.geomspace(0.9, 0.99, 9), 5)
        self.random_grid = range(3)
        self.grid_search = GridSearch(self).set_enabled(False)

    def create_model(self, input_size, output_size):
        return SolutionModel(input_size, output_size, self)

    def get_key(self):
        return "{}_{}_{}_{}_{}".format(self.lr, self.momentum, self.hidden_sizes, self.activations, self.loss);

    # Return number of steps used
    def train_model(self, model, train_data, train_target, context):
        # get off noise
        key = self.get_key()
        if not key in self.sols:
            self.sols[key] = 0
            self.solsSum[key] = 0
        if self.sols[key] == -1:
            return
        
        step = 0
        # Put model in train mode
        model.train()
#        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        
        while True:
            time_left = context.get_timer().get_time_left()
            if time_left < 0.1 or (self.grid_search.enabled and step > 15):
                self.sols[key] = -1
                break
            data = train_data
            target = train_target
            # model.parameters()...gradient set to zero
            optimizer.zero_grad()
            # evaluate model => model.forward(data)
            output = model(data)
            # if x < 0.5 predict 0 else predict 1
            predict = output.round()
            # Number of correct predictions
            correct = predict.eq(target.view_as(predict)).long().sum().item()
            # Total number of needed predictions
            total = target.view(-1).size(0)
            if correct == total:
                self.sols[key] += 1
                self.solsSum[key] += step
                print("{:.4f} {}".format(step, key))
                if self.sols[key] == len(self.random_grid):
                    print('---confirmed {:.4f} ---'.format(float(self.solsSum[key])/self.sols[key]))
                break
            # calculate loss
            loss = model.calc_loss(output, target)
            self.grid_search.log_step_value('loss', loss.item(), step)
            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            loss.backward()
            # print progress of the learning
#            self.print_stats(step, loss, correct, total)
            # update model: model.parameters() -= lr * gradient
            optimizer.step()
            step += 1
        return step

    def print_stats(self, step, loss, correct, total):
        if step % 1000 == 0:
            print("Step = {} Prediction = {}/{} Error = {}".format(step, correct, total, loss.item()))

###
###
### Don't change code after this line
###
###
class Limits:
    def __init__(self):
        self.time_limit = 2.0
        self.size_limit = 10000
        self.test_limit = 1.0

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10

    def create_data(self, input_size, seed):
        random.seed(seed)
        data_size = 1 << input_size
        data = torch.FloatTensor(data_size, input_size)
        target = torch.FloatTensor(data_size)
        for i in range(data_size):
            for j in range(input_size):
                input_bit = (i>>j)&1
                data[i,j] = float(input_bit)
            target[i] = float(random.randint(0, 1))
        return (data, target.view(-1, 1))

    def create_case_data(self, case):
        input_size = min(3+case, 7)
        data, target = self.create_data(input_size, case)
#        x_norm = np.linalg.norm(data)
#        data = data / x_norm
#        print(data[:3])
        return sm.CaseData(case, Limits(), (data, target), (data, target)).set_description("{} inputs".format(input_size))


class Config:
    def __init__(self):
        self.max_samples = 1000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()

# If you want to run specific case, put number here
tic = time.time()
sm.SolutionManager(Config()).run(case_number=-1)
toc = time.time()
print('proceed at %f s' %(toc - tic))
