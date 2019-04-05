# There are random function from 8 inputs and X random inputs added.
# We split data in 2 parts, on first part you will train and on second
# part we will test
import time
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import sys
sys.path.append('./../utils')
import solutionmanager as sm
from gridsearch import GridSearch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import re
from collections import OrderedDict

LOSSES = {
        'MSELoss': nn.MSELoss(reduction='mean'),
        'SELoss': nn.MSELoss(reduction='sum'),
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
        self.activations = ['relu'] + (['relu'] * (len(params.hidden_sizes) - 1)) + ['sigmoid']
        
        if params.grid_search.enabled:
            torch.manual_seed(params.random)

        sizes = (input_size, *params.hidden_sizes, output_size)
        
        sequence = []
        for x in range(1, len(sizes)):
            sequence.append(('linear%i' % x, nn.Linear(sizes[x - 1], sizes[x])))
            sequence.append(('bn%i' % x, nn.BatchNorm1d(sizes[x], track_running_stats=False)))
            sequence.append(('%s%i' % (self.activations[x - 1], x), ACTIVATIONS_GRID[self.activations[x - 1]]))
            if x == 1: 
                sequence.append(('dropout%i' % x, nn.Dropout(p=params.prob_out)))
        self.fc = nn.Sequential(OrderedDict(sequence))

    def forward(self, x):
        x = self.fc(x)
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
        self.plot_main_key = 'weight_decay'

        layers = 2

        self.hidden_sizes = [80] * layers
        
        self.loss = 'MSELoss'
        self.lr = .01
        self.momentum = 0.9
        self.batch_size = 256
        self.batch_norm = True
        self.lr_gamma = 1
        self.lr_step_size = [40]
        self.weight_decay = 1e-4
        self.prob_out = 1e-4
        
        self.random = 10
        
#        self.activations_grid = [
#           [i] * self.n_hidden + [j]
#           for i in ['relu']
#           for j in ['sigmoid']
#        ]
        
#        self.hidden_sizes_grid = [
#                [250] * layers,
#                [250, 250, 125, 125, 125, 125, 75],
#            ]
        
#        self.hidden_sizes_grid = []
#        for n in [250, 500]:#np.around(np.geomspace(10, 500, 5)).astype(int):
#            for l in self.layers_range:
#                self.hidden_sizes_grid.append([np.ceil(n / 2**r).astype(int) for r in range(l)])
                
#        self.layers_range = [2]
#        self.hidden_sizes_grid = [
#            [i] * n
##            for i in [80]
#            for n in self.layers_range
#            for i in np.linspace(60, 140, 5, dtype=int)
#        ]
#        
#        self.lr_grid = np.round(np.geomspace(0.001, 10, 5), 5)
#        self.lr_grid = np.round(0.5 + 10**(-0 - 1 * np.random.rand(10)), 5)
#        self.lr_grid = np.append(1, np.round(np.geomspace(0.05, 3., 10), 5)) #SGD
#        self.lr_grid = np.append(0.01, np.round(np.geomspace(0.005, 0.05, 5), 5)) #Adam

#        self.batch_size_grid = np.logspace(5, 9, base=2, num=5, dtype=int)
#        self.batch_norm_grid = [True, False]
        
#        self.loss_grid = list(LOSSES)
#        self.activation_grid = list(ACTIVATIONS_GRID)
        
#        self.momentum_grid = np.round(1 - 10**(-1 - np.random.rand(10)), 5)
#        self.momentum_grid = np.round(np.geomspace(0.999, 0.9, 10), 5)
#        self.momentum_grid = [0.5, 0.9, 0.95, 0.975, 0.99, 0.995]
        
#        self.lr_gamma_grid = [0.5]
#        self.lr_step_size_grid = [[150, 200, 250],
#                                  [1e+10]]
        
#        self.weight_decay_grid = np.append(0, np.geomspace(1e-6, 1e-2, 5))
#        self.prob_out_grid = np.append(0, np.geomspace(0.01, 0.1, 4))
        
#        self.random_grid = range(6,11)
        self.merge_random_param = False
        
        self.grid_search = GridSearch(self).set_enabled(False)

    def create_model(self, input_size, output_size):
        return SolutionModel(input_size, output_size, self)

    def get_key(self, merge_random=False):
        if merge_random:
            p = re.compile('random-.')
            return p.sub('', self.grid_search.choice_str)
        else:
            return self.grid_search.choice_str
    
    # Return number of steps used
    def train_model(self, model, train_data, train_target, context):
        # get off noise
        steps_limit = 1e+10
        break_on_correct = True
        random_size = len(self.random_grid) if hasattr(self, 'random_grid') else 1
        
        if self.grid_search.enabled:
            key = self.get_key(self.merge_random_param)
            plot_main_key = "{}-{}".format(self.plot_main_key, getattr(self, self.plot_main_key))
            if not key in self.sols:
                self.sols[key] = 0
                self.solsSum[key] = 0
                # plotting init
                if not plot_main_key in ACCS:
                    ACCS[plot_main_key] = {}
                ACCS[plot_main_key][key] = []
            if self.sols[key] == -1:
                return
        step = 0
        step_mini = 0
        # Put model in train mode
        model.train()
        
#        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, betas=(self.momentum, 0.999), weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_step_size, gamma=self.lr_gamma)
        
        train_dataset = Data.TensorDataset(train_data, train_target)
        train_loader = Data.DataLoader(dataset=train_dataset, batch_size=int(self.batch_size), shuffle=True)

        while True:
            
            for train_data_mini, train_target_mini in train_loader:
                scheduler.step()
                time_left = context.get_timer().get_time_left()
                if time_left < 0.1:
                    if self.grid_search.enabled:
                        self.sols[key] = -1
                    break
                data = train_data_mini
                target = train_target_mini
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
                accuracy = correct/total*100
                self.grid_search.log_step_value('ratio', accuracy, step)
                
                if self.grid_search.enabled:
                    ACCS[plot_main_key][key].append(accuracy)
                    
                # calculate loss
                loss = model.calc_loss(output, target)
                self.grid_search.log_step_value('loss', loss.item(), step)
                # calculate deriviative of model.forward() and put it in model.parameters()...gradient
                loss.backward()
                # print progress of the learning
#                self.print_stats(step_mini, loss, correct, total, time_left, optimizer)
                # update model: model.parameters() -= lr * gradient
                optimizer.step()
                step_mini += 1
            else:
                stat = sm.SolutionManager(Config()).calc_model_stats(model, train_data, train_target)
                if stat['correct'] == stat['total']:
                    if self.grid_search.enabled:
                        self.sols[key] += 1
                        self.solsSum[key] += step
                        print("{:.4f} {} {:.1f}".format(step, key, context.get_timer().get_execution_time()))
                        if self.sols[key] == random_size and random_size > 1:
                            print('---confirmed {:.4f} ---'.format(float(self.solsSum[key])/self.sols[key]))
                        break
                    elif break_on_correct:
                        break
                step += 1
                continue
            break
        return step

    def print_stats(self, step, loss, correct, total, time_left, optimizer):
        if step % 100 == 0:
#            print(get_lr(optimizer))
            print("Step = {} Prediction = {}/{} Ratio = {:.3f} Error = {:.4f} time = {:.1f}".format(step, correct, total, correct/total, loss.item(), time_left))

###
###
### Don't change code after this line
###
###
class Limits:
    def __init__(self):
        self.time_limit = 2.0
        self.size_limit = 1000000
        self.test_limit = 1.0

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10

    def create_data(self, data_size, input_size, random_input_size, seed):
        torch.manual_seed(seed)
        function_size = 1 << input_size
        function_input = torch.ByteTensor(function_size, input_size)
        for i in range(function_input.size(0)):
            fun_ind = i
            for j in range(function_input.size(1)):
                input_bit = fun_ind&1
                fun_ind = fun_ind >> 1
                function_input[i][j] = input_bit
        function_output = torch.ByteTensor(function_size).random_(0, 2)

        if data_size % function_size != 0:
            raise "Data gen error"

        data_input = torch.ByteTensor(data_size, input_size).view(-1, function_size, input_size)
        target = torch.ByteTensor(data_size).view(-1, function_size)
        for i in range(data_input.size(0)):
            data_input[i] = function_input
            target[i] = function_output
        data_input = data_input.view(data_size, input_size)
        target = target.view(data_size)
        if random_input_size > 0:
            data_random = torch.ByteTensor(data_size, random_input_size).random_(0, 2)
            data = torch.cat([data_input, data_random], dim=1)
        else:
            data = data_input
        perm = torch.randperm(data.size(1))
        data = data[:,perm]
        perm = torch.randperm(data.size(0))
        data = data[perm]
        target = target[perm]
        return (data.float(), target.view(-1, 1).float())

    def create_case_data(self, case):
        data_size = 256*32
        input_size = 8
        random_input_size = min(32, (case-1)*4)

        data, target = self.create_data(2*data_size, input_size, random_input_size, case)
        return sm.CaseData(case, Limits(), (data[:data_size], target[:data_size]), (data[data_size:], target[data_size:])).set_description("{} inputs and {} random inputs".format(input_size, random_input_size))

class Config:
    def __init__(self):
        self.max_samples = 10000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()

tic = time.time()
ACCS = {}
sm.SolutionManager(Config()).run(case_number=-1)

if len(ACCS) > 0:
    keys_len = len(ACCS)
    f, axs_tmp = plt.subplots(keys_len, 1, figsize=(12, 6*keys_len))
    axs = axs_tmp if keys_len > 1 else [axs_tmp]
    for i, (main_key, sub_accs) in enumerate(ACCS.items()):
        for key, values in sub_accs.items():
            axs[i].plot(values, label=key)
        axs[i].legend()
        axs[i].set_title(main_key)

toc = time.time()
print('proceed at %f s' %(toc - tic))