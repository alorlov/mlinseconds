# There are random function from 8 inputs.
# There are random input vector of size 8 * number of voters.
# We calculate function number of voters times and sum result.
# We return 1 if sum > voters/2, 0 otherwise
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

first_model = [{}]

class SolutionFirstModel(nn.Module):
    def __init__(self, input_size, output_size, params):
        super(SolutionFirstModel, self).__init__()
        self.params = params
        self.input_size = input_size
        self.activations = ['relu'] + (['relu'] * (len(params['hidden_sizes']) - 1)) + ['sigmoid']

        sizes = (self.input_size, *params['hidden_sizes'], output_size)
        
        sequence = []
        for x in range(1, len(sizes)):
            sequence.append(('linear%i' % x, nn.Linear(sizes[x - 1], sizes[x])))
            sequence.append(('bn%i' % x, nn.BatchNorm1d(sizes[x], track_running_stats=False)))
            sequence.append(('%s%i' % (self.activations[x - 1], x), ACTIVATIONS_GRID[self.activations[x - 1]]))
        self.fc = nn.Sequential(OrderedDict(sequence))

    def forward(self, x):
        return self.fc(x)

    def calc_loss(self, output, target):
        loss = LOSSES[self.params['loss']](output, target)
        return loss
        
    def calc_predict(self, output):
        predict = output.round()
        return predict
    
class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size, params):
        super(SolutionModel, self).__init__()
        self.params = params
        self.input_size = input_size // params.features_size * params.features_out_size
        self.activations = ['sigmoid'] + (['relu'] * (len(params.hidden_sizes) - 1)) + ['sigmoid']
        
        if params.grid_search.enabled:
            torch.manual_seed(params.random)

        sizes = (self.input_size, *params.hidden_sizes, output_size)
        
        sequence = []
        for x in range(1, len(sizes)):
            sequence.append(('linear%i' % x, nn.Linear(sizes[x - 1], sizes[x])))
            sequence.append(('bn%i' % x, nn.BatchNorm1d(sizes[x], track_running_stats=False)))
            sequence.append(('%s%i' % (self.activations[x - 1], x), ACTIVATIONS_GRID[self.activations[x - 1]]))
            if x == 1: 
                sequence.append(('dropout%i' % x, nn.Dropout(p=params.prob_out)))
        self.fc = nn.Sequential(OrderedDict(sequence))

        # Create first model
        params1 = {
            'input_size': params.features_size,
            'hidden_sizes': params.hidden_sizes1,
            'batch_norm': True,
        }
        first_model[0] = SolutionFirstModel(params1['input_size'], params.features_out_size, params1)

    def forward(self, data):
        output = first_model[0](data.view(-1, self.params.features_size))
        first_model[0].output = output.view(-1, self.input_size)
        
        return self.fc(first_model[0].output)

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
        self.plot_main_key = 'lr'
        
        # General settings
        self.features_size = 8
        self.features_out_size = 2
        self.batch_size = 256
        self.batch_norm = True
        
        # First Model
        layers1 = 2
        self.hidden_sizes1 = [30] * layers1

        # Main Model
        layers = 1
        self.hidden_sizes = [15] * layers
        self.lr = 0.02
        
        self.loss = 'MSELoss'
        self.momentum = 0.5
        
        self.lr_factor = 0.1
        self.lr_step_size = [100000]
        self.weight_decay = 0
        self.prob_out = 0
        
        self.random = 3
        
#        self.features_out_size_grid = [1,2,3,4]
#        
#        layers_range = [2]
#        self.hidden_sizes1_grid = [
#            [i] * n
##            for i in [80]
#            for n in layers_range
#            for i in [30, 60, 90]#np.around(np.geomspace(4, 100, 4)).astype(int)
#        ]
#        layers_range = [1,2]
#        self.hidden_sizes_grid = [
#            [i] * n
##            for i in [80]
#            for n in layers_range
#            for i in np.around(np.geomspace(4, 60, 5)).astype(int)
#        ]
#        self.lr_grid = np.round(np.geomspace(0.001, 1., 4), 5)
        self.lr_grid = [.08, .04, .02, .01, .008]

#        self.batch_size_grid = np.around(np.geomspace(128, 512, 3, dtype=int)).astype(int)
        
#        self.loss_grid = list(LOSSES)
        
        self.momentum_grid = [0, .1, .2, 0.5, .7, 0.9, 0.95, 0.975, 0.99, 0.995]
        
#        self.lr_factor_grid = [0.1, 0.5]
#        self.lr_step_size_grid = [[i] for i in [100, 1e+5]]
        
#        self.weight_decay_grid = np.append(0, np.geomspace(1e-6, 1e-2, 5))
#        self.prob_out_grid = np.append(0, np.geomspace(0.01, 0.1, 4))
        
        self.random_grid = range(6,11)
        self.merge_random_param = True
        
        self.grid_search = GridSearch(self).set_enabled(IS_GRID_SEARCH)

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
        break_on_correct = True
        tune = 0
        random_size = len(self.random_grid) if hasattr(self, 'random_grid') else 1
        
        if self.grid_search.enabled:
            key = self.get_key(self.merge_random_param)
            if tune:
                print(key)
            plot_main_key = "{}-{}".format(self.plot_main_key, getattr(self, self.plot_main_key))
            if not key in self.sols:
                self.sols[key] = 0
                self.solsSum[key] = 0
                # plotting init
                if not plot_main_key in ACCS:
                    ACCS[plot_main_key] = {}
                    ACCS[plot_main_key+'_model1'] = {}
                ACCS[plot_main_key][key] = []
                ACCS[plot_main_key+'_model1'][key] = []
            if self.sols[key] == -1:
                return
        step = 0
        model1 = first_model[0]

        # Put model in train mode
        model.train()
        
#        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        optimizer = optim.Adam([
                                    {'params': model1.parameters()},
                                    {'params': model.parameters()},
                                ],
            lr=self.lr, betas=(self.momentum, 0.999), weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_step_size, gamma=self.lr_factor)
        
        train_dataset = Data.TensorDataset(train_data, train_target)
        train_loader = Data.DataLoader(dataset=train_dataset, batch_size=int(self.batch_size), shuffle=True)
        
#        test_data, test_target = context.get_case_data().test_data
        dev_size = train_data.shape[0] // 2
        dev_data, dev_target = train_data[:dev_size], train_target[:dev_size]
        
        while True:
            
            for train_data_mini, train_target_mini in train_loader:
                time_left = context.get_timer().get_time_left()
                if time_left < 0.1:
                    if self.grid_search.enabled:
                        self.sols[key] = -1
                    break
                
                # model.parameters()...gradient set to zero
                optimizer.zero_grad()
#                optimizer1.zero_grad()
                
                data = train_data_mini
                target = train_target_mini
                
                # evaluate model => model.forward(data)
                output = model(data)
                # if x < 0.5 predict 0 else predict 1
                predict = output.round()
                predict1 = (model1.output.round().sum(dim=1) > (model.input_size / 2)).type_as(target)
                # Number of correct predictions
                correct = predict.eq(target.view_as(predict)).long().sum().item()
                correct1 = predict1.eq(target.view_as(predict1)).long().sum().item()
                # Total number of needed predictions
                total = target.view(-1).size(0)
                
                accuracy = correct/total*100
                accuracy1 = correct1/total
                self.grid_search.log_step_value('ratio', accuracy, step)
                
                # calculate loss
                loss = model.calc_loss(output, target)

                if self.grid_search.enabled:
                    ACCS[plot_main_key][key].append(loss)
                    ACCS[plot_main_key + '_model1'][key].append(correct1/total)
                
                self.grid_search.log_step_value('loss', loss.item(), step)

                
                # calculate deriviative of model.forward() and put it in model.parameters()...gradient
                loss.backward()

                # print progress of the learning
                if tune and step % 50 == 0:
                    print("Step = {} Prediction = {}/{} Ratio = {:.3f} Error = {:.4f}/{:.4f} time = {:.1f}"
                          .format(step, correct, total, correct/total, loss.item(), 1 - accuracy1, time_left))
                
                # update model: model.parameters() -= lr * gradient
                optimizer.step()
                scheduler.step()
                step += 1
            else:
                if total == correct:
                    stat = sm.SolutionManager(Config()).calc_model_stats(model, dev_data, dev_target)
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
                    else:
                        self.print_stats(step, stat['loss'], stat['correct'], stat['total'], time_left, factor=1)
                    
                continue
            break
        return step
    
    def print_stats(self, step, loss, correct, total, time_left, optimizer=None, factor=100):
        if step % factor == 0:
#            print(get_lr(optimizer))
            print("Step = {} Prediction = {}/{} Ratio = {:.3f} Error = {:.4f} time = {:.1f}".format(step, correct, total, correct/total, loss, time_left))

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
###
###
### Don't change code after this line
###
###
class Limits:
    def __init__(self):
        self.time_limit = TIME_LIMIT
        self.size_limit = 1000000
        self.test_limit = 1.0

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10

    def get_index(self, tensor_index):
        index = 0
        for i in range(tensor_index.size(0)):
            index = 2*index + tensor_index[i].item()
        return index

    def calc_value(self, input_data, function_table, input_size, input_count_size):
        count = 0
        for i in range(input_count_size):
            count += function_table[self.get_index(input_data[i*input_size: (i+1)*input_size])].item()
        if count > input_count_size/2:
            return 1
        else:
            return 0

    def create_data(self, data_size, input_size, input_count_size, seed):
        torch.manual_seed(seed)
        function_size = 1 << input_size
        function_table = torch.ByteTensor(function_size).random_(0, 2)
        total_input_size = input_size*input_count_size
        data = torch.ByteTensor(data_size, total_input_size).random_(0, 2)
        target = torch.ByteTensor(data_size)
        for i in range(data_size):
            target[i] = self.calc_value(data[i], function_table, input_size, input_count_size)
        return (data.float(), target.view(-1, 1).float())

    def create_case_data(self, case):
        input_size = 8
        data_size = (1<<input_size)*32
        input_count_size = case
        data, target = self.create_data(2*data_size, input_size, input_count_size, case)
        return sm.CaseData(case, Limits(), (data[:data_size], target[:data_size]), (data[data_size:], target[data_size:])).set_description("{} inputs per voter and {} voters".format(input_size, input_count_size))

class Config:
    def __init__(self):
        self.max_samples = 10000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()

tic = time.time()
ACCS = {}
IS_GRID_SEARCH = 0
CASE_NUMBER = -1
TIME_LIMIT = 2.0

sm.SolutionManager(Config()).run(case_number=CASE_NUMBER)

if len(ACCS) > 0:
    keys_len = len(ACCS)
    f, axs_tmp = plt.subplots(keys_len, 1, figsize=(12, 6*keys_len))
    axs = axs_tmp if keys_len > 1 else [axs_tmp]
    for i, (main_key, sub_accs) in enumerate(ACCS.items()):
        for key, values in sub_accs.items():
            axs[i].plot(values, label=key)
        axs[i].legend()
        axs[i].set_xlabel('steps')
#        axs[i].set_ylabel('accuracy')
        axs[i].set_title(main_key)
#        axs[i].set_ylim(0, 0.5)

toc = time.time()
print('proceed at %f s' %(toc - tic))
