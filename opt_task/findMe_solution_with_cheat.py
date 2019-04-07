import numpy as np
import torch
from itertools import combinations 
import time

# Task is to write solution without using key
def create_data(data_size, input_size, random_input_size, seed):
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

def get_ind(arr):
    return int("".join(map(str, np.array(arr, dtype=int))), 2)

class SolutionModel():
    def __init__(self):
        self.sols = {}
        self.mask = ()
    
    def find_mask(self, datab, target, input_size, random_input_size):
        data = datab[:1024]
        n = input_size + random_input_size
        comb = combinations(range(n), random_input_size) 
        count = 0
        for c in list(comb):
            count += 1
#            continue
            results = np.array([])
            for s in range(10):
                input_sample = np.delete(data, c, axis=1)
                samples = np.nonzero((input_sample == input_sample[s]).all(axis=1))
                is_equal = (target[samples] == target[samples][0]).all(axis=0)
                if not is_equal:
                    break
                results = np.append(results, is_equal.item())
            else:
                is_equal = (results == True).all(axis=0)
                if is_equal:
                    self.mask = c
                    return self.mask
        print('Combinations of random columns: ', count)
            
    # generate solutions' list
    def train(self, data, target, input_size, random_input_size):
        mask = self.find_mask(data, target, input_size, random_input_size)
        arr = np.delete(data, mask, axis=1)
        for i in range(arr.shape[0]):
            key = get_ind(arr[i])
            self.sols[key] = target[i].item()
    
    def predict(self, data):
        arr = np.delete(data, self.mask, axis=1)
        results = []
        for i in range(arr.shape[0]):
            results.append(self.sols[get_ind(arr[i])])
            
        return np.array(results)
        


input_size = 8
random_input_size = 12
data_size = 256*32

data, target = create_data(2*data_size, input_size, random_input_size, 1)
train_data, train_target = np.array(data[:data_size].tolist()), np.array(target[:data_size].tolist())
test_data, test_target = np.array(data[data_size:].tolist()), np.array(target[data_size:].tolist())

print('{} inputs and {} random inputs'.format(input_size, random_input_size))
model = SolutionModel()
tic = time.time()
model.train(train_data, train_target, input_size, random_input_size)
toc = time.time()
print('Train time: %f s' %(toc - tic))
predicts = model.predict(test_data).reshape(-1, 1)
correct = np.where(predicts == test_target, 1, 0).sum()
print("Test correct/total = {}/{}".format(correct, len(test_data)))


