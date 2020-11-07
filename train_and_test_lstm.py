#!/usr/bin/env python

import torch
from dataset import Dataset
from feedforward_network import LSTMNetwork
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sys import argv
from sklearn import metrics

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Choosing parameters

batch_size = 1
layer_number = 3
layer_width = 50
lr8 = 0.001

# Setting parameters
train_params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 1}

test_params = {'batch_size': 1,
            'shuffle': False,
            'num_workers': 1}

max_epochs = 100

jar = open('data/glove_lstm/partition_dict.pkl','rb')
partition = pickle.load(jar)
jar2 = open('data/glove_lstm/label_dict.pkl','rb')
labels = pickle.load(jar2)
jar.close()
jar2.close()

# Generators

training_set = Dataset(partition['train'], labels,'glove_lstm')
training_generator = torch.utils.data.DataLoader(training_set, **train_params)

test_set = Dataset(partition['test'], labels,'glove_lstm')
test_generator = torch.utils.data.DataLoader(test_set, **test_params)

# Models

ffn = LSTMNetwork(100, layer_number, layer_width)
ffn.cuda()
# criterion = nn.BCELoss()
criterion = nn.L1Loss()
optimizer = optim.SGD(ffn.parameters(), lr=float(lr8), momentum=0.9)

def test_loss():
    'Calculates loss from model on test set'
    ffn.eval()
    test_loss = 0
    for x, y in test_generator:
        ffn.zero_grad()
        x = torch.transpose(x,0,1)
        x = x.to(device=torch.device('cuda:0')).float()
        y = y.to(device=torch.device('cuda:0'))
        pred_output = ffn(x).squeeze(1)
        pred_output = pred_output[-1,:]
        loss = criterion(pred_output.float(), y.unsqueeze(1).float())
        test_loss += loss.data.cpu().numpy()

    return test_loss

num_epochs = 0
test_loss_list = []


# Loop over epochs

for epoch in range(max_epochs):
    print('epoch {} done'.format(num_epochs))
    # Training
    total_loss = []
    ffn.train()
    num_batches = 0
    for x, y in training_generator:

        ffn.zero_grad()

        x = torch.transpose(x,0,1)
        num_batches += 1
        # Transfer to GPU
        x, y = x.to(device).float(), y.to(device)
        # y = y.unsqueeze(1)
        # Model computations
        pred_labels = ffn(x).squeeze(1)
        pred_labels = pred_labels[-1,:]
        # exit()
        # pred_labels = pred_labels.squeeze(1)
        loss = criterion(pred_labels.float(), y.float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss.append(loss.data.cpu().numpy())

    # print('mean loss in epoch ', epoch, sum(total_loss)/len(total_loss))
    t_loss = test_loss()
    print('test loss: ', t_loss)
    test_loss_list.append(t_loss)


    # Early stop
    if num_epochs > 10:
        test_loss_list.pop(0)
        if t_loss > test_loss_list[0]:
            break

    num_epochs += 1
    torch.save(ffn, 'ffn.p')
# Evaluation

pred_output_list=[]
labels=[]

ffn = torch.load('ffn.p')

# push test set through one more time

ffn.eval()
for x, y in test_generator:
    x = x.to(device=torch.device('cuda:0')).float().transpose(0,1)
    y = y.to(device=torch.device('cuda:0'))
    pred_output = ffn(x)[-1,:,:]
    # print(pred_output)
    # print(pred_output.cpu().detach().numpy()[0][0])
    # exit()

    pred_output_list.append(pred_output.cpu().detach().numpy()[0][0])
    labels.append(y.cpu().numpy()[0])

# generate predictions given models outputs, then compute f1

pred_labels = [0 if output-0.5<0 else 1 for output in pred_output_list]
print('\n\n\n ################# RESULTS ##############\n\n')
print(metrics.f1_score(labels, pred_labels, pos_label=1))
