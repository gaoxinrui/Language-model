# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn

import data
import model
import os
import os.path as osp

parser = argparse.ArgumentParser(description='PyTorch ptb Language Model')
parser.add_argument('--model', type=str, default='LSTM', help='type of recurrent net (RNN, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200, help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200, help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2, help='number of layers')
parser.add_argument('--epochs', type=int, default=40, help='upper epoch limit')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--lr', type=float, default=20, help='initial learning rate')
parser.add_argument('--batch_size', type=int, default=20, metavar='N', help='batch size')
parser.add_argument('--max_sql', type=int, default=35, help='sequence length')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true', help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1234,help='set random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA device')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU device id used')
parser.add_argument('--log_interval', type=int, default=100,help='report interval')# 每隔多少个批次输出一次状态
parser.add_argument('--save', type=str, default='./model/model.pt', help='path to save the final model')
args = parser.parse_args()
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
# Use gpu or cpu to train
use_gpu = True
if use_gpu:
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(args.gpu_id)
else:
    device = torch.device("cpu")

# load data
data_loader = data.Corpus("./data/ptb", args.batch_size, args.max_sql)
'''
file_id1 = data_loader.tokenize("./data/ptb/train.txt")
file_id2 = data_loader.tokenize("./data/ptb/valid.txt")
data_loader.set_train()
data1, target1, end_flag1 = data_loader.get_batch()
'''

nvoc = len(data_loader.vocabulary)
model = model.LMModel(args.model, nvoc, args.emsize, args.nhid, args.nlayers, args.dropout).to(device)
#print(model)

criterion = nn.CrossEntropyLoss()

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
    
# Evaluation Function
def evaluate():
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    nvoc = len(data_loader.vocabulary)
    hidden = model.init_hidden(args.batch_size)
    data_loader.set_valid()
    with torch.no_grad():
        for i in range(1,data_loader.valid_batch_num+1):
            data, target, end_flag = data_loader.get_batch()
            data = data.to(device)
            target = target.to(device)
            output, hidden = model(data,hidden)
            output_flag = output.view(-1,nvoc)

            total_loss += len(data) * criterion(output_flag, target).item()
            hidden = repackage_hidden(hidden)
            if end_flag == True:
                break
    return (total_loss / len(data_loader.valid))

# Calculate the average cross-entropy loss between the prediction and the ground truth word.
# And then exp(average cross-entropy loss) is perplexity.

# Train Function
def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.0
    nvoc = len(data_loader.vocabulary)
    hidden = model.init_hidden(args.batch_size)
    data_loader.set_train()
    for batch in range(1,data_loader.train_batch_num+1):
        data, target, end_flag = data_loader.get_batch()
        data = data.to(device)
        target = target.to(device)
        if end_flag == True:
            break
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)#data.size=35*20,output.size=35*20*10000,此处数据是如何计算的？
        loss = criterion(output.view(-1,nvoc),target)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(),args.clip)
        for p in model.parameters():
            p.data.add_(-lr,p.grad.data)
        total_loss += loss.item()
        if batch % args.log_interval == 0 and batch  > 0 :
            cur_loss = total_loss / args.log_interval
            print("Epoch: {:3d}, {:5d}/{:5d} batches, lr: {:2.3f}, loss: {:5.2f}, ppl: {:8.2f}".format(epoch,batch, len(data_loader.train)//args.max_sql, lr, cur_loss, math.exp(cur_loss)))
            total_loss = 0.0

# Loop over epochs.
lr = args.lr
best_val_loss = 10000
# Loop over epochs.
for epoch in range(1, args.epochs+1):
    train()    

    if epoch % 10 == 0: 
        lr /= 2.0
    val_loss = evaluate()
    
    
    print("Epoch: {:3d}, valid loss: {:5.2f}, valid ppl: {:8.2f}".format(epoch,val_loss, math.exp(val_loss)))
    print('-' * 89)
    if val_loss < best_val_loss:
        with open(args.save, 'wb') as f:
            torch.save(model, f)
        best_val_loss = val_loss
'''
plt.figure()
plt.subplot(121)
plt.plot(train_lossHis, 'r-', label='train loss')
plt.plot(val_lossHis, 'b-', lable='valid loss')
plt.legend(loc='upper left')
plt.title('train&valid loss')
plt.subplot(122)   
plt.plot(train_pplHis, 'b-', lable='valid loss')
plt.plot(val_pplHis, 'b-', lable='valid ppl')
plt.legend(loc='upper left')
plt.title('train&valid ppl')
'''
