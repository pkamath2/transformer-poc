from util import d, here, tic, toc
from NSynthDataSet_RawAudio import NSynthDataSet_RawAudio
from transformers import GTransformer

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.data import RandomSampler

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa

from argparse import ArgumentParser

import random, sys, math, gzip, os
from tqdm import tqdm

# Data location
data_dir = '/home/purnima/appdir/Github/DATA/nsynth.64.76.dl/'

### Constants 
sample_rate = 16000
batch_size = 32
lr = 0.0001
lr_warmup = 10000
epochs = 60000

sample_length = 512 # For context
embedding_size = 128 
num_heads = 8 # Number of chunks for 'parallel/ensemble' computation
depth = 12 # Number of transformer layers
num_tokens = 256 #Size of the dictionary

lower_pitch_limit = 64
upper_pitch_limit = 76

def save_model(epoch, model, opt, loss, model_location):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'loss': loss,
        }, model_location) 
    print('Saved Model', flush=True)

def get_scaled_pitch(pitch):
    return (pitch - lower_pitch_limit)/(upper_pitch_limit - lower_pitch_limit)

def load_model(model, opt, model_location):
    checkpoint = torch.load(model_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss']
    epoch = checkpoint['epoch']
    return model, opt, loss, epoch

train_ds = NSynthDataSet_RawAudio(data_dir=data_dir, sr=sample_rate, sample_length=sample_length)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

model = GTransformer(emb=embedding_size, heads=num_heads, depth=depth, seq_length=sample_length, num_tokens=num_tokens, attention_type=None)
model = model.cuda()

opt = torch.optim.Adam(lr=lr, params=model.parameters())#, weight_decay=0.01)
sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (lr_warmup / batch_size), 1.0), verbose=False)
loss = torch.nn.NLLLoss(reduction='mean')

print(model)

def train():
    training_loss = 0.0
    model.train()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Epoch:{epoch}')):
        opt.zero_grad()
        
        b, cols, seq_len = data.shape
        reshaped_data = data.view(b, seq_len, cols)
        reshaped_data = reshaped_data.cuda().float()
        target = target.cuda()
        
        output = model(reshaped_data)
        running_loss = loss(output.transpose(2, 1), target)
        training_loss += running_loss.item()

        running_loss.backward() # backward pass
        
        gradient_clipping = 1.0
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

        opt.step() # stochastic gradient descent step
#         sch.step()
    
    sch.step()
    training_loss /= len(train_loader)
    print(f'Epoch training loss = {training_loss}, Epoch last LR = {sch.get_last_lr()}', flush=True)
    return training_loss


history_train = {'loss': []}

for epoch in range(0, epochs, 1):
    train_loss = train()
    history_train['loss'].append(train_loss)
    
    if epoch%10 == 0 or epoch == epochs-1:
        fig, axes = plt.subplots(ncols=1, figsize=(7, 7))
        axes.plot(history_train['loss'])
        axes.set_title('Train Loss')
        axes.set_xlabel('epoch')
        axes.set_ylabel('loss')
        
        plt.savefig(f'plots/train_loss_{epoch}.png')
        
        plt.close(fig)

        save_model(epoch, model, opt, loss, f'checkpoint/attention-{epoch}.pt')













