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
from torch.utils.tensorboard import SummaryWriter

import random, sys, math, gzip, os
from tqdm import tqdm

#Constants/Config
base_data_dir = '/home/purnima/appdir/Github/DATA/NSynth/'
train_data_dir = os.path.join(base_data_dir,'nsynth-train','audio')
test_data_dir = os.path.join(base_data_dir,'nsynth-test','audio')
validate_data_dir = os.path.join(base_data_dir,'nsynth-valid','audio')

labels_dir = '/home/purnima/appdir/Github/DATA/NSynth'
labels_file_name = 'examples.json'
# labels_file_name = 'examples-subset-full-acoustic-3000.json'

labels_train_dir = os.path.join(labels_dir,'nsynth-train', labels_file_name)
labels_test_dir = os.path.join(labels_dir,'nsynth-test', labels_file_name)
labels_validate_dir = os.path.join(labels_dir,'nsynth-valid', labels_file_name)

batch_size = 16
sample_length = 256 #For context
sample_rate = 16000
embedding_size = 1024 #256
num_heads = 8
depth = 12
num_tokens = 256 

lr = 0.001
lr_warmup = 5000
epochs = 20

def main():
    tbw = SummaryWriter(log_dir='./runs')

    train_ds = NSynthDataSet_RawAudio(meta_data_file=labels_train_dir, audio_dir=train_data_dir, sr=sample_rate)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    test_ds = NSynthDataSet_RawAudio(meta_data_file=labels_test_dir, audio_dir=test_data_dir, sr=sample_rate)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # validate_ds = NSynthDataSet_RawAudio(meta_data_file=labels_validate_dir, audio_dir=validate_data_dir, sr=sample_rate)
    # validate_loader = torch.utils.data.DataLoader(validate_ds, batch_size=batch_size, shuffle=False)

    print('Data Loaded')

    model = GTransformer(emb=embedding_size, heads=num_heads, depth=depth, seq_length=sample_length, num_tokens=num_tokens, attention_type=None)
    model = model.cuda()

    opt = torch.optim.Adam(lr=lr, params=model.parameters())
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (lr_warmup / batch_size), 1.0))

    
    iteration = 0
    for epoch in range(epochs):
        for i, (data, target) in enumerate(tqdm(train_loader)):
            opt.zero_grad()
            data = data.cuda()
            target = target.cuda()

            tic()
            output = model(data)
            t = toc()

            # Compute the loss
            loss = F.nll_loss(output.transpose(2, 1), target, reduction='mean')

            tbw.add_scalar('transformer/train-loss', float(loss.item()), iteration)
            # tbw.add_scalar('transformer/time-forward', t)

            loss.backward() # backward pass

            opt.step() # stochastic gradient descent step
            sch.step() # update the learning rate
            iteration += 1

            
        #Prepare for sampling - only on epoch%5
        iterator = iter(train_loader)
        orig_test_data, _ = iterator.next()
        
        
        test_data = orig_test_data[13].view(1,-1)
        orig_waveform = mulawDecode(test_data.view(-1).cpu())
        test_data = test_data.cuda()

        test_data = test_data.detach().clone()
        test_data_ = test_data.detach().clone()
        if epoch%5 == 0:
            with torch.no_grad():
                print(f'Epoch = {epoch} and Loss = {loss.item()}')

                for ind in range(sample_length):
                    sample_data = model(test_data)
                    sample_data = sample_data.transpose(2, 1)
                    sample_data = sample_data[0, -1, :].argmax()
                    
                    sample_data = sample_data.view(1,-1)
                    test_data_ = torch.cat([test_data_, sample_data], dim=1)
                    test_data = test_data_[:,:sample_length].view(1,-1)


            waveform = mulawDecode(test_data_.view(-1).cpu())
            # print(waveform)
            plt.figure()
            plt.plot(orig_waveform)
            plt.savefig(f'samples/{epoch}-orig.png')
            plt.figure()
            plt.plot(waveform)
            plt.savefig(f'samples/{epoch}.png')

            sf.write(f'samples/{epoch}-orig.wav', orig_waveform, sample_rate)
            sf.write(f'samples/{epoch}.wav', waveform, sample_rate)

            save_model(epoch, model, opt, loss, f'checkpoint/attention-{epoch}.pt')

# From Huz's MTCRNN codebase
def mulawDecode(output):
    # quantization_channels = 255
    # mu = (quantization_channels - 1)

    # expanded = (output / quantization_channels) * 2. - 1
    # waveform = np.sign(expanded) * (
    #             np.exp(np.abs(expanded) * np.log(mu + 1)) - 1
    #         ) / mu
    output = output.numpy() - 128
    waveform = librosa.mu_expand(output, quantize=True)
    return waveform

def save_model(epoch, model, opt, loss, model_location):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': loss,
            }, model_location) 
        print('Saved Model')


def load_model(model, opt, model_location):
    checkpoint = torch.load(model_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss']
    epoch = checkpoint['epoch']
    return model, opt, loss, epoch
    

if __name__ == "__main__":
    main()