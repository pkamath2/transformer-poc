import librosa
import os
import numpy as np
import torch as torch
from torch.utils.data import Dataset
import pandas as pd
import json

class NSynthDataSet_RawAudio(Dataset):
    def __init__(self, meta_data_file, audio_dir, sr=16000, lower_pitch_limit=60, upper_pitch_limit=74, split='train'):
        self.meta_data_file = meta_data_file
        self.audio_dir = audio_dir
        self.sr = sr

        sample_rate = 16000

        # For guitar
#         self.lower_pitch_limit = 44 #104hz
#         self.upper_pitch_limit = 80 #831Hz

        # For reed 1
        self.lower_pitch_limit = lower_pitch_limit #262hz #c4=60
        self.upper_pitch_limit = upper_pitch_limit #467Hz #d5=74 -- 1 octave + 2 extra pitches
        
        # For reed 2
#         self.lower_pitch_limit = 44 #104Hz
#         self.upper_pitch_limit = 80 #831Hz

        self.sample_length = 512
        self.classes = [x for x in range(self.lower_pitch_limit, self.upper_pitch_limit)]

        self.quantization_channels = 255 #mu in librosa
        
        self.instruments = ['reed', 'brass']
        
        with open(meta_data_file) as f:
            params = json.load(f)
            self.nsynth_meta_df = pd.DataFrame.from_dict(params)
            self.nsynth_meta_df = self.nsynth_meta_df.transpose()
            
            if split == 'train':
                self.nsynth_meta_df = self.nsynth_meta_df[(self.nsynth_meta_df['instrument_str'] == 'reed_acoustic_000') | (self.nsynth_meta_df['instrument_str'] == 'brass_acoustic_018')]
            else:
                self.nsynth_meta_df = self.nsynth_meta_df[(self.nsynth_meta_df['instrument_str'] == 'reed_acoustic_011') | (self.nsynth_meta_df['instrument_str'] == 'brass_acoustic_015')]
            
            self.nsynth_meta_df = self.nsynth_meta_df[(self.nsynth_meta_df['pitch'] >= self.lower_pitch_limit) \
                                                      & (self.nsynth_meta_df['pitch'] <= self.upper_pitch_limit)]

        
            
            self.nsynth_meta_df['amplitude_scale'] = 0.0001
            nsynth_meta_df_orig = self.nsynth_meta_df.copy(deep=True)
            for amplitude_scale in np.arange(0.1, 1, 0.1):
                nsynth_meta_df_dupe = nsynth_meta_df_orig.copy(deep=True)
                nsynth_meta_df_dupe['amplitude_scale'] = amplitude_scale
                nsynth_meta_df_dupe.index = nsynth_meta_df_dupe.index + f'-{amplitude_scale}'
                self.nsynth_meta_df = pd.concat([self.nsynth_meta_df, nsynth_meta_df_dupe])
            
            
            num_folds = 5
            self.nsynth_meta_df['fold'] = 0
            nsynth_meta_df_orig = self.nsynth_meta_df.copy(deep=True)
            for fold in range(num_folds):
                nsynth_meta_df_dupe = nsynth_meta_df_orig.copy(deep=True)
                nsynth_meta_df_dupe['fold'] = fold
                nsynth_meta_df_dupe.index = nsynth_meta_df_dupe.index + f'-fold{amplitude_scale}'
                self.nsynth_meta_df = pd.concat([self.nsynth_meta_df, nsynth_meta_df_dupe])
            
        
        print(self.nsynth_meta_df.shape)
        print('Unique pitches = ', sorted(self.nsynth_meta_df['pitch'].unique()))
        
    
    def __len__(self):
        return self.nsynth_meta_df.shape[0]
    
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx): 
            idx = idx.tolist()
        
        

        audio_file_name = self.nsynth_meta_df.iloc[idx].note_str + '.wav'
        audio_data, _ = librosa.load(os.path.join(self.audio_dir, audio_file_name), sr=self.sr)
        
        # Find start_location. If randomly chosen start_location is larger than actual audio_data length, change the start_location.
        start_location = np.random.randint(1*self.sr, 3*self.sr) # Select random point between 1 and 3 seconds
        a_data = np.flip(audio_data)
        a_data = a_data[np.argmax(a_data>0.1):]
        a_data = np.flip(a_data)
        if(start_location > len(a_data)):
            start_location = int(len(a_data)/2) + np.random.randint(1, self.sample_length)
        
        amplitude_scale = self.nsynth_meta_df.iloc[idx].amplitude_scale
       
        
        input_data = audio_data[start_location:start_location + self.sample_length] * amplitude_scale
        input_data = (librosa.mu_compress(input_data, quantize=False) + 1)/2 # Bring values from range [-1 to 1] to [0 to 1]
        
        audio_pitch = self.nsynth_meta_df.iloc[idx].pitch
        audio_pitch = (audio_pitch - self.lower_pitch_limit)/(self.upper_pitch_limit - self.lower_pitch_limit) # Bring values to to 0-1
        pitch = np.broadcast_to(np.array([audio_pitch]), input_data.shape)
        pitch.setflags(write=1) #Fix for weird Pytorch message "The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors."
        
        scale = np.broadcast_to(np.array([amplitude_scale]), input_data.shape)
        scale.setflags(write=1) #Fix for weird Pytorch message "The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors."
        
        audio_instrument = self.instruments.index(self.nsynth_meta_df.iloc[idx].instrument_family_str)
        instrument = np.broadcast_to(np.array([audio_instrument]), input_data.shape)
        instrument.setflags(write=1) #Fix for weird Pytorch message "The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors."
        
        input_data = np.stack((input_data, pitch, scale, instrument), axis=0)
        input_data = input_data.astype(np.float)
        
        target = audio_data[start_location + 1:start_location + 1 + self.sample_length] * amplitude_scale
        target = librosa.mu_compress(target, quantize=True) + 127
        target = target.astype(np.long)
        
        return input_data, target

    
    
#         audio_pitch = (((audio_pitch - self.lower_pitch_limit) * 2 / (self.upper_pitch_limit - self.lower_pitch_limit)) - 1) #Scale to -1 to +1