import librosa
import os
import numpy as np
import torch as torch
from torch.utils.data import Dataset
import pandas as pd
import json

class NSynthDataSet_RawAudio(Dataset):
    def __init__(self, meta_data_file, audio_dir, sr=16000):
        self.meta_data_file = meta_data_file
        self.audio_dir = audio_dir
        self.sr = sr

        sample_rate = 16000

        # For guitar
#         self.lower_pitch_limit = 44 #104hz
#         self.upper_pitch_limit = 80 #831Hz

        # For reed 1
#         self.lower_pitch_limit = 60 #262hz
#         self.upper_pitch_limit = 70 #467Hz
        
        # For reed 2
        self.lower_pitch_limit = 44 #104Hz
        self.upper_pitch_limit = 80 #831Hz

        self.sample_length = 512
        self.classes = [x for x in range(self.lower_pitch_limit, self.upper_pitch_limit)]

        self.quantization_channels = 255 #mu in librosa
        
        with open(meta_data_file) as f:
            params = json.load(f)
            self.nsynth_meta_df = pd.DataFrame.from_dict(params)
            self.nsynth_meta_df = self.nsynth_meta_df.transpose()
#             self.nsynth_meta_df = self.nsynth_meta_df[self.nsynth_meta_df['instrument_family_str'] == 'guitar']
            self.nsynth_meta_df = self.nsynth_meta_df[self.nsynth_meta_df['instrument_family_str'] == 'reed']
            self.nsynth_meta_df = self.nsynth_meta_df[(self.nsynth_meta_df['pitch'] >= self.lower_pitch_limit) \
                                                      & (self.nsynth_meta_df['pitch'] <= self.upper_pitch_limit)]

        
            # Augment this dataset by copying itself
            self.nsynth_meta_df['fold'] = 0
            
            nsynth_meta_df_orig = self.nsynth_meta_df.copy(deep=True)
            
            total_dupes = 7
            
            for dupe_idx in range(total_dupes):
                nsynth_meta_df_dupe = nsynth_meta_df_orig.copy(deep=True)
                nsynth_meta_df_dupe['fold'] = dupe_idx + 1
                nsynth_meta_df_dupe.index = nsynth_meta_df_dupe.index + f'-{dupe_idx+1}'
                self.nsynth_meta_df = pd.concat([self.nsynth_meta_df, nsynth_meta_df_dupe])
        
        print(self.nsynth_meta_df.shape)
        print('Unique pitches = ', sorted(self.nsynth_meta_df['pitch'].unique()))
        
    
    def __len__(self):
        return self.nsynth_meta_df.shape[0]
    
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx): 
            idx = idx.tolist()
        
        start_location = np.random.randint(1*self.sr, 3*self.sr) # Select random point between 1 and 3 seconds

        audio_file_name = self.nsynth_meta_df.iloc[idx].note_str + '.wav'
        audio_pitch = self.nsynth_meta_df.iloc[idx].pitch
        audio_data, _ = librosa.load(os.path.join(self.audio_dir, audio_file_name), sr=self.sr)
        input_data = audio_data[start_location:start_location + self.sample_length]
        input_data = (librosa.mu_compress(input_data, quantize=False) + 1)/2 # Bring values from range [-1 to 1] to [0 to 1]
        
        target = audio_data[start_location + 1:start_location + 1 + self.sample_length]
        target = librosa.mu_compress(target, quantize=True) + 127
        target = target.astype(np.long)
        
        return input_data, target


