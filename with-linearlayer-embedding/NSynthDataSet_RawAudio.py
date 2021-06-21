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

        self.lower_pitch_limit = 44 #104hz
        self.upper_pitch_limit = 80 #831Hz

        self.sample_length = 512
        self.classes = [x for x in range(self.lower_pitch_limit, self.upper_pitch_limit)]

        self.quantization_channels = 255 #mu in librosa
        
        with open(meta_data_file) as f:
            params = json.load(f)
            self.nsynth_meta_df = pd.DataFrame.from_dict(params)
            self.nsynth_meta_df = self.nsynth_meta_df.transpose()
            self.nsynth_meta_df = self.nsynth_meta_df[self.nsynth_meta_df['instrument_family_str'] == 'guitar']
            self.nsynth_meta_df = self.nsynth_meta_df[(self.nsynth_meta_df['pitch'] >= self.lower_pitch_limit) \
                                                      & (self.nsynth_meta_df['pitch'] < self.upper_pitch_limit)]

            # Augment this dataset 4 times by copying itself
            self.nsynth_meta_df['fold'] = 1
            nsynth_meta_df_2 = self.nsynth_meta_df.copy(deep=True)
            nsynth_meta_df_2['fold'] = 2
            nsynth_meta_df_2.index = nsynth_meta_df_2.index + '-2'
            nsynth_meta_df_3 = self.nsynth_meta_df.copy(deep=True)
            nsynth_meta_df_3['fold'] = 3
            nsynth_meta_df_3.index = nsynth_meta_df_3.index + '-3'
            nsynth_meta_df_4 = self.nsynth_meta_df.copy(deep=True)
            nsynth_meta_df_4['fold'] = 4
            nsynth_meta_df_4.index = nsynth_meta_df_4.index + '-4'
            self.nsynth_meta_df = pd.concat([self.nsynth_meta_df, nsynth_meta_df_2, nsynth_meta_df_3, nsynth_meta_df_4])
        
        print(self.nsynth_meta_df.shape)
    
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
        target = audio_data[start_location + self.sample_length + 1]
    
        input_data = librosa.mu_compress(input_data, quantize=True)
        input_data = input_data/self.quantization_channels
        target = librosa.mu_compress(target, quantize=True) + 127
#         target = target/self.quantization_channels

        return input_data, target.astype(np.long)


