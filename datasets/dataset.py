import os
import torch
import random
from torch.utils.data import Dataset

from datasets.spectrogram_augmentations import time_masking, freq_masking


class BirdClefDataset(Dataset):
    
    def __init__(self, audio_files_path, labels, main_folder_path, bird2label_dict, label2bird_dict, spec_augment, spec_aug_prob=0.5):

        self.audio_files_path = audio_files_path
        self.labels = labels
        self.main_folder_path = main_folder_path
        self.bird2label_dict = bird2label_dict
        self.label2bird_dict = label2bird_dict
        self.spec_augment = spec_augment
        self.probability_0 = (1 - spec_aug_prob)
        self.probability_1 = spec_aug_prob
        
    
    # this function is used by 'Dataset' class to check the total number of samples in dataset
    def __len__(self):
        return len(self.audio_files_path)
    
    
    def __getitem__(self, index):
        # get path of audio file indicated by index
        # self.audio_files_path is a list which contains paths of audio files. each element in this list corresponds to the path of an audio file.
        audio_path = self.audio_files_path[index]  # audio_path contains the address of audio which is specified in index
    
    
        # get full path of the mel file
        # audio_full_path is a variable that stores the complete file path to the audio file we want to load.
        audio_full_path = os.path.join(self.main_folder_path, audio_path)

        mel = torch.load(audio_full_path)
        label = torch.tensor(self.labels[index])

        # Apply augmentation if required
        if self.spec_augment:
            random_numbers = random.choices([0, 1], weights=[self.probability_0, self.probability_1], k=1)
            if random_numbers[0] == 1:
                mel = freq_masking(mel)
                mel = time_masking(mel)

        return mel, label, self.label2bird_dict[label.item()]