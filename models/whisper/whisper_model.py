from . import whisper

import torch
from torch import nn
import torch.nn.functional as F


class CNN(torch.nn.Module):
    def __init__(self, n_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 372 * 125, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 372 * 125)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    

class WhisperModel(torch.nn.Module):
    def __init__(self, n_classes=264, pre_trained=True):
        super(WhisperModel, self).__init__()
        assert n_classes is not None, "'n_classes' cannot be None. Specify 'n_classes' present in the dataset."
        
        self.audio_encoder = whisper.load_model("base", pre_trained=pre_trained).encoder
        self.classifier = CNN(n_classes)

    def forward(self, x):
        # Pass input through Whisper encoder
        features = self.audio_encoder(x)

        # Unsqueeze to add a channel dimension
        features = features.unsqueeze(1)

        # Pass the features through the CNN classifier
        logits = self.classifier(features)

        return logits


    

