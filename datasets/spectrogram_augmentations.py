import torch
import random
import numpy as np


def time_masking(mel_spec):
    starting_point = random.sample(range(3000), 1)
    width = random.sample(range(300, 500), 1)
    temp_mel = mel_spec.clone()
    temp_mel[:, starting_point[0] : starting_point[0] + width[0]] = -1

    # tensor_to_img(temp_mel.numpy())
    return temp_mel

def freq_masking(mel_spec):
    starting_point = random.sample(range(1, 80), 1)
    width = random.sample(range(12, 16), 1)
    temp_mel = mel_spec.clone()
    temp_mel[starting_point[0] : starting_point[0] + width[0], :] = -1

    # tensor_to_img(temp_mel.numpy())
    return temp_mel