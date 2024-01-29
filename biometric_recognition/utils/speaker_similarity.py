import random
import os
import numpy as np
import pickle

from deep_speaker.audio import read_mfcc
from deep_speaker.batcher import sample_from_mfcc
from deep_speaker.constants import SAMPLE_RATE, NUM_FRAMES
from deep_speaker.conv_models import DeepSpeakerModel
from deep_speaker.test import batch_cosine_similarity

from utils.root_path import root

# Reproducible results.
np.random.seed(123)
random.seed(123)

# Define the model here.
model = DeepSpeakerModel()

# Sample some inputs for WAV/FLAC files for the same speaker.
# To have reproducible results every time you call this function, set the seed every time before calling it.
# np.random.seed(123)
# random.seed(123)

# Path: speaker_similarity.py
# Load the checkpoint.
model.m.load_weights(
    f'{root}/model/ResCNN_triplet_training_checkpoint_265.h5',
    by_name=True)

# Sample some inputs for WAV/FLAC files for the same speaker.
predict_list = []
voice_similarity = []

# get only .wav file
voice_list = [f for f in os.listdir(f"{root}/data/voice_data") if f.endswith(".wav")]

for voice in voice_list:
    print(f'{root}/data/voice_data/{voice}')
    mfcc = sample_from_mfcc(
        read_mfcc(
            f'{root}/data/voice_data/{voice}',
            SAMPLE_RATE),
        NUM_FRAMES)

    predict = model.m.predict(
        np.expand_dims(
            mfcc,
            axis=0))

    predict_list.append(predict)

    # save predict_list in .pkl file
    with open(f"{root}/data/predict_list.pkl", "wb") as f:
        pickle.dump(predict_list, f)

print(predict_list)

# Compute the cosine similarity and check that it is higher for the same
# for i in range(len(predict_list)):
#     voice_similarity.append(
#         batch_cosine_similarity(
#             predict_list[i],
#             predict_list[j]))

