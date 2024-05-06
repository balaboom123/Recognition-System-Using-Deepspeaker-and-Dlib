import os
import numpy as np
import pyaudio
import soundfile as sf  # For saving audio to FLAC file
from deep_speaker.audio import read_mfcc
from deep_speaker.batcher import sample_from_mfcc
from deep_speaker.constants import SAMPLE_RATE, NUM_FRAMES
from deep_speaker.conv_models import DeepSpeakerModel
from deep_speaker.test import batch_cosine_similarity
from scipy import signal
import noisereduce as nr


def normalize_audio(audio):
    return audio / np.max(np.abs(audio))


# Define a mapping from speaker embeddings to speaker names
speaker_mapping = {
    f"speaker_{str(i).zfill(3)}": f"Speaker{i}" for i in range(1, 1000)
}

# Reproducible results.
np.random.seed(123)

# Define the model here.
deep_speaker = DeepSpeakerModel()

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the checkpoint.
res_cnn = f"{script_dir}/biometric_recognition/model/ResCNN_softmax_pre_training_checkpoint_102.h5"
deep_speaker.m.load_weights(res_cnn, by_name=True)

# Directory containing speaker audio files
speakers_dir = f"{script_dir}/speech_normalize_denoised"


# Function to record audio from microphone
def record_audio(duration=10):
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    fs = 44100  # Sample rate
    seconds = duration

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    print('Recording...')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    # Store data in chunks for specified seconds
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print('Finished recording.')

    audio_data = b''.join(frames)
    audio_array = np.frombuffer(audio_data, dtype=np.int16)

    audio_array = nr.reduce_noise(
        y=audio_array,
        sr=fs)
    audio_array = normalize_audio(audio_array)

    # Estimate the noise spectrum (e.g., from a silent part of the signal)
    # noise_spectrum = np.abs(np.fft.fft(noise_array))

    # Apply Wiener filter
    # audio_array = signal.wiener(audio_array, mysize=len(audio_array), noise=noise_spectrum)

    # Save the recorded data as a FLAC file with a unique filename
    index = 0
    while True:
        filename = f'input_audio_{index}.wav'
        if not os.path.exists(filename):
            break
        index += 1

    with sf.SoundFile(filename, mode='x', samplerate=fs, channels=channels, subtype='PCM_16') as f:
        f.write(audio_array)

    return filename


# Record audio and save to file
input_audio_file = record_audio(duration=10)

# Process the recorded audio
input_mfcc = sample_from_mfcc(
    read_mfcc(
        input_audio_file,
        SAMPLE_RATE),
    NUM_FRAMES)
predicted_embedding = deep_speaker.m.predict(
    np.expand_dims(input_mfcc, axis=0))

# Initialize a flag for match
match_found = False

# Loop through each audio file in the directory
for speaker_file in os.listdir(speakers_dir):
    if speaker_file.endswith(".wav"):
        # Construct the full path
        speaker_path = os.path.join(speakers_dir, speaker_file)

        # Extract speaker ID from the filename
        speaker_id = os.path.splitext(speaker_file)[0]

        # Get the embedding for the current speaker audio
        speaker_embedding = deep_speaker.m.predict(
            np.expand_dims(
                sample_from_mfcc(
                    read_mfcc(
                        speaker_path,
                        SAMPLE_RATE),
                    NUM_FRAMES),
                axis=0))

        # Calculate cosine similarity between the input and current speaker
        similarity = batch_cosine_similarity(
            predicted_embedding, speaker_embedding)

        # Set a threshold for similarity
        similarity_threshold = 0.9  # Adjust as needed

        # Determine if it's a match or not
        print(f'Input audio Speaker {speaker_id} and the Similarity: {similarity}')
        if similarity > similarity_threshold:
            print(
                f'MATCH: Input audio matches Speaker {speaker_id} - Similarity: {similarity}')
            match_found = True
            break

# If no match is found, print "Access Denied"
if not match_found:
    print("Access Denied")
