import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Function to read and plot a WAV file


def plot_wav(file_path):
    # Load the WAV file
    sample_rate, data = wavfile.read(file_path)

    # Extract the audio data
    if data.ndim == 2:
        # If the audio is stereo, average the channels
        data = np.mean(data, axis=1)

    # Calculate the time array
    time = np.arange(len(data)) / sample_rate

    # Plot the waveform
    plt.plot(time, data, label=os.path.basename(file_path))


# List of WAV files to read and plot
file_paths = ['gorden_no_filter.wav', 'gorden_with_filter.wav',
              'gorden_with_noise_filter.wav', "gorden_input_audio_0.wav"]  # Replace with your WAV file paths

# Plot waveforms for each file
for file_path in file_paths:
    plt.figure(figsize=(20, 4))
    plot_wav(file_path)
    # Customize the plot
    plt.title('Waveforms')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()
