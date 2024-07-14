import librosa
import numpy as np
from scipy.signal import resample


def preprocess_audio(audio_file):
	# Sample audio signal (this would be your actual audio data)
	audio_data, sr = librosa.load(audio_file, sr=16000)

	# Calculate the average amplitude
	average_amplitude = np.mean(np.abs(audio_data))

	# Retain only segments where the amplitude is greater than the average
	audio_data = audio_data[np.abs(audio_data) > average_amplitude]

	# Compute MFCCs
	mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=64, hop_length=256, fmin=30, fmax=1500)
	# y=signal, sr=SAMPLE_RATE, n_mfcc=n_mfcc, n_mels=64, hop_length=256
	resized_mfccs = resample(mfccs, 160, axis=1)

	# normalized_spectrogram = (resized_mfccs - np.min(resized_mfccs)) / (
	# 				np.max(resized_mfccs) - np.min(resized_mfccs))

	mfccs = np.expand_dims(resized_mfccs.T, axis=-1)

	return mfccs  # To check the shape of the resulting MFCC array


def preprocess_audio_test(audio_file):
	# Load audio file
	audio_data, sr = librosa.load(audio_file, sr=16000)  # resample to 16kHz

	# Calculate the average amplitude
	# average_amplitude = np.mean(np.abs(audio_data))

	# Retain only segments where the amplitude is greater than the average
	# audio_data = audio_data[np.abs(audio_data) > (average_amplitude / 2)]

	# Create mel spectrogram with 64 mel bands
	mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=16000, n_mels=64, hop_length=256, fmin=20,
													 fmax=4000)

	# Convert to decibels (log scale)
	log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
	reshaped_spectrogram = log_mel_spectrogram
	# Resize to have 160 frames
	# resized_spectrogram = resample(log_mel_spectrogram, 160, axis=1)

	# Normalize spectrogram
	# normalized_spectrogram = (resized_spectrogram - np.min(resized_spectrogram)) / (
	# 			np.max(resized_spectrogram) - np.min(resized_spectrogram))

	# Reshape to match model input shape (160 frames, 64 mel bands, 1 channel)
	# reshaped_spectrogram = np.expand_dims(resized_spectrogram.T, axis=-1)

	return reshaped_spectrogram