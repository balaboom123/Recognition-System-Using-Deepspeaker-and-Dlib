import os
import cv2
import numpy as np
import pyaudio
import soundfile as sf  # For saving audio to FLAC file
from scipy import signal
import noisereduce as nr
from tqdm import tqdm
import time


# Function to record audio from microphone
def record_audio(p, duration=1):
	chunk = 1024
	sample_format = pyaudio.paInt16
	channels = 1
	fs = 44100  # Sample rate
	seconds = duration

	p = pyaudio.PyAudio()  # Create an interface to PortAudio

	print('Audio Recording...')

	stream = p.open(
		format=sample_format,
		channels=channels,
		rate=fs,
		frames_per_buffer=chunk,
		input=True
	)

	frames = []  # Initialize array to store frames

	# Store data in chunks for specified seconds
	for i in tqdm(range(0, int(fs / chunk * seconds)), desc="Recording"):
		data = stream.read(chunk)
		frames.append(data)

	# Stop and close the stream
	stream.stop_stream()
	stream.close()

	print('Finished recording.')

	# Convert the recorded data to 16-bit PCM format
	audio_data = b''.join(frames)
	audio_array = np.frombuffer(audio_data, dtype=np.int16)

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


def record_images(cap, num_picture=1):
	# Directory to save the images
	output_dir = 'recorded_images'
	os.makedirs(output_dir, exist_ok=True)

	# Initialize the images path list
	filename = []

	print('Image Capturing...')

	count = 0
	while True:
		ret, frame = cap.read()
		if not ret:
			break

		# Display the frame
		# cv2.imshow('frame', frame)

		# Save frame as an image
		image_path = os.path.join(output_dir, f'image_{count}.jpg')
		cv2.imwrite(image_path, frame)
		filename.append(image_path)

		count += 1
		# Break the loop if duration is exceeded
		if count >= num_picture:
			break

		# cv2.imshow('frame', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	print('Finished Capturing.')

	return filename
