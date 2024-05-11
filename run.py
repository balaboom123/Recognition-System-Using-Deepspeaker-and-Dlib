import time

import cv2
import numpy as np
import os
import dlib
import json
import pickle
import pyaudio
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from rich.console import Console
from rich.panel import Panel

from biometric_recognition.face_feature_extract import landmark
from biometric_recognition.face_feature_extract import face_detect_resize
from biometric_recognition.run.run_init import init_model, init_recorder, init_features
from biometric_recognition.run.captcha import random_question, calculate_similarity
from biometric_recognition.run.record import record_images, record_audio
from biometric_recognition.run.preprocessing import preprocess_audio
from biometric_recognition.speech_recog import recognize_speech_from_audio

from deep_speaker.audio import read_mfcc
from deep_speaker.batcher import sample_from_mfcc
from deep_speaker.constants import SAMPLE_RATE, NUM_FRAMES
from deep_speaker.conv_models import DeepSpeakerModel
from deep_speaker.test import batch_cosine_similarity

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the model
recognizer_model = f"{script_dir}/biometric_recognition/model/dlib_face_recognition_resnet_model_v1.dat"
sp68 = f"{script_dir}/biometric_recognition/model/shape_predictor_68_face_landmarks.dat"
res_cnn = f"{script_dir}/biometric_recognition/model/ResCNN_softmax_pre_training_checkpoint_102.h5"
recognizer, shape_predictor_68, detector, deep_speaker = init_model(
    rec_path=recognizer_model,
    sp_path=sp68,
    ds_path=res_cnn
)

# Load the features
person_root = f"{script_dir}/data/person_data"
# person_root = f"{script_dir}/data/temp"
landmark_x, landmark_y = init_features(person_root="data/person_data")

# Initialize the camera and audio
audio, cap = init_recorder()
console = Console()

count = 0
while True:
    # choose command
    command = input("Enter command: 1.login 2.exit\n")

    if command == "exit":
        break
    elif command == "login":
        pass
    else:
        print("Invalid command")
        continue

    # Record the images
    file_names = record_images(cap, num_picture=1)

    face_cls = None
    file_name = file_names[0]
    
    rect, img, resize_img, face_count = face_detect_resize(file_name, detector)

    if face_count != 1:
        panel = Panel("Face Not Detected. Please Try Again!", title="Access Denied")
        console.print(panel)
        continue

    landmark_feature_68 = landmark(
        img, rect, recognizer, shape_predictor_68, save_img=True)

    euclidean_distance = [
        np.linalg.norm(
            landmark_feature_68 -
            avg_vector) for avg_vector in landmark_x]
    min_index = np.argmin(euclidean_distance)

    face_cls = landmark_y[min_index]

    # Show the image
    img = mpimg.imread('face_shape_output.jpg')
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    if face_cls is None:
        panel = Panel("Face Not Recognized", title="Access Denied")
        console.print(panel)
        continue
    else:
        print("Face Recognized: ", face_cls)
        print("- - - - - - - - - - - - - - - - - - -")

    speakers_dir = f"{script_dir}/data/person_data/{face_cls}"
    question = random_question(speakers_dir)
    print("Speak The Sentence below After Showing Recording")
    print("The Sentence: ", question)
    print("- - - - - - - - - - - - - - - - - - -")

    time.sleep(2)
    # Record the audio
    input_audio_file = record_audio(audio, duration=7)
    input_audio_text = recognize_speech_from_audio(input_audio_file)
    print("Your Answer: ", input_audio_text)

    if input_audio_text is None:
        panel = Panel("Voice Not Recognized", title="Access Denied")
        console.print(panel)
        continue
    else:
        match = calculate_similarity(question, input_audio_text)

    if match > 0.7:
        print("Match")
        print("- - - - - - - - - - - - - - - - - - -")
    else:
        panel = Panel("Not Match", title="Access Denied")
        console.print(panel)
        continue

    # Process the recorded audio
    preprocessed_audio = preprocess_audio(input_audio_file)
    predicted_embedding = deep_speaker.m.predict(np.expand_dims(preprocessed_audio, axis=0))

    # Initialize variables for highest similarity
    best_similarity = 0.0
    best_speaker_id = None

    # Loop through each audio file in the directory
    for speaker_file in os.listdir(speakers_dir):
        # Skip non-WAV files
        if not speaker_file.endswith(".wav"):
            continue

        # Construct the full path
        speaker_path = os.path.join(speakers_dir, speaker_file)

        # Extract speaker ID from the filename
        # speaker_id = os.path.splitext(speaker_file)[0]

        # Get the embedding for the current speaker audio
        speaker_audio = preprocess_audio(speaker_path)
        speaker_embedding = deep_speaker.m.predict(np.expand_dims(speaker_audio, axis=0))

        # Calculate cosine similarity between the input and current speaker
        similarity = batch_cosine_similarity(predicted_embedding, speaker_embedding)

        # Determine if it's a match or not
        if similarity > best_similarity:
            best_similarity = similarity
            # best_speaker_id = speaker_id

    # Set a threshold for similarity
    similarity_threshold = 0.995  # Adjust as needed

    # Determine if it's a match or not
    if best_similarity > similarity_threshold:
        panel = Panel(f'Speaker, {face_cls}, match. Similarity: {best_similarity}', title="Access Granted")
        console.print(panel)

    else:
        panel = Panel("Speaker not match", title="Access Denied")
        console.print(panel)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()

# Terminate the PyAudio interface
audio.terminate()
