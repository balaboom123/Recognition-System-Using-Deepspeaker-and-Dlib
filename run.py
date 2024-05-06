import cv2
import numpy as np
import os
import dlib
import json
import pickle
import pyaudio
import soundfile as sf

from biometric_recognition.face_feature_extract import landmark
from biometric_recognition.face_feature_extract import face_detect_resize
from biometric_recognition.run.run_init import init_model, init_recorder, init_features, init_question
from biometric_recognition.run.record import record_images, record_audio
from biomeetric_recognition.speech_recog import recognize_speech_from_audio

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
print("name include", landmark_y)

# Initialize the camera and audio
audio, cap = init_recorder()

count = 0
while True:
    # choose command
    command = input("Enter command: 1.login 2.exit")

    if command == "exit":
        break
    elif command == "login":
        pass

    # Record the images
    file_names = record_images(cap, num_picture=1)

    face_cls = None
    for file_name in file_names:
        rect, img, resize_img, face_count = face_detect_resize(file_name, detector)

        if face_count != 1:
            continue

        landmark_feature_68 = landmark(
            img, rect, recognizer, shape_predictor_68, save_img=False)

        euclidean_distance = [
            np.linalg.norm(
                landmark_feature_68 -
                avg_vector) for avg_vector in landmark_x]
        min_index = np.argmin(euclidean_distance)

        face_cls = landmark_y[min_index]

    print("face recognition: ", face_cls)

    speakers_dir = f"{script_dir}/data/person_data/{face_cls}"
    question = init_question(speakers_dir)
    print("question: ", question)

    # Record the audio
    input_audio_file = record_audio(audio, duration=5)
    recognize_speech_from_audio(input_audio_file)
    print("answer: ", input_audio_file)
    
    # Process the recorded audio
    input_mfcc = sample_from_mfcc(read_mfcc(input_audio_file, SAMPLE_RATE), NUM_FRAMES)
    predicted_embedding = deep_speaker.m.predict(np.expand_dims(input_mfcc, axis=0))

    # Initialize a flag for match
    match_found = False

    # Loop through each audio file in the directory

    for speaker_file in os.listdir(speakers_dir):
        # Skip non-WAV files
        if not speaker_file.endswith(".wav"):
            continue

        # Construct the full path
        speaker_path = os.path.join(speakers_dir, speaker_file)

        # Extract speaker ID from the filename
        speaker_id = os.path.splitext(speaker_file)[0]

        # Get the embedding for the current speaker audio
        speaker_embedding = deep_speaker.m.predict(
            np.expand_dims(sample_from_mfcc(read_mfcc(speaker_path, SAMPLE_RATE), NUM_FRAMES), axis=0))

        # Calculate cosine similarity between the input and current speaker
        similarity = batch_cosine_similarity(predicted_embedding, speaker_embedding)

        # Set a threshold for similarity
        similarity_threshold = 0.4  # Adjust as needed

        # Determine if it's a match or not
        if similarity > similarity_threshold:
            print(f'MATCH: Input audio matches Speaker {speaker_id} - Similarity: {similarity}')
            match_found = True
            break

    # If no match is found, print "Access Denied"
    if not match_found:
        print("Access Denied")

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()

# Terminate the PortAudio interface
p.terminate()
