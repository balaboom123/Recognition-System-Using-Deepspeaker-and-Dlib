import speech_recognition as sr


def recognize_speech_from_audio(audio_file):
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Load the audio file
    with sr.AudioFile(audio_file) as source:
        # Listen to the audio file and store it in audio_data
        audio_data = recognizer.record(source)
        # Recognize the speech in the audio file
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            print("could not understand the audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results Speech Recognition service; {e}")
            return None


