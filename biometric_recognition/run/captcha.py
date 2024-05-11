from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import os


def random_question(dir):
    speech_list = [os.path.splitext(f) for f in os.listdir(dir) if f.endswith(".wav")]
    speech_list = [name for name, _ in speech_list]
    random_element = random.choice(speech_list)

    return random_element


def calculate_similarity(sentence1, sentence2):
    sentence1, sentence2 = sentence1.lower(), sentence2.lower()
    # Vectorize the sentences
    vectorizer = CountVectorizer().fit_transform([sentence1, sentence2])

    # Calculate cosine similarity
    cos_sim = cosine_similarity(vectorizer)

    # Return the similarity score
    return cos_sim[0][1]

