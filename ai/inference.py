from huggingsound import SpeechRecognitionModel
import warnings
from typing import Iterator, Optional
import numpy as np
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_waveforms(pahts: list[str], sampling_rate: Optional[int] = 16000) -> list[np.ndarray]:
    """
    Get waveforms from audio files.

    Parameters:
    ----------
        pahts: list[str]
            paths to audio files

        sampling_rate: Optional[int] = 16000
            sampling rate of waveforms

    Returns:
    ----------
        list[np.ndarray]: waveforms from audio files
    """

    waveforms = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for path in pahts:
            waveform, sr = librosa.load(path, sr=sampling_rate)
            waveforms.append(waveform)

    return waveforms


def inference(audio_paths: list[str]):
    processor = Wav2Vec2Processor.from_pretrained('../models/processor')

    model = Wav2Vec2ForCTC.from_pretrained('../models/model')

    waveforms = get_waveforms(audio_paths)
    inputs = processor(waveforms, sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_sentences = processor.batch_decode(predicted_ids)

    return determine_closest_sentence(predicted_sentences[0])


def determine_closest_sentence(question: str):
    # List of Persian texts
    texts = [
        "درباره کتاب های خود صحبت کنید.",
        "در مورد زندگی خود بگید.",
        "یک نصیحت کوتاه بگویید."
    ]

    # New Persian text to compare
    new_text = question

    # Initialize TfidfVectorizer
    vectorizer = TfidfVectorizer()

    # Combine the texts and the new text for TF-IDF calculation
    combined_texts = texts + [new_text]

    # Fit and transform the texts to a TF-IDF representation
    tfidf_matrix = vectorizer.fit_transform(combined_texts)

    # Calculate cosine similarity between the new text and all other texts
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    # Print the similarities
    for i, similarity in enumerate(cosine_similarities[0]):
        print(f"Similarity with text {i + 1}: {similarity:.4f}")

    # If you want to find the most similar text
    most_similar_index = cosine_similarities[0].argmax()
    print(
        f"The most similar text is: '{texts[most_similar_index]}' with a similarity score of {cosine_similarities[0][most_similar_index]:.4f}")

    if cosine_similarities[0][most_similar_index] > 0.3:
        return {"similar_question": texts[most_similar_index],
                "similarity": cosine_similarities[0][most_similar_index]}

    else:
        return {"similar_question": "",
                "similarity": 0}

# audio_paths = ['zendegi.mp3']
#
# inference(audio_paths)
