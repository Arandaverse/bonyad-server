import warnings
from typing import Iterator, Optional
import numpy as np
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from sentence_transformers import SentenceTransformer, util
import random


def get_waveforms(paths: list[str], sampling_rate: Optional[int] = 16000) -> list[np.ndarray]:
    waveforms = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for path in paths:
            print(f"Loading file from path: {path}")  # Debug log
            print("before librosa loading")
            waveform, sr = librosa.load(path, sr=sampling_rate)
            waveforms.append(waveform)
    return waveforms


def inference(audio_paths: list[str]):
    processor = Wav2Vec2Processor.from_pretrained('../models/processor')

    model = Wav2Vec2ForCTC.from_pretrained('../models/model')
    print("before wavefroms")
    waveforms = get_waveforms(audio_paths)
    print("after waverforms")
    inputs = processor(waveforms, sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_sentences = processor.batch_decode(predicted_ids)

    return determine_closest_sentence(predicted_sentences[0])


def determine_closest_sentence(question: str):
    # List of Persian texts
    texts = [
        "درباره کتاب های خود بگویید.",
        "درباره زندگی خود بگویید.",
        "یک نصیحت کوتاه بگویید."
    ]

    # New Persian text to compare
    new_text = question
    print("question", new_text)

    # Load a pre-trained model
    model = SentenceTransformer('../models/sentence_transformer')

    # Encode the texts and the new text
    embeddings = model.encode(texts + [new_text])

    # Calculate cosine similarity between the new text and all other texts
    cosine_similarities = util.pytorch_cos_sim(embeddings[-1], embeddings[:-1])

    # Print the similarities
    for i, similarity in enumerate(cosine_similarities[0]):
        print(f"Similarity with text {i + 1}: {similarity.item():.4f}")

    # Find the most similar text
    most_similar_index = cosine_similarities[0].argmax().item()
    most_similar_score = cosine_similarities[0][most_similar_index].item()

    print(
        f"The most similar text is: '{texts[most_similar_index]}' with a similarity score of {most_similar_score:.4f}")

    if most_similar_score > 0.5:
        response_index = 0
        if most_similar_index < 2:
            response_index = most_similar_index + 1
        else:
            response_index = random.randint(3, 5)
        return {"similar_question": texts[most_similar_index],
                "similarity": most_similar_score,
                "response_index": response_index}
    else:
        return {"similar_question": "",
                "similarity": 0,
                "response_index": 0}

# question = "درباره زندگی نامه خود بگویید."
# result = determine_closest_sentence(question)
# print(result)
#
# api_key = AIzaSyBTklF1FPjavaBJL7RO0EW65uZVAccP6hI
