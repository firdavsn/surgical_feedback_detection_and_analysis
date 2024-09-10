import openai
import torch
from transformers import PreTrainedTokenizer, WhisperForConditionalGeneration
from typing import Union, List
import torchaudio

device = torch.device("cuda")

def whisper_transcribe(audio_path):
    with open(audio_path, "rb") as audio_file:
        transcription = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="en"
        )

    return transcription.text

def local_transcribe(model: WhisperForConditionalGeneration, processor: PreTrainedTokenizer, audio_path: Union[List[str], str]):
    sequences = []
    if isinstance(audio_path, list):
        for path in audio_path:
            wav, sr = torchaudio.load(path)
            wav = wav.mean(dim=0).numpy()
            inputs = processor(wav, return_tensors="pt").to(device)
            input_features = inputs.input_features
            seq = model.generate(inputs=input_features)
            print(seq)
            print(seq.shape)
            sequences.append(seq)
    else:
        wav, sr = torchaudio.load(audio_path)
        wav = wav.mean(dim=0).numpy()
        inputs = processor(wav, return_tensors="pt").to(device)
        input_features = inputs.input_features
        seq = model.generate(inputs=input_features)
        print(seq)
        print(seq.shape)
        sequences.append(seq)
        # sequences = seq
    print(type(sequences), len(sequences))
    transcription = processor.batch_decode(sequences, skip_special_tokens=True)[0]

    return transcription
