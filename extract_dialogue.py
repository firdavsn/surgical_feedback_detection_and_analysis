from utils import *
import torch
from models import ExtractDialogueModel
from transcribe import whisper_transcribe
from utils import set_openai_key

def init_model(case_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    full_audio_path = f'../../full_audios/LFB{case_id}_full.wav'
    if case_id in [4, 11]:
        full_audio_path = f'../../full_audios/LFB{case_id}_full_1.wav'
    
    params_extract_dialogue = {
        'speaker_diarization_model': 'pyannote/speaker-diarization-3.1',
        'speaker_embedding_model': 'pyannote/embedding',
        'hf_token_path': 'huggingface_token.txt',
        'openai_key_path': 'openai_api_key.txt',
        'transcribe_fn': whisper_transcribe,
        'full_audio_path': full_audio_path,
        'interval': 180,
        'vad_activity_path': f'../../full_VADs/LFB{case_id}_full_activity.csv',
        'diarizations_save_path': f'results/extract_dialogue/diarizations/LFB{case_id}_full.csv',
        'transcriptions_save_path': f'results/extract_dialogue/transcriptions/LFB{case_id}_full.csv',
        'identifications_save_path': f'results/extract_dialogue/identifications/LFB{case_id}_full.csv',
        'audio_clips_dir': 'results/extract_dialogue/audio_clips',
        'trainer_anchors_dir': f'results/extract_dialogue/anchors/LFB{case_id}/trainer',
        'trainee_anchors_dir': f'results/extract_dialogue/anchors/LFB{case_id}/trainee',
        'tmp_dir': 'tmp',
        'seed': 42,
        'min_n_speakers': 2,
        'max_n_speakers': 2,
        'embedding_dist_thresh': 0.8
    }
    openai_key_path = 'openai_api_key.txt'
    set_openai_key(openai_key_path)

    model = ExtractDialogueModel(params_extract_dialogue, device)
    
    return model

def run_diarization(model: ExtractDialogueModel, load_saved):
    model.full_diarization(load_saved=load_saved)

def run_transcription(model: ExtractDialogueModel, load_saved):
    model.full_transcription(load_saved=load_saved)

def main():
    for case_id in range(30, 34):
        print(f'Processing case {case_id}')
        print('Initializing model...')
        model = init_model(case_id)
        print('Running diarization...')
        run_diarization(model, load_saved=True)
        
        print("Deleting models from memory..")
        del model.speaker_diarization_model
        del model.speaker_embedding_model
        
        model.speaker_diarization_model = None
        model.speaker_embedding_model = None
        torch.cuda.empty_cache()
        
        print('Running transcription')
        run_transcription(model, load_saved=False)

if __name__ == '__main__':
    main()