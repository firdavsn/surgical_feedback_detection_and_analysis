import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import set_seed
from models import TemporalDetectionModel
from transcribe import whisper_transcribe

torch.cuda.empty_cache()

def get_metrics(clf_model_class, clf_model_dir, case_id, vad_threshold, from_end_threshold, aux=None, vad_only=False):
    openai_api_key = None
    if os.path.exists("openai_api_key.txt"):
        with open("openai_api_key.txt", 'r') as f:
            openai_api_key = f.read().strip()
    f.close()

    full_vid_dir = '../../full_downsampled_videos/'
    full_vid_paths = os.listdir(full_vid_dir)
    full_vid_path = [os.path.join(full_vid_dir, x) for x in full_vid_paths if x.split('/')[-1].startswith(f"LFB{case_id}_")][0]
    
    vad_activity_dir = '../../full_VADs/'
    vad_activity_paths = os.listdir(vad_activity_dir)
    vad_activity_path = [os.path.join(vad_activity_dir, x) for x in vad_activity_paths if x.split('/')[-1].startswith(f"LFB{case_id}_")]
    vad_activity_path.sort()
    vad_activity_path = vad_activity_path[0]
    
    params_temporal = {
        'full_vid_path': full_vid_path,
        'case_id': case_id,
        'rolling_shift': 5,
        'rolling_duration': 10,
        'fragments_dir': 'results/rolling_fragments/',
        'clf_model_class': clf_model_class,
        'clf_model_dir': clf_model_dir,
        'vad_activity_path': vad_activity_path,
        'feature_extractor': 'superb/wav2vec2-base-superb-er',
        'tokenizer': 'bert-base-uncased',
        'fb_clips_path': '../../clips_no_wiggle/fb_clips_no_wiggle',
        'console_times_path': '../../annotations/console_times/combined_console_times.csv',
        'annot_dir': '../../annotations',
        'openai_api_key': openai_api_key,
        'transcribe_fn': whisper_transcribe,
    }
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(params_temporal)
    
    name = f'LFB{case_id} model={clf_model_class} vad_threshold={vad_threshold} fe-threshold={from_end_threshold}{" " + aux if aux is not None else ""}.csv'
    print(name)
    model = TemporalDetectionModel(params_temporal, device)
    
    print("Fragmenting full video...")
    model.fragment_full_video()
    
    print("Predicting...")
    predictions_df = model.rolling_predict(name, load_saved=True, vad_threshold=vad_threshold, only_vad_filter=vad_only)
    
    print("Getting true labels...")
    true_labels = model.get_true_labels(name, load_saved=True, from_end_threshold=from_end_threshold)
    
    print("Computing metrics...")
    metrics_dict = {
        'binary': model.score(F1_weighting='binary'),
        'weighted': model.score(F1_weighting='weighted'),
    }
    
    return metrics_dict

trainer2cases = {
    'A1': [1, 2, 6, 7, 21, 22, 33, 35],
    'A2': [3, 4, 5, 8, 12, 13, 14, 16, 18, 20, 24, 26, 29, 30, 31, 32],
    'A3': [9, 15, 17, 23, 25],
    'A4': [10, 19, 34]
}

"""
{'A1': {'fb': 1351, 'no_fb': 1351},
 'A2': {'fb': 2079, 'no_fb': 2038},
 'A3': {'fb': 508, 'no_fb': 509},
 'A4': {'fb': 268, 'no_fb': 268}}
"""

def main():
    # metrics_df = pd.DataFrame(columns=['Case', 'Model Class', 'Model Dir', 'VAD Threshold', 'From End Threshold', 'Metrics Weighting', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC'])
    metrics_df = pd.read_csv('results/rolling_fragments/metrics/unseen_surgeon.csv')
    
    test_models = ['multimodal']
    
    for trainer_id, case_ids in trainer2cases.items():
        vad_threshold = 0
        from_end_threshold = 3
        
        if trainer_id == 'A2':
            case_ids = [3, 4, 5, 8, 11, 12, 13, 14, 16, 18, 20, 24, 26, 28, 29, 30, 31, 32]
        if trainer_id == 'A3':
            case_ids = [9, 15, 17, 23, 25, 27]
        
        remove_cases_str = ",".join([str(x) for x in case_ids]) if case_ids else 'None'

        models = {
            'text': {'model_class': 'TextModel', 'model_dir': f'results/checkpoints/text/Whiper-BERT remove={remove_cases_str} test=None seed=42'},
            'audio': {'model_class': 'AudioModel', 'model_dir': f'results/checkpoints/audio/Wav2Vec2 remove={remove_cases_str} test=None seed=42'},
            'multimodal': {'model_class': 'AudioTextFusionModel', 'model_dir': f'results/checkpoints/multimodal/Whiper-BERT + Wav2Vec2  remove={remove_cases_str} test=None seed=42'},
            'vad': None
        }
        
        for model_type, model_params in models.items():
            if model_type not in test_models:
                continue
            
            if model_type == 'vad':
                model_params = {'model_class': models['text']['model_class'], 'model_dir': models['text']['model_dir']}
            
            clf_model_class = model_params['model_class']
            clf_model_dir = model_params['model_dir']
            
            for case_id in trainer2cases[trainer_id]:
                if case_id > 33:
                    continue
                print("=========================================")
                print(f"Trainer ID, Case ID: {trainer_id}, {case_id}")
                print(f"Model Type: {model_type}")
                print(f"Model: {clf_model_class}")
                print(f"Model Dir: {clf_model_dir}")
                print(f"VAD Threshold: {vad_threshold}")
                print(f"From End Threshold: {from_end_threshold}")
                
                # try: 
                metrics = get_metrics(
                    clf_model_class, 
                    clf_model_dir, 
                    case_id, 
                    vad_threshold=vad_threshold, 
                    from_end_threshold=from_end_threshold, 
                    aux='vad_unseen_surgeon' if model_type == 'vad' else 'unseen_surgeon', 
                    vad_only=True if model_type == 'vad' else False,
                )
                print(metrics)
                
                metrics_df.loc[len(metrics_df)] = pd.Series({
                    'Case': case_id,
                    'Model Class': clf_model_class if model_type != 'vad' else 'VAD',
                    'Model Dir': clf_model_dir,
                    'VAD Threshold': vad_threshold,
                    'From End Threshold': from_end_threshold,
                    'Metrics Weighting': 'binary',
                    'ROC AUC': metrics['binary']['roc_auc'],
                    'Accuracy': metrics['binary']['accuracy'],
                    'Precision': metrics['binary']['precision'],
                    'Recall': metrics['binary']['recall'],
                    'F1': metrics['binary']['f1'],
                })
                metrics_df.loc[len(metrics_df)] = pd.Series({
                    'Case': case_id,
                    'Model Class': clf_model_class if model_type != 'vad' else 'VAD',
                    'Model Dir': clf_model_dir,
                    'VAD Threshold': vad_threshold,
                    'From End Threshold': from_end_threshold,
                    'Metrics Weighting': 'weighted',
                    'ROC AUC': metrics['weighted']['roc_auc'],
                    'Accuracy': metrics['weighted']['accuracy'],
                    'Precision': metrics['weighted']['precision'],
                    'Recall': metrics['weighted']['recall'],
                    'F1': metrics['weighted']['f1'],
                })
                # except Exception as e:
                #     print(f"An error occurred: {e}.")
            
            print("=========================================")
            print("\n\n")
    
        metrics_df.to_csv('results/rolling_fragments/metrics/unseen_surgeon.csv', index=False)
    
if __name__ == '__main__':
    main()