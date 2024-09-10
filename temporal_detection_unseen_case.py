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

def get_metrics(clf_model_class, clf_model_dir, case_id, vad_threshold, from_end_threshold, aux=None, vad_only=False, F1_weighting='weighted'):
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
    
    model = TemporalDetectionModel(params_temporal, device)
    print("Predicting...")
    predictions_df = model.rolling_predict(name, load_saved=True, vad_threshold=vad_threshold, only_vad_filter=vad_only)
    
    print("Getting true labels...")
    true_labels = model.get_true_labels(name, load_saved=True, from_end_threshold=from_end_threshold)
    
    print("Computing metrics...")
    metrics = model.score(F1_weighting=F1_weighting)
    
    return metrics

def main():
    metrics_df = pd.DataFrame(columns=['Case', 'Model Class', 'Model Dir', 'VAD Threshold', 'From End Threshold', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC'])
    i = 0
    for case_id in [1, 2, 18, 9, 10]:
        for vad_threshold in [0, 0.1, 0.3, 0.5]:
            for from_end_threshold in [0, 3, 5]:
                clf_model_class = 'TextModel'
                clf_model_dir = f'results/checkpoints/text/Whiper-BERT remove={case_id} test=None seed=42'
                
                # clf_model_class = 'AudioModel'
                # clf_model_dir = f'results/checkpoints/audio/Wav2Vec2 remove={case_id} test=None seed=42'
                
                # clf_model_class = 'AudioTextFusionModel'
                # clf_model_dir = f'results/checkpoints/multimodal/Whiper-BERT + Wav2Vec2 remove={case_id} test=None seed=42'
                
                print("=========================================")
                print(f"Case ID: {case_id}")
                print(f"Model: {clf_model_class}")
                print(f"Model Dir: {clf_model_dir}")
                print(f"VAD Threshold: {vad_threshold}")
                print(f"From End Threshold: {from_end_threshold}")
                
                metrics = get_metrics(
                    clf_model_class, 
                    clf_model_dir, 
                    case_id, 
                    vad_threshold=vad_threshold, 
                    from_end_threshold=from_end_threshold, 
                    aux='vad_only', 
                    vad_only=True,
                    F1_weighting='weighted'
                )
                print(metrics)
                print("\n\n")
                
                metrics_df.loc[i] = pd.Series({
                    'Case': case_id,
                    'Model Class': clf_model_class,
                    'Model Dir': clf_model_dir,
                    'VAD Threshold': vad_threshold,
                    'From End Threshold': from_end_threshold,
                    'ROC AUC': metrics['roc_auc'],
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1': metrics['f1'],
                })
                i += 1
    
    metrics_df['Model Class'] = 'VAD'
    metrics_df.to_csv('results/rolling_fragments/metrics/vad_F1_weighted.csv', index=False)
    
if __name__ == '__main__':
    main()