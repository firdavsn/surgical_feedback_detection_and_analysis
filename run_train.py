import os
import shutil
from transformers import set_seed
import torch
import logging
from transformers import logging as transformers_logging
# transformers_logging.set_verbosity_error()
# transformers_logging.set_verbosity(logging.CRITICAL)

from train import train_text_model, train_audio_model, train_multimodal_model
from utils import (
    load_transcriptions_df,
    remove_cases_from_dfs,
    split_transcriptions_df,
    get_all_video_files,
    create_clips_df,
    save_all_wavs,
    load_wavs_df,
    split_wavs_df,
    get_transcriptions_df
)
from transcribe import whisper_transcribe

REMOVE_CASES = []

trainer2cases = {
    'A1': [1, 2, 6, 7, 21, 22, 33, 35],
    'A2': [3, 4, 5, 8, 11, 12, 13, 14, 16, 18, 20, 24, 26, 28, 29, 30, 31, 32],
    'A3': [9, 15, 17, 23, 25, 27],
    'A4': [10, 19, 34]
}

"""
{'A1': {'fb': 1351, 'no_fb': 1351},
 'A2': {'fb': 2079, 'no_fb': 2038},
 'A3': {'fb': 508, 'no_fb': 509},
 'A4': {'fb': 268, 'no_fb': 268}}
"""

TEST_CASES = []
SEED = 42

PRETRAINED_MODEL_NAMES = {
    'text': 'bert-base-uncased',
    'audio': 'superb/wav2vec2-base-superb-er'
    # 'audio': 'facebook/wav2vec2-base-960h',
}

PATHS = {
    'transcriptions_df': "results/transcriptions/whisper_vad_thresh_en.csv",
    'audio_clips_dir': "results/audio_clips",
    'audio_features_dir': "results/features/audio-superb",
    'openai_api_key': "openai_api_key.txt",
    'fb_clips': "../../clips_no_wiggle/fb_clips_no_wiggle",
    'no_fb_clips': "../../clips_no_wiggle/no_fb_clips_vad_thresh",
    'fb_clips_annot': "../../clips_no_wiggle/fbk_cuts_no_wiggle_0_4210.csv",
}

def train(model_type, seed, remove_cases, test_cases, aux=None, use_pretrained_audio_text=False):
    set_seed(seed)
    remove_cases_str = ",".join([str(x) for x in remove_cases]) if remove_cases else 'None'
    test_cases_str = ",".join([str(x) for x in test_cases]) if test_cases else 'None'
    
    # model_type: checkpoint_dir
    checkpoint_dirs = {
        # 'text': f'results/checkpoints/text/Whiper-BERT remove={remove_cases_str} seed={seed}{" aux="+aux if aux is not None else ""}',
        # 'audio': f'results/checkpoints/audio/Wav2Vec2 remove={remove_cases_str} seed={seed}{" aux="+aux if aux is not None else ""}',
        # 'multimodal': f'results/checkpoints/multimodal/Whiper-BERT + Wav2Vec2 remove={remove_cases_str} seed={seed}{" aux="+aux if aux is not None else ""}',
        
        'text': f'results/checkpoints/text/Whiper-BERT remove={remove_cases_str} test={test_cases_str} seed={seed}{" aux="+aux if aux is not None else ""}',
        'audio': f'results/checkpoints/audio/Wav2Vec2 remove={remove_cases_str} test={test_cases_str} seed={seed}{" aux="+aux if aux is not None else ""}',
        'multimodal': f'results/checkpoints/multimodal/Whiper-BERT + Wav2Vec2 {"pretrained" if use_pretrained_audio_text else ""} remove={remove_cases_str} test={test_cases_str} seed={seed}{" aux="+aux if aux is not None else ""}',
    }
    # if os.path.exists(checkpoint_dirs[model_type]):
    #     shutil.rmtree(checkpoint_dirs[model_type])
    os.makedirs(checkpoint_dirs[model_type], exist_ok=True)
    
    if model_type == 'text':
        transcriptions_df = load_transcriptions_df(PATHS['transcriptions_df'])
        transcriptions_df, _, _ = remove_cases_from_dfs(transcriptions_df, None, None, remove_cases)
        train_transcriptions_df, eval_transcriptions_df = split_transcriptions_df(transcriptions_df, splits={'train': 0.8, 'test': 0.2}, test_cases=test_cases, seed=seed)
        
        print(f"len(train_transcriptions_df): {len(train_transcriptions_df)}")
        print(f"len(eval_transcriptions_df): {len(eval_transcriptions_df)}")
        
        print("Train transcriptions case value counts:", train_transcriptions_df['case'].value_counts().to_dict())
        print("Eval transcriptions case value counts:", eval_transcriptions_df['case'].value_counts().to_dict())
        
        model = train_text_model(
            text_model=PRETRAINED_MODEL_NAMES['text'],
            train_transcriptions_df=train_transcriptions_df,
            eval_transcriptions_df=eval_transcriptions_df,
            num_classes=2,
            class_weights=[0.5, 0.5],
            device='cuda',
            output_dir=checkpoint_dirs['text'],
            seed=seed,
            epochs=20,
            batch_size=32,
            wandb_project_name=checkpoint_dirs['text'].split('/')[-1],
            # lr_scheduler_type='reduce_lr_on_plateau',  # linear
            # lr_scheduler_kwargs={'patience': 5}, # {}
            warmup_steps=500,
            weight_decay=0.1,
            save_steps=100,
            eval_steps=100,
            eval_save_strategy='steps',
            metric_for_best_model='eval_roc_auc',
            report_to='wandb'
        )
    elif model_type == 'audio':
        fb_clips = get_all_video_files(PATHS['fb_clips'])
        no_fb_clips = get_all_video_files(PATHS['no_fb_clips'])
        fb_clips_df = create_clips_df(fb_clips)
        no_fb_clips_df = create_clips_df(no_fb_clips)

        _, fb_clips_df, no_fb_clips_df = remove_cases_from_dfs(None, fb_clips_df, no_fb_clips_df, remove_cases)
        
        print(f"len(fb_clips_df): {len(fb_clips_df)}")
        print(f"len(no_fb_clips_df): {len(no_fb_clips_df)}")
        
        wavs_df = load_wavs_df(PATHS['audio_clips_dir'], fb_clips_df, no_fb_clips_df)
        train_wavs_df, eval_wavs_df = split_wavs_df(wavs_df, splits={'train': 0.8, 'test': 0.2}, test_cases=test_cases, seed=seed)
        
        print(f"len(train_wavs_df): {len(train_wavs_df)}")
        print(f"len(eval_wavs_df): {len(eval_wavs_df)}")
        
        print("Train wavs case value counts:", train_wavs_df['case'].value_counts().to_dict())
        print("Eval wavs case value counts:", eval_wavs_df['case'].value_counts().to_dict())
        
        model = train_audio_model(
            audio_model=PRETRAINED_MODEL_NAMES['audio'],
            train_wavs_df=train_wavs_df,
            eval_wavs_df=eval_wavs_df,
            channel='both',
            audio_features_dir=PATHS['audio_features_dir'],
            num_classes=2,
            class_weights=[0.5, 0.5],
            device='cuda',
            output_dir=checkpoint_dirs['audio'],
            seed=seed,
            epochs=20,
            batch_size=12,
            wandb_project_name=checkpoint_dirs['audio'].split('/')[-1],
            # lr_scheduler_type='reduce_lr_on_plateau',  # linear
            # lr_scheduler_kwargs={'patience': 5}, # {}
            warmup_steps=500,
            weight_decay=0.1,
            save_steps=200,
            eval_steps=200,
            eval_save_strategy='steps',
            metric_for_best_model='eval_roc_auc',
            report_to='wandb'
        )
    elif model_type == 'multimodal':
        transcriptions_df = load_transcriptions_df(PATHS['transcriptions_df'])
        transcriptions_df, _, _ = remove_cases_from_dfs(transcriptions_df, None, None, remove_cases)
        train_transcriptions_df, eval_transcriptions_df = split_transcriptions_df(transcriptions_df, splits={'train': 0.8, 'test': 0.2}, test_cases=test_cases, seed=seed)
        
        print(f"len(train_transcriptions_df): {len(train_transcriptions_df)}")
        print(f"len(eval_transcriptions_df): {len(eval_transcriptions_df)}")
        
        print("Train transcriptions case value counts:", train_transcriptions_df['case'].value_counts().to_dict())
        print("Eval transcriptions case value counts:", eval_transcriptions_df['case'].value_counts().to_dict())
        
        bert_pretrained_dir=None
        wav2vec2_pretrained_dir=None
        if use_pretrained_audio_text:
            bert_pretrained_dir = checkpoint_dirs['text']
            wav2vec2_pretrained_dir = checkpoint_dirs['audio']
        
        model = train_multimodal_model(
            text_model=PRETRAINED_MODEL_NAMES['text'],
            audio_model=PRETRAINED_MODEL_NAMES['audio'],
            train_transcriptions_df=train_transcriptions_df,
            eval_transcriptions_df=eval_transcriptions_df,
            num_classes=2,
            num_features=256,
            class_weights=[0.5, 0.5],
            device='cuda',
            output_dir=checkpoint_dirs['multimodal'],
            audio_features_dir=PATHS['audio_features_dir'],
            seed=seed,
            epochs=20,
            batch_size=12,
            wandb_project_name=checkpoint_dirs['multimodal'].split('/')[-1],
            # lr_scheduler_type='reduce_lr_on_plateau',  # linear
            # lr_scheduler_kwargs={'patience': 5}, # {}
            warmup_steps=500,
            weight_decay=0.1,
            save_steps=100,
            eval_steps=100,
            eval_save_strategy='steps',
            metric_for_best_model='eval_roc_auc',
            report_to='wandb',
            bert_pretrained_dir=bert_pretrained_dir,
            wav2vec2_pretrained_dir=wav2vec2_pretrained_dir
        )

def prepare_audio():
    fb_clips = get_all_video_files(PATHS['fb_clips'])
    no_fb_clips = get_all_video_files(PATHS['no_fb_clips'])
    fb_clips_df = create_clips_df(fb_clips)
    no_fb_clips_df = create_clips_df(no_fb_clips)
    
    print(f"len(fb_clips_df): {len(fb_clips_df)}")
    print(f"len(no_fb_clips_df): {len(no_fb_clips_df)}")

    save_all_wavs(fb_clips_df, no_fb_clips_df, PATHS['audio_clips_dir'])

def prepare_transcriptions(verbose=True):
    fb_clips = get_all_video_files(PATHS['fb_clips'])
    no_fb_clips = get_all_video_files(PATHS['no_fb_clips'])
    fb_clips_df = create_clips_df(fb_clips)
    no_fb_clips_df = create_clips_df(no_fb_clips)
    
    print(f"len(fb_clips_df): {len(fb_clips_df)}")
    print(f"len(no_fb_clips_df): {len(no_fb_clips_df)}")
    
    openai_api_key = None
    with open(PATHS['openai_api_key'], 'r') as f:
        openai_api_key = f.read().strip()
    f.close()
    
    transcriptions_df = get_transcriptions_df(
        clips_df=fb_clips_df,
        no_clips_df=no_fb_clips_df,
        audio_clips_dir=PATHS['audio_clips_dir'],
        transcribe_fn=whisper_transcribe,
        openai_api_key=openai_api_key,
        verbose=verbose
    )
    transcriptions_df.to_csv(PATHS['transcriptions_df'], index=False)

if __name__ == "__main__":
    # for remove_cases in ([1], [2], [9], [10], [18]):
    for remove_cases in (trainer2cases['A2'], trainer2cases['A3'], trainer2cases['A4']):
        print("====================================")
        print(f"remove_cases: {remove_cases}")
        torch.cuda.empty_cache()
        # train('text', SEED, remove_cases, TEST_CASES)
        train('multimodal', SEED, remove_cases, TEST_CASES)
        print("\n\n")
        
    # train('text', SEED, remove_cases=[], test_cases=[], aux='test')
    # train('audio', SEED)
    # train('multimodal', SEED)
    # prepare_audio()
    # prepare_transcriptions()

