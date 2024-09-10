import torch
from torch import nn
from moviepy.editor import VideoFileClip
import os
import cv2
import pandas as pd
from pydub import AudioSegment
import wave
import numpy as np
from transformers import AutoFeatureExtractor, AutoTokenizer
from tqdm import tqdm
import openai
import json
from safetensors import safe_open
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from .TextModel import TextModel
from .AudioModel import AudioModel
from .AudioTextFusionModel import AudioTextFusionModel

def get_vid_len(vid_path):
    clip = VideoFileClip(vid_path)
    duration = clip.duration
    
    return duration

def get_HMS_from_secs(secs):
    h = secs // (60*60)
    m = (secs - h*(60*60)) // 60
    s = secs - h*(60*60) - m*60

    return h, m, s

def get_fragment_secs(fragment_name):
    h = int(fragment_name.split('_')[-1].split('-')[0])
    m = int(fragment_name.split('_')[-1].split('-')[1])
    s = int(fragment_name.split('_')[-1].split('-')[2].split('.')[0])
    
    return h*60*60 + m*60 + s

def get_all_video_files(directory, extension='.avi'):
    avi_files = []
    
    # Loop through the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                avi_files.append(os.path.join(root, file))

    return avi_files

# Helper functions for converting string wall clock time to video time
def convTime2Secs(ref_time, stime):
    # reference start time
    rtc = ref_time.split(":")
    rh = int(rtc[0])
    rm = int(rtc[1])
    rs = int(rtc[2])

    rtc_sec = rh*60*60+rm*60+rs

    # frame time
    stc = str(stime).split(":")
    sh = int(stc[0])
    sm = int(stc[1])
    ss = int(stc[2])-1

    stc_sec = sh*60*60+sm*60+ss

    return (stc_sec-rtc_sec)

def extract_video_clip(vid_path, dst_path, secs, duration, ext=".mp4"):
    print('-Loading ' + vid_path)
    wnd_len = duration # seconds

    cap = cv2.VideoCapture(vid_path)
    print(f"Is opened?: {cap.isOpened()}")

    print(f"Source video clip: {vid_path}")
    print(f"Destination video clip: {dst_path}")

    clip = VideoFileClip(vid_path) # load sfull surgery video

    subclip = clip.subclip(secs, secs+wnd_len)

    subclip.write_videofile(dst_path, fps=5, codec='libx264', audio=True, audio_fps=100*160, audio_codec="pcm_s16le", verbose=False, logger=None) #, bitrate="8460k" # codec='libvpx')
    print(f"Video clip saved at {dst_path}")
    print()

    return 0

def extract_audio(video_path, audio_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path, verbose=False, logger=None)
    
    audio = AudioSegment.from_wav(audio_path)
    audio = audio.set_frame_rate(16000)
    audio.export(audio_path, format="wav")
    
    return 1
    
def extract_audio_features(audio_path, features_path, feature_extractor):
    def get_waveforms(wav_path, include_channels=['both', 'left', 'right']):
        temp_audio_path = 'temp_wavs/temp_audio.wav'
        
        # Extract waveforms from wav File
        wav_file = wave.open(wav_path, 'rb')

        both_channels = np.frombuffer(wav_file.readframes(-1), dtype=np.int16)
        left_channel = both_channels[0::2]
        right_channel = both_channels[1::2]
        
        waveforms = {}
        
        for channel in include_channels:
            if channel == 'both':
                waveforms['both'] = both_channels
            elif channel == 'left':
                waveforms['left'] = left_channel
            elif channel == 'right':
                waveforms['right'] = right_channel
        
        sampling_rate = wav_file.getframerate()
        
        return waveforms, sampling_rate
    
    channel = 'both'
    waveforms, sampling_rate = get_waveforms(audio_path, include_channels=[channel])
    
    features = feature_extractor(waveforms[channel], sampling_rate=16000, return_tensors='pt')
    torch.save(features, features_path)
    
    return features

def extract_transcription(audio_path, transcription_path, transcribe_fn):    
    transcription = transcribe_fn(audio_path)
    
    file_ = open(transcription_path, 'w')
    file_.write(transcription)
    file_.close()
    
    return transcription

def extract_transcription_features(transcription, features_path, tokenizer):
    features = tokenizer(transcription, return_tensors='pt')
    torch.save(features, features_path)
    
    return features

def is_trainee_giving_fb(console_times, case_id, fragment_secs, rolling_duration=10):
    console_times_tmp = console_times.copy()
    console_times_tmp = console_times_tmp[console_times_tmp['is_eligible_time'] == 0]
    console_times_tmp.reset_index(drop=True, inplace=True)
    for i in range(len(console_times_tmp)):
        interval_start = console_times_tmp.loc[i, 'On time (secs)']
        interval_end = console_times_tmp.loc[i, 'Off time (secs)']
        
        if interval_start < fragment_secs and interval_end > fragment_secs:
            return True
        
    return False   

def load_vad_activity_df(vad_path):
    vad_activity = pd.read_csv(vad_path)
    vad_activity['time_2'] = None
    for i in range(len(vad_activity)):
        m, s = vad_activity.loc[i, 'time'].split(':')
        secs = int(m)*60 + int(s)
        h, m, s = get_HMS_from_secs(secs)
        vad_activity.loc[i, 'time_2'] = pd.Timestamp(hour=h, minute=m, second=s, year=1970, month=1, day=1)
    # vad_activity['time_2'] = pd.to_datetime(vad_activity['time_2'], format='%H:%M:%S')
    vad_activity = vad_activity[['time_2', 'activity']]
    vad_activity.columns = ['time', 'activity']
    return vad_activity

def get_best_checkpoint_dir(checkpoint_dir):
    checkpoints = os.listdir(checkpoint_dir)
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))
    last_checkpoint = checkpoints[-1]
    last_trainer_state = json.load(open(f'{checkpoint_dir}/{last_checkpoint}/trainer_state.json'))
    best_checkpoint_dir = last_trainer_state['best_model_checkpoint']
    
    return best_checkpoint_dir

def load_clf_model(model_dir, model_class, device, params_temporal):
    params_model = {
        "num_classes": 2,
        "class_weights": None,
        "num_features": 256,
        "text_model": params_temporal['tokenizer'],
        "audio_model": params_temporal['feature_extractor'],
    }
    
    if model_class == "AudioModel" or model_class == AudioModel:
        model = AudioModel(params_model, device)
    elif model_class == "TextModel" or model_class == TextModel:
        model = TextModel(params_model, device)
    elif model_class == "AudioTextFusionModel" or model_class == AudioTextFusionModel:
        params_model['audio_features_dir'] = None
        model = AudioTextFusionModel(params_model, device)
    
    best_checkpoint_dir = get_best_checkpoint_dir(model_dir)

    print(f"Loading model from {best_checkpoint_dir}")
    with safe_open(f'{best_checkpoint_dir}/model.safetensors', framework='pt', device=device) as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    
    try:
        del state_dict['loss_func.weight']
        
        if isinstance(model, AudioTextFusionModel):
            del state_dict['audio_model.loss_func.weight'] 
            del state_dict['text_model.loss_func.weight'] 
    except Exception as e:
        print(f"Error: {e}")
    
    model.load_state_dict(state_dict)
    
    return model

def is_fb(fb_clips_df, case_id, f_secs, rolling_duration=10, format_fb_clips_df=True, threshold=0):
    if format_fb_clips_df or 'secs' not in fb_clips_df.columns:
        fb_clips_df = fb_clips_df.copy()
        fb_clips_df = fb_clips_df[fb_clips_df['case'] == case_id]
        fb_clips_df['secs'] = fb_clips_df['time'].apply(lambda x: 3600*int(x.split(':')[0]) + 60*int(x.split(':')[1]) + int(x.split(':')[2]))
    
    # fb_clips_df = fb_clips_df.copy()[((fb_clips_df['secs'] >= f_secs) & (fb_clips_df['secs'] < f_secs + rolling_duration - threshold)) |
                                    #  ((fb_clips_df['secs'] + rolling_duration >= f_secs) & (fb_clips_df['secs'] + rolling_duration < f_secs + rolling_duration - threshold))]
    fb_clips_df = fb_clips_df.copy()[(fb_clips_df['secs'] >= f_secs) & (fb_clips_df['secs'] < f_secs + rolling_duration - threshold)]
    
    return len(fb_clips_df) > 0

def extract_info_from_filename(filename):
    base_name = os.path.basename(filename)
    parts = base_name.split('_')
    case = int(parts[0][1:])
    idx = int(parts[1][1:])
    time = parts[2].replace('-', ':').split('.')[0]
    return case, idx, time

def create_clips_df(file_list):
    data = []
    for file in file_list:
        case, idx, time = extract_info_from_filename(file)
        data.append({'file': file, 'case': case, 'idx': idx, 'time': time})
    
    clips_df = pd.DataFrame(data, columns=['file', 'case', 'idx', 'time'])
    return clips_df

def load_fb_clips_df(fb_clips_path):
    fb_clips = get_all_video_files(fb_clips_path, extension='.avi')
    fb_clips_df = create_clips_df(fb_clips)
    return fb_clips_df

def load_annot_df(annot_dir, verbose=False):
    def textLen(row):
        charLen = len(str(row['Dialogue']))
        wordLen = len(str(row['Dialogue']).split(' '))

        return pd.Series([charLen, wordLen])
    
    print("Loading the dataframe...") if verbose else None
    fb_df = pd.DataFrame()

    for fbi in list(range(1,34)):
        filepath = os.path.join(annot_dir, f"LFB{fbi}_ALL.xlsx")
        if os.path.exists(filepath):
            tmp_df = pd.read_excel(filepath, index_col=None, header=1)
            try:
                tmp_df = tmp_df[~tmp_df['Dialogue'].isna()].copy()
                tmp_df['Case'] = fbi
                tmp_df['Dialogue'] = tmp_df['Dialogue'].astype('str')
                tmp_df[['charLen','wordLen']] = tmp_df.apply(textLen, axis=1)
                tmp_df['Timestamp'] = tmp_df['Timestamp'].apply(lambda x: x.strftime('%H:%M:%S'))
            except KeyError as err:
                print(f"LFB:{fbi}, Error: {err}, Columns: {tmp_df.columns}") if verbose else None
            except FileNotFoundError as err:
                print(f"LFB:{fbi}, Error: {err}") if verbose else None

            print(f"LFB:{fbi} -> {tmp_df.shape}") if verbose else None
            #display(tmp_df.head(1))

            # rename columns
            tmp_df = tmp_df.rename(columns={"Error of omission": "t_omission",
                                  "Error (Provoked)": "t_comission",
                                  "Error (Cautionary)": "t_warning",
                                  "Resident action (positive)": "t_good_action",
                                  "Responding to resident question": "t_question",
                                  "Responding to resident question/statement": "t_question",
                                  "Responding to resident": "t_question",
                                  "Positive Reinforcement": "f_praise",
                                  "Positive Punishment": "f_criticism",
                                  "Anatomic": "f_anatomic",
                                  "Procedural": "f_proecdural",
                                  "Technique": "f_technical",
                                  "Technical": "f_technical",
                                  "Visual aid": "f_visual_aid",
                                  "Other": "f_other",
                                  "Trainee acknowledges feedback (VERBAL)": "r_t_verb",
                                  "Trainee acknowledges feedback (BEHAVIORAL ADJUSTMENT)": "r_t_beh",
                                  "Trainee asked for Clarification": "r_t_clarify",
                                  "Mentor affirms behavioral change \"ok\", \"yup\", \"uh-huh\"": "r_m_approve",
                                  "Mentor denies behavioral change \"no\", \"try-again\", \"not quite\"": "r_m_disapprove",
                                  "Mentor Repeated FB\n(identical)": "r_m_rep_identical",
                                  "Mentor Repeated FB (similar)": "r_m_rep_similar",
                                  "Mentor takes over premptively (as precaution)": "r_m_control_safety",
                                  "Mentor takes over premptively (as precaution, safety)": "r_m_control_safety",
                                  "Mentor takes over (for teaching)": "r_m_control_other"})
            
            tmp_df = tmp_df.drop(columns=["If \"other\", specify"], errors="ignore")

            fb_df = pd.concat([fb_df, tmp_df], ignore_index=True)

    return fb_df

def compute_metrics(evals, F1_weighting):
    labels = evals['labels']
    preds = evals['preds']
    
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=F1_weighting)
    roc_auc = roc_auc_score(labels, preds)
    
    return {'accuracy': accuracy,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'f1': f1}

class TemporalDetectionOutput:
    def __init__(self, pred, prob, transcription, video_fragment_path, audio_fragment_path, vad_filtered=False):
        self.pred = pred
        self.prob = prob
        self.transcription = transcription
        self.video_fragment_path = video_fragment_path
        self.audio_fragment_path = audio_fragment_path
        self.vad_filtered = vad_filtered
    
    def __str__(self):
        return str({
            'pred': self.pred, 
            'prob': self.prob, 
            'transcription': self.transcription, 
            'video_fragment_path': self.video_fragment_path,
            'audio_fragment_path': self.audio_fragment_path,
            'vad_filtered': self.vad_filtered})

    def __repr__(self):
        return str({
            'pred': self.pred, 
            'prob': self.prob, 
            'transcription': self.transcription, 
            'video_fragment_path': self.video_fragment_path,
            'audio_fragment_path': self.audio_fragment_path,
            'vad_filtered': self.vad_filtered})

class TemporalDetectionModel:
    def __init__(self, params_temporal, device):
        # OpenAI API
        openai.api_key = params_temporal['openai_api_key']
        
        # DataFrames
        self.console_times_df = pd.read_csv(params_temporal['console_times_path'], index_col='Unnamed: 0')
        self.fb_clips_df = load_fb_clips_df(params_temporal['fb_clips_path'])
        self.vad_activity = load_vad_activity_df(params_temporal["vad_activity_path"])
        self.fb_annot_df = load_annot_df(params_temporal['annot_dir'])
        
        # Constants
        self.full_vid_path = params_temporal["full_vid_path"]
        self.case_id = params_temporal["case_id"]
        self.rolling_shift = params_temporal["rolling_shift"]
        self.rolling_duration = params_temporal["rolling_duration"]
        self.fragments_dir = params_temporal["fragments_dir"]
        self.device = device
        self.trascribe_fn = params_temporal['transcribe_fn']
        
        # Load model
        clf_model_class = params_temporal["clf_model_class"]
        clf_model_dir = params_temporal["clf_model_dir"]
        self.clf_model = load_clf_model(clf_model_dir, clf_model_class, device, params_temporal)
        
        # Load feature extractor and tokenizer
        self.feature_extractor = None
        self.tokenizer = None
        if isinstance(self.clf_model, AudioModel) or isinstance(self.clf_model, AudioTextFusionModel):
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(params_temporal['feature_extractor'])
        
        if isinstance(self.clf_model, TextModel) or isinstance(self.clf_model, AudioTextFusionModel):
            self.tokenizer = AutoTokenizer.from_pretrained(params_temporal['tokenizer'])
            
        # Predictions
        self.true_labels_df = pd.DataFrame(columns=['secs', 'duration', 'fb_instance'])
        self.predictions_df = pd.DataFrame(columns=['case', 'start_time', 'duration', 'pred', 'prob', 'transcription', 'vad_filtered', 'video_fragment_path', 'audio_fragment_path', 'trainee_on_console'])
        self.predictions_df.index.name = 'fragment_id'

    def fragment_full_video(self, overwrite=False):
        dst_dir = f'{self.fragments_dir}/video/LFB{self.case_id}/'
        if os.path.exists(dst_dir) and not overwrite:
            return 0
        os.makedirs(dst_dir, exist_ok=True)
        
        full_duration = get_vid_len(self.full_vid_path)
        
        # Make it next value multiple of rolling_duration
        full_duration = int(full_duration + (self.rolling_duration - full_duration % self.rolling_duration))
        
        for i, sec in enumerate(range(0, full_duration, self.rolling_shift)):
            h, m, s = get_HMS_from_secs(sec)
            dst_path = os.path.join(dst_dir, f'v_c{self.case_id}_{h}-{m}-{s}.avi')
            if not os.path.exists(dst_path) or overwrite:
                try:
                    extract_video_clip(self.full_vid_path, dst_path, secs=sec, duration=self.rolling_duration, ext='.avi')
                except Exception as e:
                    print(f"Error: {e}")
                    return 0
        return 1

    def vad_filter(self, start_time, duration, threshold):
        vad_activity = self.vad_activity[(self.vad_activity['time'] >= start_time) & (self.vad_activity['time'] <= start_time + pd.Timedelta(seconds=duration))]
        vad_activity = vad_activity[vad_activity['activity'] >= threshold]
        return len(vad_activity) > 0
    
    def _predict(self, start_time, vad_threshold, only_vad_filter=False):
        start_time_str = start_time.strftime('%H-%M-%S')
        
        # Delete leading 0s
        start_time_lst = start_time_str.split('-')
        for i in range(3):
            start_time_lst[i] = str(int(start_time_lst[i]))
        start_time_str = '-'.join(start_time_lst)
        
        video_path = f'{self.fragments_dir}/video/LFB{self.case_id}/v_c{self.case_id}_{start_time_str}.avi'
        
        # Extract audio 
        audio_path = f'{self.fragments_dir}/audio/LFB{self.case_id}/a_c{self.case_id}_{start_time_str}.wav'
        os.makedirs(f'{self.fragments_dir}/audio/LFB{self.case_id}/', exist_ok=True)
        if not os.path.exists(audio_path):
            extract_audio(video_path, audio_path)
            
        # VAD filter
        passed_vad_filter = self.vad_filter(
            start_time=pd.Timestamp(hour=start_time.hour, minute=start_time.minute, second=start_time.second, year=1970, month=1, day=1), 
            duration=10,
            threshold=vad_threshold)
        if not passed_vad_filter:
            return TemporalDetectionOutput(pred=0, prob=0, transcription='', video_fragment_path=video_path, audio_fragment_path=audio_path, vad_filtered=True)
        elif only_vad_filter:
            return TemporalDetectionOutput(pred=1, prob=1, transcription='', video_fragment_path=video_path, audio_fragment_path=audio_path, vad_filtered=True)
        
        # Extract audio features
        if self.feature_extractor:
            audio_features_path = f'{self.fragments_dir}/audio_feature/LFB{self.case_id}/af_c{self.case_id}_{start_time_str}.pt'
            os.makedirs(f'{self.fragments_dir}/audio_feature/LFB{self.case_id}/', exist_ok=True)
            if not os.path.exists(audio_features_path):
                audio_features = extract_audio_features(audio_path, audio_features_path, self.feature_extractor)
            else:
                audio_features = torch.load(audio_features_path)

        # Extract transcription and its tokens
        transcription_path = f'{self.fragments_dir}/text/LFB{self.case_id}/t_c{self.case_id}_{start_time_str}.txt'
        os.makedirs(f'{self.fragments_dir}/text/LFB{self.case_id}/', exist_ok=True)
        if not os.path.exists(transcription_path):
            transcription = extract_transcription(audio_path, transcription_path, self.trascribe_fn)
        else:
            with open(transcription_path, 'r') as f:
                transcription = f.read()
            f.close()
        
        # Tokenize transcription
        if self.tokenizer:
            transcription_features_path = f'{self.fragments_dir}/text_feature/LFB{self.case_id}/tf_c{self.case_id}_{start_time_str}.pt'
            os.makedirs(f'{self.fragments_dir}/text_feature/LFB{self.case_id}/', exist_ok=True)
            if not os.path.exists(transcription_features_path):
                transcription_features = extract_transcription_features(transcription, transcription_features_path, self.tokenizer)
            else:
                transcription_features = torch.load(transcription_features_path)
        
        # Predict
        audio_inputs = None
        text_inputs = None
        if isinstance(self.clf_model, AudioModel) or isinstance(self.clf_model, AudioTextFusionModel):
            audio_inputs = {'input_values': audio_features['input_values'].float(),
                            'attention_mask': audio_features['attention_mask']}
        
        if isinstance(self.clf_model, TextModel) or isinstance(self.clf_model, AudioTextFusionModel):
            text_inputs = {'input_ids': transcription_features['input_ids'],
                            'attention_mask': transcription_features['attention_mask'],
                            'token_type_ids': transcription_features['token_type_ids']}


        if isinstance(self.clf_model, AudioModel):
            outputs = self.clf_model.forward(
                labels=None, 
                input_values=audio_inputs['input_values'], 
                attention_mask=audio_inputs['attention_mask'])
            logits = outputs.logits
            pred_prob = torch.softmax(logits, dim=-1)[0][1].item()
            pred = torch.argmax(logits, dim=-1).item()
        elif isinstance(self.clf_model, TextModel):
            outputs = self.clf_model(
                labels=None, 
                input_ids=text_inputs['input_ids'], 
                attention_mask=text_inputs['attention_mask'], 
                token_type_ids=text_inputs['token_type_ids'],
            )
            logits = outputs.logits
            pred_prob = torch.softmax(logits, dim=-1)[0][1].item()
            pred = torch.argmax(logits, dim=-1).item()
        elif isinstance(self.clf_model, AudioTextFusionModel):
            outputs = self.clf_model(
                text_input_ids=text_inputs['input_ids'],
                text_attention_mask=text_inputs['attention_mask'],
                text_token_type_ids=text_inputs['token_type_ids'],
                
                audio_input_values=audio_inputs['input_values'],
                audio_attention_mask=audio_inputs['attention_mask'],
                
                labels=None
            )
            logits = outputs.logits
            pred_prob = torch.softmax(logits, dim=-1)[0][1].item()
            pred = torch.argmax(logits, dim=-1).item()

        pred_output = TemporalDetectionOutput(pred=pred, prob=pred_prob, transcription=transcription, video_fragment_path=video_path, audio_fragment_path=audio_path, vad_filtered=False)
        
        return pred_output
    
    def rolling_predict(self, predictions_name=None, load_saved=True, vad_threshold=0.3, only_vad_filter=False):
        save_dir = os.path.join(self.fragments_dir, f'predictions/LFB{self.case_id}/')
        os.makedirs(save_dir, exist_ok=True)
        name = f'LFB{self.case_id}_predictions.csv' if predictions_name is None else predictions_name
        save_path = os.path.join(save_dir, name)
        print(f"Save path: {save_path}")
        if load_saved and os.path.exists(save_path):
            self.predictions_df = pd.read_csv(save_path, index_col='fragment_id')
            if len(self.predictions_df) > 0:
                return self.predictions_df
        
        console_times_df = self.console_times_df.copy()[self.console_times_df['case_id'] == self.case_id]
        console_times_df.reset_index(drop=True, inplace=True)
        predictions_df = self.predictions_df.copy()
        fb_annot_df = self.fb_annot_df.copy()
        fb_annot_df = fb_annot_df[fb_annot_df['Case'] == self.case_id]
        
        case_start_timestamp = min(
            fb_annot_df[fb_annot_df['Case'] == self.case_id].iloc[0]['Timestamp'],
            console_times_df['On time (hh:mm:ss)'].iloc[0]
        )
        
        console_times_df['On time (secs)'] = None
        console_times_df['Off time (secs)'] = None
        for i in range(len(console_times_df)):
            console_times_df.loc[i, 'On time (secs)'] = convTime2Secs(case_start_timestamp, console_times_df.loc[i, 'On time (hh:mm:ss)'])
            console_times_df.loc[i, 'Off time (secs)'] = convTime2Secs(case_start_timestamp, console_times_df.loc[i, 'Off time (hh:mm:ss)'])
        
        
        fragment_files = get_all_video_files(os.path.join(self.fragments_dir, f'video/LFB{self.case_id}/'), extension='.avi')
        for fragment_file in tqdm(fragment_files):
            f_secs = get_fragment_secs(fragment_file)
            h, m, s = get_HMS_from_secs(f_secs)
            
            fragment_id = fragment_file.split('/')[-1].split('.')[0].replace('v_', '')
            
            trainee_giving_fb = is_trainee_giving_fb(console_times_df, self.case_id, f_secs, self.rolling_duration)
            if trainee_giving_fb:
                predictions_df.loc[fragment_id] = {
                    'case': self.case_id, 
                    'start_time': pd.Timestamp(hour=h, minute=m, second=s, year=1970, month=1, day=1), 
                    'duration': self.rolling_duration,
                    'pred': None, 
                    'prob': None, 
                    'transcription': None,
                    'vad_filtered': None,
                    'video_fragment_path': None,
                    'audio_fragment_path': None,
                    'trainee_on_console': True}
                
                continue
            
            pred_output = self._predict(start_time=pd.Timestamp(f'{h}:{m}:{s}'), vad_threshold=vad_threshold, only_vad_filter=only_vad_filter)
            if only_vad_filter and not pred_output.vad_filtered:
                pred_output.pred = 1
                pred_output.prob = 1
                
            predictions_df.loc[fragment_id] = {
                'case': self.case_id, 
                'start_time': pd.Timestamp(hour=h, minute=m, second=s, year=1970, month=1, day=1), 
                'duration': self.rolling_duration,
                'pred': pred_output.pred, 
                'prob': pred_output.prob, 
                'transcription': pred_output.transcription,
                'vad_filtered': pred_output.vad_filtered,
                'video_fragment_path': pred_output.video_fragment_path,
                'audio_fragment_path': pred_output.audio_fragment_path,
                'trainee_on_console': False}
            
            # print(fragment_file, pred_output)

        predictions_df.sort_values(by='start_time', inplace=True)
        predictions_df['start_time'] = predictions_df['start_time'].apply(lambda x: x.strftime('%H:%M:%S'))
        
        predictions_df.to_csv(save_path, index=True)
        
        self.predictions_df = predictions_df
        
        return predictions_df
    
    def get_true_labels(self, name=None, load_saved=True, from_end_threshold=0):
        save_dir = f"{self.fragments_dir}/true_labels/"
        os.makedirs(save_dir, exist_ok=True)
        if name is None:
            path = os.path.join(save_dir, f"LFB{self.case_id}_true_labels.csv")
        else:
            path = os.path.join(save_dir, name)
        if load_saved and os.path.exists(path):
            self.true_labels_df = pd.read_csv(path, index_col='Unnamed: 0')
            return self.true_labels_df
        
        console_times_df = self.console_times_df.copy()[self.console_times_df['case_id'] == self.case_id]
        console_times_df.reset_index(drop=True, inplace=True)
        fb_annot_df = self.fb_annot_df.copy()
        fb_annot_df = fb_annot_df[fb_annot_df['Case'] == self.case_id]
        
        case_start_timestamp = min(
            fb_annot_df[fb_annot_df['Case'] == self.case_id].iloc[0]['Timestamp'],
            console_times_df['On time (hh:mm:ss)'].iloc[0]
        )
        
        console_times_df['On time (secs)'] = None
        console_times_df['Off time (secs)'] = None
        for i in range(len(console_times_df)):
            console_times_df.loc[i, 'On time (secs)'] = convTime2Secs(case_start_timestamp, console_times_df.loc[i, 'On time (hh:mm:ss)'])
            console_times_df.loc[i, 'Off time (secs)'] = convTime2Secs(case_start_timestamp, console_times_df.loc[i, 'Off time (hh:mm:ss)'])

        full_duration = get_vid_len(self.full_vid_path)
        full_duration = int(full_duration + (self.rolling_duration - full_duration % self.rolling_duration))
        
        fb_clips_df = self.fb_clips_df.copy()
        fb_clips_df = fb_clips_df[fb_clips_df['case'] == self.case_id]
        fb_clips_df['secs'] = fb_clips_df['time'].apply(lambda x: 3600*int(x.split(':')[0]) + 60*int(x.split(':')[1]) + int(x.split(':')[2]))
        
        true_labels_df = self.true_labels_df.copy()
        for f_secs in tqdm(range(0, full_duration, self.rolling_shift)):
            h, m, s = get_HMS_from_secs(f_secs)
            
            trainee_giving_fb = is_trainee_giving_fb(console_times_df, self.case_id, f_secs, rolling_duration=self.rolling_duration)
            if trainee_giving_fb:
                true_labels_df.loc[len(true_labels_df)] = {'secs': f_secs, 'duration': self.rolling_duration, 'fb_instance': None}
                continue
            
            fb = is_fb(fb_clips_df, self.case_id, f_secs, rolling_duration=self.rolling_duration, format_fb_clips_df=False, threshold=from_end_threshold)
            if fb:
                true_labels_df.loc[len(true_labels_df)] = {'secs': f_secs, 'duration': self.rolling_duration, 'fb_instance': 1}
            else:
                true_labels_df.loc[len(true_labels_df)] = {'secs': f_secs, 'duration': self.rolling_duration, 'fb_instance': 0}
        
        true_labels_df.to_csv(path, index=True)    
        self.true_labels_df = true_labels_df
        
        return self.true_labels_df
            
    def score(self, F1_weighting='weighted'):
        if len(self.predictions_df) == 0 or len(self.true_labels_df) == 0:
            raise ValueError("Predictions and true labels are not available. Run rolling_predict() and get_true_labels() methods first.")
        
        predictions_df = self.predictions_df.copy()
        true_labels = self.true_labels_df['fb_instance'].values[:len(predictions_df)]
        predictions_df['true_labels'] = true_labels
        predictions_df = predictions_df[~predictions_df['true_labels'].isna()]
        metrics = compute_metrics(
            evals={'labels': list(predictions_df['true_labels'].values), 'preds': list(predictions_df['pred'].values)},
            F1_weighting=F1_weighting
        )
        
        return metrics