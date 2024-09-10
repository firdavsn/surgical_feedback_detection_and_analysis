import os
import pandas as pd
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from tqdm import tqdm
from transformers import set_seed
import wave
import numpy as np
import openai

#### GENERAL
def set_openai_key(key_path):
    with open(key_path, "r") as f:
        key = f.read().strip()
    f.close()
    openai.api_key = key
    os.environ["OPENAI_API_KEY"] = key

def get_all_video_files(directory, extension='.avi'):
    avi_files = []
    
    # Loop through the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                avi_files.append(os.path.join(root, file))

    return avi_files

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

def remove_cases_from_dfs(transcriptions_df, fb_clips_df, no_fb_clips_df, cases):
    for case in cases:
        fb_clips_df = fb_clips_df.copy()[fb_clips_df['case'] != case] if fb_clips_df is not None else None
        no_fb_clips_df = no_fb_clips_df.copy()[no_fb_clips_df['case'] != case] if no_fb_clips_df is not None else None
        transcriptions_df = transcriptions_df.copy()[transcriptions_df['file'].str.contains(f'c{case}_') == False] if transcriptions_df is not None else None
    
    return transcriptions_df, fb_clips_df, no_fb_clips_df

def load_annot_df(annot_dir):
    def textLen(row):
        charLen = len(str(row['Dialogue']))
        wordLen = len(str(row['Dialogue']).split(' '))

        return pd.Series([charLen, wordLen])
    
    print("Loading the dataframe...")
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
                print(f"LFB:{fbi}, Error: {err}, Columns: {tmp_df.columns}")
            except FileNotFoundError as err:
                print(f"LFB:{fbi}, Error: {err}")

            print(f"LFB:{fbi} -> {tmp_df.shape}")
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

#### AUDIO
def extract_audio_from_video(video_path, output_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(output_path, verbose=False, logger=None)
    
def save_all_wavs(fb_clips_df, no_fb_clips_df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    rows = []
    for vid_idx in fb_clips_df.index:
        rows.append(dict(fb_clips_df.loc[vid_idx]))
    for vid_idx in no_fb_clips_df.index:
        rows.append(dict(no_fb_clips_df.loc[vid_idx]))
    
    audio_paths = os.listdir(save_dir)
    audio_paths = [f"{save_dir}/{audio_path}" for audio_path in audio_paths]
    for row in tqdm(rows):
        video_path = row['file']
        audio_path = f"{save_dir}/{video_path.split('/')[-1].replace('.avi', '.wav')}"
        if audio_path in audio_paths:
            print(f"Duplicate audio path: {audio_path}")
            continue
        
        extract_audio_from_video(video_path, audio_path)
        
        audio = AudioSegment.from_wav(audio_path)
        audio = audio.set_frame_rate(16000)
        audio.export(audio_path, format="wav")
        
        audio_paths.append(audio_path)
        print(f"Saved audio: {audio_path}")

def get_waveforms(wav_path, include_channels=['both', 'left', 'right']):
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

def load_wavs_df(wavs_dir, fb_clips_df, no_fb_clips_df):
    fb_clips_paths = fb_clips_df['file'].tolist()
    no_fb_clips_paths = no_fb_clips_df['file'].tolist()
    wav_file = [f"{wavs_dir}/{path.split('/')[-1].replace('.avi', '.wav')}" for path in fb_clips_paths + no_fb_clips_paths]
    case = fb_clips_df['case'].tolist() + no_fb_clips_df['case'].tolist()
    
    data = {
        'file': wav_file,
        'fb_instance': [1] * len(fb_clips_paths) + [0] * len(no_fb_clips_paths),
        'case': case
    }
    wavs_df = pd.DataFrame(data)
    
    return wavs_df

def split_wavs_df(wavs_df, splits={'train': 0.8, 'eval': 0, 'test': 0.2}, test_cases=None, seed=42):
    set_seed(seed)
    wavs_df = wavs_df.sample(frac=1, random_state=seed)
    
    if test_cases:
        train_df = wavs_df[~wavs_df['case'].isin(test_cases)]
        test_df = wavs_df[wavs_df['case'].isin(test_cases)]
        
        return train_df, test_df
    else:
        n = len(wavs_df)
        train_size = int(splits['train'] * n)
        # eval_size = int(splits['eval'] * n)
        
        # train_df = wavs_df.iloc[:train_size]
        # eval_df = wavs_df.iloc[train_size:train_size+eval_size] if eval_size > 0 else None
        # test_df = wavs_df.iloc[train_size+eval_size:]
        train_df = wavs_df.iloc[:train_size]
        test_df = wavs_df.iloc[train_size:]
        
        return train_df, test_df


#### TEXT
def load_transcriptions_df(transcriptions_path):
    transcriptions_df = pd.read_csv(transcriptions_path)
    return transcriptions_df

def split_transcriptions_df(transcriptions_df, splits={'train': 0.8, 'test': 0.2}, test_cases=None, seed=42):
    set_seed(seed)
    transcriptions_df = transcriptions_df.sample(frac=1, random_state=seed)
    
    if test_cases:
        train_df = transcriptions_df[~transcriptions_df['case'].isin(test_cases)]
        test_df = transcriptions_df[transcriptions_df['case'].isin(test_cases)]
        
        return train_df, test_df
    else:
        n = len(transcriptions_df)
        train_size = int(splits['train'] * n)
        # eval_size = int(splits['eval'] * n)
        
        # train_df = transcriptions_df.iloc[:train_size]
        # eval_df = transcriptions_df.iloc[train_size:train_size+eval_size] if eval_size > 0 else None
        # test_df = transcriptions_df.iloc[train_size+eval_size:]
        
        train_df = transcriptions_df.iloc[:train_size]
        test_df = transcriptions_df.iloc[train_size:]
                
        return train_df, test_df

def get_transcriptions_df(clips_df, no_clips_df, audio_clips_dir, transcribe_fn, openai_api_key, verbose=False, existing_transcriptions_df=None):
    openai.api_key = openai_api_key
    
    rows = []
    for vid_idx in clips_df.index:
        rows.append(dict(clips_df.loc[vid_idx]))
    for vid_idx in no_clips_df.index:
        rows.append(dict(no_clips_df.loc[vid_idx]))

    data = []
    i = 0
    iter_rows = rows if verbose else tqdm(rows)
    for row in iter_rows:
        video_path = row['file']
        audio_path = f"{audio_clips_dir}/{video_path.split('/')[-1].replace('.avi', '.wav')}"
        case = row['case']
        fb_instance = 1 if video_path.split('/')[3].startswith('fb_clips') else 0
        
        
        if existing_transcriptions_df and audio_path in existing_transcriptions_df['file'].values:
            transcription = existing_transcriptions_df[existing_transcriptions_df['file'] == audio_path]['whisper_transcription'].values[0]
            
            data.append({'file': audio_path, 'case': row['case'], 'whisper_transcription': transcription, 'fb_instance': fb_instance})
            
            if verbose:
                print(f"{'fb' if fb_instance else 'no_fb'} i={i} / {len(rows)}, path={audio_path}")
                print(f"Existing transcription: \"{transcription}\"")
                print()
            
            i += 1
            continue
        
        if verbose:
            print(f"{'fb' if fb_instance else 'no_fb'} i={i} / {len(rows)}, path={audio_path}")
        
        transcription = None
        try:
            transcription = transcribe_fn(audio_path)
        except Exception as e:
            if verbose:
                print(f"Error: {e}")
        
        if verbose:
            print(f"Whisper Transcription: \"{transcription}\"")
            print()
        
        data.append({'file': audio_path, 'case': case, 'whisper_transcription': transcription, 'fb_instance': fb_instance})
        
        i += 1
    
    transcriptions_df = pd.DataFrame(data, columns=['file', 'case', 'whisper_transcription', 'fb_instance'])
    return transcriptions_df


#### AUDIO + TEXT
def split_transcriptions_wavs_df(transcriptions_df, wavs_df, splits={'train': 0.8, 'eval': 0, 'test': 0.2}, seed=42):
    set_seed(seed)
    transcriptions_df = transcriptions_df.sample(frac=1, random_state=seed)
    
    n = len(transcriptions_df)
    train_size = int(splits['train'] * n)
    eval_size = int(splits['eval'] * n)
    
    train_transcriptions_df = transcriptions_df.iloc[:train_size]
    eval_transcriptions_df = transcriptions_df.iloc[train_size:train_size+eval_size] if eval_size > 0 else None
    test_transcriptions_df = transcriptions_df.iloc[train_size+eval_size:]

    train_wavs_df = wavs_df[wavs_df['wav_file'].isin(train_transcriptions_df['file'])]
    eval_wavs_df = wavs_df[wavs_df['wav_file'].isin(eval_transcriptions_df['file'])] if eval_size > 0 else None
    test_wavs_df = wavs_df[wavs_df['wav_file'].isin(test_transcriptions_df['file'])]
    
    return {
        'train': (train_transcriptions_df, train_wavs_df),
        'eval': (eval_transcriptions_df, eval_wavs_df),
        'test': (test_transcriptions_df, test_wavs_df)
    }