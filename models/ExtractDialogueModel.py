from pyannote.audio import Pipeline
from pyannote.audio import Model
from pyannote.audio import Inference
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
import pandas as pd
import wave
from scipy.io import wavfile
import torchaudio
from tqdm import tqdm
import os
import torch
import pickle
from transformers import set_seed
import numpy as np
import torchaudio.transforms as T
from openai import OpenAI
import ast
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score

from utils import set_openai_key

TRAINER2CASES = {
    'A1': [1, 2, 6, 7, 21, 22, 33, 35],
    'A2': [3, 4, 5, 8, 11, 12, 13, 14, 16, 18, 20, 24, 26, 28, 29, 30, 31, 32],
    'A3': [9, 15, 17, 23, 25, 27],
    'A4': [10, 19, 34]
}
IDENTIFIABLE_CASES = [1, 2, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 28, 29, 33]

def get_hf_token(token_path):
    with open(token_path, "r") as f:
        token = f.read()
    f.close()
    return token

def get_HMS_from_secs(secs):
    h = secs // (60*60)
    m = (secs - h*(60*60)) // 60
    s = secs - h*(60*60) - m*60

    return h, m, s

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

def get_audio_duration(audio_path):
    with wave.open(audio_path, "rb") as wave_file:
        framerate = wave_file.getframerate()
        frames = wave_file.getnframes()
        duration = frames / float(framerate)
    return duration

def load_console_times(console_times_path):
    console_times = pd.read_csv(console_times_path, index_col='Unnamed: 0')
    return console_times

def load_fb_annot(fb_annot_path):
    fb_annot = pd.read_csv(fb_annot_path, index_col='Unnamed: 0')
    return fb_annot

def get_attending_annots(console_times, case_id):
    console_times = console_times[console_times['case_id'] == case_id]
    df = console_times[['On time (secs)', 'Off time (secs)', 'Mentor ID', 'Trainee ID']]
    df.reset_index(drop=True, inplace=True)
    prev_m, prev_t = None, None
    for i in range(len(df)):
        if df.loc[i, 'Mentor ID'] in [np.nan, 'NaN']:
            df.loc[i, 'Mentor ID'] = prev_m
        else:
            prev_m = df.loc[i, 'Mentor ID']
            
        if df.loc[i, 'Trainee ID'] in [np.nan, 'NaN']:
            df.loc[i, 'Trainee ID'] = prev_t
        else:
            prev_t = df.loc[i, 'Trainee ID']
        
    return df


def get_rag_annotations(annotations_dir):    # os.path.join(rag_embeddings_dir, 'annotations'))
    all_annotations = {}
    for file in sorted(os.listdir(annotations_dir), key=lambda x: int(x.split('_')[0][3:])):
        case_id = int(file.split('_')[0][3:])
        if file.endswith('.csv'):
            all_annotations[case_id] = pd.read_csv(os.path.join(annotations_dir, file), index_col=0)
            all_annotations[case_id].replace({'True': True, 'False': False}, inplace=True)

    return all_annotations

def rag_sample_unseen_fb(available_annotations, fb_k=None, no_fb_k=None):
    df = pd.concat(available_annotations.values(), ignore_index=True)
    
    fb_k = fb_k if fb_k is not None else len(df[df['fb_instance'] == True])
    no_fb_k = no_fb_k if no_fb_k is not None else len(df[df['fb_instance'] == False])    
    if fb_k > len(df[df['fb_instance'] == True]) or no_fb_k > len(df[df['fb_instance'] == False]):
        raise ValueError("k is greater than the number of available instances")
    
    fb_df = df[df['fb_instance'] == True]
    no_fb_df = df[df['fb_instance'] == False]
    
    fb_sample = fb_df.sample(fb_k)
    no_fb_sample = no_fb_df.sample(no_fb_k)
    
    return fb_sample, no_fb_sample

def rag_most_similar_fb(embedding, model: SentenceTransformer, fb_sample, no_fb_sample, num_examples=3):
    fb_sample = fb_sample.copy()
    no_fb_sample = no_fb_sample.copy()
    
    fb_embeddings = np.array([np.load(path) for path in fb_sample['embedding_path']])
    no_fb_embeddings = np.array([np.load(path) for path in no_fb_sample['embedding_path']])
    
    fb_sample['similarity'] = model.similarity(embedding, fb_embeddings).T
    no_fb_sample['similarity'] = model.similarity(embedding, no_fb_embeddings).T
    
    fb_sample = fb_sample.sort_values('similarity', ascending=False).head(num_examples)
    no_fb_sample = no_fb_sample.sort_values('similarity', ascending=False).head(num_examples)
    
    return fb_sample, no_fb_sample        

def rag_sample_examples_unseen_case_fb(all_annotations, case_id, embedding, model: SentenceTransformer, num_examples=3, fb_k=None, no_fb_k=None):
    available_annotations = {c: annotations for c, annotations in all_annotations.items() if c in IDENTIFIABLE_CASES and c != case_id}
    fb_sample, no_fb_sample = rag_sample_unseen_fb(available_annotations, fb_k, no_fb_k)

    return rag_most_similar_fb(embedding, model, fb_sample, no_fb_sample, num_examples)

def rag_sample_examples_unseen_surgeon_fb(all_annotations, case_id, embedding, model: SentenceTransformer, num_examples=3, fb_k=None, no_fb_k=None):
    surgeon_id = None
    for trainer, cases in TRAINER2CASES.items():
        if case_id in cases:
            surgeon_id = trainer
            break
        
    available_annotations = {c: annotations for c, annotations in all_annotations.items() if c in IDENTIFIABLE_CASES and c not in TRAINER2CASES[surgeon_id]}

    fb_sample, no_fb_sample = rag_sample_unseen_fb(available_annotations, fb_k, no_fb_k)
    
    return rag_most_similar_fb(embedding, model, fb_sample, no_fb_sample, num_examples)

def rag_sample_unseen_component(available_annotations, f_anatomic_k=None, f_procedural_k=None, f_technical_k=None):
    df = pd.concat(available_annotations.values(), ignore_index=True)
    
    f_anatomic_df = df[df['f_anatomic'] == True]
    f_procedural_df = df[df['f_procedural'] == True]
    f_technical_df = df[df['f_technical'] == True]
    
    f_anatomic_k = f_anatomic_k if f_anatomic_k is not None else len(f_anatomic_df)
    f_procedural_k = f_procedural_k if f_procedural_k is not None else len(f_procedural_df)
    f_technical_k = f_technical_k if f_technical_k is not None else len(f_technical_df)

    f_anatomic_sample = f_anatomic_df.sample(f_anatomic_k)
    f_procedural_sample = f_procedural_df.sample(f_procedural_k)
    f_technical_sample = f_technical_df.sample(f_technical_k)
    
    return f_anatomic_sample, f_procedural_sample, f_technical_sample

def rag_most_similar_component(embedding, model: SentenceTransformer, f_anatomic_sample, f_procedural_sample, f_technical_sample, num_examples=3):
    f_anatomic_sample = f_anatomic_sample.copy()
    f_procedural_sample = f_procedural_sample.copy()
    f_technical_sample = f_technical_sample.copy()
    
    f_anatomic_embeddings = np.array([np.load(path) for path in f_anatomic_sample['embedding_path']])
    f_procedural_embeddings = np.array([np.load(path) for path in f_procedural_sample['embedding_path']])
    f_technical_embeddings = np.array([np.load(path) for path in f_technical_sample['embedding_path']])
    
    f_anatomic_sample['similarity'] = model.similarity(embedding, f_anatomic_embeddings).T
    f_procedural_sample['similarity'] = model.similarity(embedding, f_procedural_embeddings).T
    f_technical_sample['similarity'] = model.similarity(embedding, f_technical_embeddings).T
    
    f_anatomic_sample = f_anatomic_sample.sort_values('similarity', ascending=False).head(num_examples)
    f_procedural_sample = f_procedural_sample.sort_values('similarity', ascending=False).head(num_examples)
    f_technical_sample = f_technical_sample.sort_values('similarity', ascending=False).head(num_examples)
    
    return f_anatomic_sample, f_procedural_sample, f_technical_sample   

def rag_sample_examples_unseen_case_component(all_annotations, case_id, embedding, model: SentenceTransformer, num_examples=3, f_anatomic_k=None, f_procedural_k=None, f_technical_k=None):
    available_annotations = {c: annotations for c, annotations in all_annotations.items() if c in IDENTIFIABLE_CASES and c != case_id}
    f_anatomic_sample, f_procedural_sample, f_technical_sample = rag_sample_unseen_component(available_annotations)

    return rag_most_similar_component(embedding, model, f_anatomic_sample, f_procedural_sample, f_technical_sample, num_examples)
    
class ExtractDialogueModel:
    def __init__(self, params_extract_dialogue, device):
        # Load models
        self.speaker_diarization_model = Pipeline.from_pretrained(
            params_extract_dialogue['speaker_diarization_model'],
            use_auth_token=get_hf_token(params_extract_dialogue['hf_token_path'])
        ).to(device)
        
        self.speaker_embedding_model = Model.from_pretrained(
            params_extract_dialogue['speaker_embedding_model'],
            use_auth_token=get_hf_token(params_extract_dialogue['hf_token_path'])
        ).to(device)
        self.speaker_embedding_inference = Inference(self.speaker_embedding_model, window="whole")
        
        sentence_transformer_name = params_extract_dialogue['sentence_transformer_name'] if 'sentence_transformer_name' in params_extract_dialogue else 'all-MiniLM-L6-v2'
        self.sentence_transformer_model = SentenceTransformer(sentence_transformer_name)
        
        # Set openai key
        set_openai_key(params_extract_dialogue['openai_key_path'])
        
        # Get parameters
        self.full_audio_path = params_extract_dialogue['full_audio_path']
        if not self.full_audio_path.endswith('.wav'):
            return ValueError("The full_audio_path should be a .wav file")
        self.transcribe_fn = params_extract_dialogue['transcribe_fn']
        self.interval = int(params_extract_dialogue['interval'])     # secs
        self.vad_activity = load_vad_activity_df(params_extract_dialogue["vad_activity_path"])
        self.tmp_dir = params_extract_dialogue['tmp_dir']
        self.audio_clips_dir = params_extract_dialogue['audio_clips_dir']
        self.seed = params_extract_dialogue['seed']
        self.min_n_speakers = params_extract_dialogue['min_n_speakers']
        self.max_n_speakers = params_extract_dialogue['max_n_speakers']
        self.embedding_dist_thresh = params_extract_dialogue['embedding_dist_thresh']
        self.console_times = load_console_times(params_extract_dialogue['console_times_path'])
        self.fb_annot = load_fb_annot(params_extract_dialogue['fb_annot_path'])
        # self.rag_embeddings_dir = params_extract_dialogue['rag_embeddings_dir']
        
        # Determine case_id
        self.case_id = int(self.full_audio_path.split('/')[-1].split('_')[0].replace('LFB', ''))
        
        # Get times when trainer/trainee swap with selves/others
        self.attending_annots = get_attending_annots(self.console_times, self.case_id)
        
        # Get save paths
        self.diarizations_save_path = params_extract_dialogue['diarizations_save_path']
        self.transcriptions_save_path = params_extract_dialogue['transcriptions_save_path']
        self.identifications_save_path = params_extract_dialogue['identifications_save_path']
        self.fb_detection_save_path = params_extract_dialogue['fb_detection_save_path'] 
        self.aligned_fb_detection_save_path = params_extract_dialogue['aligned_fb_detection_save_path'] if 'aligned_fb_detection_save_path' in params_extract_dialogue else self.fb_detection_save_path.replace('fb_detection', 'aligned_fb_detection') 
        self.behavior_prediction_save_path = params_extract_dialogue['behavior_prediction_save_path'] if 'behavior_prediction_save_path' in params_extract_dialogue else self.fb_detection_save_path.replace('aligned_fb_detection', 'behavior_prediction')
        self.component_classification_save_path = params_extract_dialogue['component_classification_save_path'] if 'component_classification_save_path' in params_extract_dialogue else self.fb_detection_save_path.replace('behavior_prediction', 'component_classification')
        
        # Make dirs
        os.makedirs(os.path.dirname(self.diarizations_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.transcriptions_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.identifications_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.fb_detection_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.aligned_fb_detection_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.behavior_prediction_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.component_classification_save_path), exist_ok=True)
        os.makedirs(self.tmp_dir, exist_ok=True)
        os.makedirs(self.audio_clips_dir, exist_ok=True)
        os.makedirs(params_extract_dialogue['trainer_anchors_dir'], exist_ok=True)
        os.makedirs(params_extract_dialogue['trainee_anchors_dir'], exist_ok=True)
        
        # Set seed
        set_seed(params_extract_dialogue['seed'])
        
        # Get anchor embeddings
        self.trainer_anchor_embeddings = {}
        self.trainee_anchor_embeddings = {}
        unique_trainer_ids = self.attending_annots['Mentor ID'].unique()
        unique_trainee_ids = self.attending_annots['Trainee ID'].unique()
        for trainer_id in unique_trainer_ids:
            if ',' not in trainer_id:
                ids = [trainer_id.strip()]
            else:
                ids = [x.strip() for x in trainer_id.split(',')]
                
            for id_ in ids:
                dir_ = os.path.join(params_extract_dialogue['trainer_anchors_dir'], f"{id_}")
                anchor_paths = [os.path.join(dir_, f) for f in os.listdir(dir_)]
                self.trainer_anchor_embeddings[id_] = [self.speaker_embedding_inference(anchor_path) for anchor_path in anchor_paths]
        for trainee_id in unique_trainee_ids:
            if ',' not in trainee_id:
                ids = [trainee_id.strip()]
            else:
                ids = [x.strip() for x in trainee_id.split(',')]
            
            for id_ in ids:
                dir_ = os.path.join(params_extract_dialogue['trainee_anchors_dir'], f"{id_}")
                anchor_paths = [os.path.join(dir_, f) for f in os.listdir(dir_)]
                self.trainee_anchor_embeddings[id_] = [self.speaker_embedding_inference(anchor_path) for anchor_path in anchor_paths]
        
        # Initialize dataframe vars as Nones
        self.diarizations = None
        self.transcriptions = None
        self.identifications = None
        self.fb_detection = None
        self.aligned_fb_detection = None
        self.behavior_prediction = None
        self.component_classification = None
        
        # Load and resample the full audio
        full_wav, orig_sample_rate = torchaudio.load(self.full_audio_path)
        resampler = T.Resample(orig_freq=orig_sample_rate, new_freq=16000)
        self.full_wav = resampler(full_wav)
        
        # Load RAG annotations
        self.all_annotations_fb = None
        self.all_annotations_component = None
        if 'rag_embeddings_dir' in params_extract_dialogue:
            self.all_annotations_fb = get_rag_annotations(os.path.join(params_extract_dialogue['rag_embeddings_dir'], 'annotations_fb'))
            self.all_annotations_component = get_rag_annotations(os.path.join(params_extract_dialogue['rag_embeddings_dir'], 'annotations_component'))
        
    def _speaker_diarization(self, waveform, sample_rate, save_path):
        output = self.speaker_diarization_model({'waveform': waveform, 'sample_rate': sample_rate}, 
                                                min_speakers=self.min_n_speakers, max_speakers=self.max_n_speakers)
        segments = list(output.itersegments())
        
        pickle.dump(output, open(save_path, 'wb'))
        
        diarization = pd.DataFrame(columns=['start', 'end', 'speaker', 'primary_channel'])
        for segment in segments:
            speakers = output.get_labels(segment)
            for speaker in speakers:
                start = segment.start
                end = segment.end
                
                # Determine the primary channel
                wav = waveform[:, int(start*sample_rate) : int(end*sample_rate)].clone()
                channel = 'both'
                try:
                    energy = torch.sum((wav / torch.max(torch.abs(wav), dim=1).values[:, None])**2, dim=1)
                    if energy[0] / energy[1] > 1.5:
                        channel = 'left'
                    elif energy[1] / energy[0] > 1.5:
                        channel = 'right'
                except IndexError:
                    continue
                
                diarization.loc[len(diarization)] = [start, end, speaker, channel]

        return diarization
        
    def full_diarization(self, load_saved=True):
        if load_saved and os.path.exists(self.diarizations_save_path):
            self.diarization = pd.read_csv(self.diarizations_save_path)
            return self.diarization
        
        waveform, sample_rate = torchaudio.load(self.full_audio_path)
        waveforms = []
        
        full_duration = int(get_audio_duration(self.full_audio_path))
        if self.interval is not None and full_duration > self.interval:
            for i in range(0, full_duration, self.interval):
                start = i
                end = i + self.interval
                waveforms.append([waveform[:, start*sample_rate : end*sample_rate], start, end])
        else:
            waveforms.append((waveform, 0, full_duration))
        
        self.diarizations = pd.DataFrame(columns=['start', 'end', 'speaker', 'primary_channel'])
        for waveform in tqdm(waveforms):
            wav, start, end = waveform
            
            filename = os.path.basename(self.full_audio_path).replace('.wav', f'_{int(start)}_{int(end)}.wav')
            pkl_save_path = os.path.join(self.tmp_dir, filename.replace('.wav', '.pkl'))
            
            diarization = self._speaker_diarization(wav, sample_rate, pkl_save_path)
            diarization['start'] += start
            diarization['end'] += start
            
            self.diarizations = pd.concat([self.diarizations, diarization], ignore_index=True)
            
        self.diarizations.to_csv(self.diarizations_save_path, index=False)
        return self.diarizations
    
    def _transcribe(self, clip_save_path):
        try:
            transcription = self.transcribe_fn(clip_save_path)
        except Exception as e:
            print(f"Error in transcribing {clip_save_path}: {e}")
            transcription = ''
        
        return transcription
    
    def full_transcription(self, load_saved=True):
        if load_saved and os.path.exists(self.transcriptions_save_path):
            self.transcriptions = pd.read_csv(self.transcriptions_save_path)
            return self.transcriptions
        
        if self.diarization is None:
            self.full_diarization()
        
        sample_rate = 16000
        self.transcriptions = pd.DataFrame(columns=['start', 'end', 'clip_path', 'sd_speaker', 'transcription'])
        for i in tqdm(range(len(self.diarization))):
            start = self.diarization.loc[i, 'start']
            end = self.diarization.loc[i, 'end']
            sd_speaker = self.diarization.loc[i, 'speaker']
            
            frame_offset = int(start*sample_rate)
            min_duration = 0.3
            num_frames = int(max((end-start)*sample_rate, min_duration * sample_rate))
            wav = self.full_wav[:, frame_offset : frame_offset + num_frames]
            
            clip_filename = os.path.basename(self.full_audio_path).replace('.wav', f'_{int(start)}_{int(end)}.wav')
            clip_save_path = os.path.join(self.audio_clips_dir, clip_filename)
            wavfile.write(clip_save_path, sample_rate, wav.numpy().T)
            
            transcription = self._transcribe(clip_save_path)
            if transcription and transcription[0] == transcription[-1] == '"':
                transcription = transcription[1:-1]
            
            self.transcriptions.loc[len(self.transcriptions)] = [start, end, clip_save_path, sd_speaker, transcription]
            
        self.transcriptions.to_csv(self.transcriptions_save_path, index=False)
        return self.transcriptions
    
    def _is_trainer_or_trainee(self, clip_path, start):
        # Get the speaker embeddings
        clip_embedding = self.speaker_embedding_inference(clip_path)
        
        annots_before_start = self.attending_annots[self.attending_annots['On time (secs)'] <= start]
        
        if len(annots_before_start) == 0:
            trainer_id = self.attending_annots.loc[0, 'Mentor ID'].strip()
            trainee_id = self.attending_annots.loc[0, 'Trainee ID'].strip()
        else:
            trainer_id = annots_before_start.iloc[-1]['Mentor ID'].strip()
            trainee_id = annots_before_start.iloc[-1]['Trainee ID'].strip()
        
        # print(f"Strt: {start}, Trainer ID: {trainer_id}, Trainee ID: {trainee_id}")
        
        # Calculate the average distances
        trainer_embeddings = None
        if ',' not in trainer_id:
            trainer_embeddings = self.trainer_anchor_embeddings[trainer_id]
        else:
            trainer_ids = [x.strip() for x in trainer_id.split(',')]
            trainer_embeddings = []
            for id_ in trainer_ids:
                trainer_embeddings.extend(self.trainer_anchor_embeddings[id_])
        trainer_avg_dist = np.mean([cdist(np.array([clip_embedding]), np.array([anchor_embedding]), metric='cosine') for anchor_embedding in trainer_embeddings])
        
        trainee_embeddings = None
        if ',' not in trainee_id:
            trainee_embeddings = self.trainee_anchor_embeddings[trainee_id]
        else:
            trainee_ids = [x.strip() for x in trainee_id.split(',')]
            trainee_embeddings = []
            for id_ in trainee_ids:
                trainee_embeddings.extend(self.trainee_anchor_embeddings[id_])
        trainee_avg_dist = np.mean([cdist(np.array([clip_embedding]), np.array([anchor_embedding]), metric='cosine') for anchor_embedding in trainee_embeddings])
        
        if trainer_avg_dist is None or trainee_avg_dist is None:
            return 'unknown', None, None
        
        se_speaker = None
        if trainer_avg_dist < trainee_avg_dist and trainer_avg_dist < self.embedding_dist_thresh:
            se_speaker = 'trainer'
        elif trainee_avg_dist < trainer_avg_dist and trainee_avg_dist < self.embedding_dist_thresh:
            se_speaker = 'trainee'
        else:
            se_speaker = 'unknown'
        
        trainer_avg_dist = round(trainer_avg_dist, 3)
        trainee_avg_dist = round(trainee_avg_dist, 3)
        
        return se_speaker, trainer_avg_dist, trainee_avg_dist
    
    def full_identification(self, load_saved=True):
        def apply_similarity_thresh(df, similarity_thresh):
            dist_thresh = 1 - similarity_thresh
            df = df.copy()
            for i in range(len(df)):
                trainer_avg_dist = df.loc[i, 'trainer_dist']
                trainee_avg_dist = df.loc[i, 'trainee_dist']
                se_speaker = None
                if trainer_avg_dist < trainee_avg_dist and trainer_avg_dist < dist_thresh:
                    se_speaker = 'trainer'
                elif trainee_avg_dist < trainer_avg_dist and trainee_avg_dist < dist_thresh:
                    se_speaker = 'trainee'
                else:
                    se_speaker = 'unknown'
                df.loc[i, 'speaker'] = se_speaker
            return df
        
        if load_saved and os.path.exists(self.identifications_save_path):
            self.identifications = pd.read_csv(self.identifications_save_path)
            self.identifications = apply_similarity_thresh(self.identifications, self.embedding_dist_thresh)
            return self.identifications
        
        self.identifications = pd.DataFrame(columns=['start', 'end', 'clip_path', 'sd_speaker', 'transcription', 'vad', 'trainer_dist', 'trainee_dist', 'se_speaker'])
        for i in tqdm(range(len(self.transcriptions))):
            start = self.transcriptions.loc[i, 'start']
            end = self.transcriptions.loc[i, 'end']
            clip_path = self.transcriptions.loc[i, 'clip_path']
            sd_speaker = self.transcriptions.loc[i, 'sd_speaker']
            transcription = self.transcriptions.loc[i, 'transcription']
            vad = None
            
            se_speaker, trainer_dist, trainee_dist = self._is_trainer_or_trainee(clip_path, start)
            
            self.identifications.loc[len(self.identifications)] = [start, end, clip_path, sd_speaker, transcription, vad, trainer_dist, trainee_dist, se_speaker]
        
        self.identifications.to_csv(self.identifications_save_path, index=False)
        return self.identifications
    
    def detect_feedback(self, client: OpenAI, phrase, context, verbose=False, rag=False):
        context = context.reset_index(drop=True)
        
        phrase_str = f"{phrase.loc['start_hms']}: ['{phrase.loc['se_speaker']}': '{phrase.loc['transcription']}']"
        
        context_str = ""
        for i in range(len(context)):
            context_str += f"{context.loc[i, 'start_hms']}: ['{context.loc[i, 'se_speaker']}': '{context.loc[i, 'transcription']}']\n"
        
        # Get RAG examples
        if rag:
            context.replace(np.nan, '', inplace=True)
            context.replace('NaN', '', inplace=True)
            context_dialogue = [" ".join(context['transcription'])]
            embedding = self.sentence_transformer_model.encode(context_dialogue)[0]
            fb_examples, no_fb_examples = rag_sample_examples_unseen_case_fb(self.all_annotations_fb, self.case_id, embedding, self.sentence_transformer_model)
            
            fb_examples_str = "\n".join(fb_examples['context_dialogue'].values)
            no_fb_example_str = "\n".join(no_fb_examples['context_dialogue'].values)
            
            add_on_str = f"""\n\nExamples of feedback that are similar to the given context:
{fb_examples_str}

Examples of no feedback that are similar to the given context:
{no_fb_example_str}"""
        
        prompt = f"""\
Classify whether the following phrase contains the delivery of feedback considering the given context of the last couple turns in the dialogue where the phrase is the last entry in the context.

Format your response as follows. DO NOT DO ANY OTHER FORMATTING.:
{{'feedback': 'yes'}} if the dialogue contains feedback
{{'feedback': 'no'}} if the dialogue does not contain feedback

Context:
{context_str}

Phrase:
{phrase_str}

For example:
{{'feedback': 'yes'}}
"""
        if rag:
            prompt += add_on_str

        print(prompt) if verbose else None
        
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"""You are a binary classifier that determines whether a given phrase contains delivery of feedback from a trainer to a trainee where the trainee is conducting urology surgery using the da Vinci robot. The dialogue is between two speakers, a trainer and a trainee. There are multiple turns in the dialogue where the same speaker can go back to back because a piece of dialogue from the other speaker might not have been picked up or because the other speaker didn't speak as much (usually the trainer speaks more than the trainee). There can be 6 types of feedback:
                 
1. Anatomic: familiarity with anatomic structures and landmarks. i.e. 'Stay in the correct plane, between the 2 fascial layers.'
2. Procedura: pertains to timing and sequence of surgical steps. i.e. 'You can switch to the left side now.'
3. Technical: performnace of a discrete task with appropriate knowledge of factors including exposure, instruments, and traction. i.e. 'Buzz it.'
4. Praise: a positive remark. i.e. 'Good job.'
5. Criticism: a negative remark. i.e. 'It should never be like this.' """},
                {"role": "user", "content": prompt}
            ],
            seed=self.seed,
        )

        content = completion.choices[0].message.content

        print(content) if verbose else None
        
        try:
            classification = ast.literal_eval(content)
        except Exception as e:
            print(e)
            print(content)
            
            if 'yes' in content:
                classification = {'feedback': 'yes'}
            else:
                classification = {'feedback': 'no'}
        
        print(classification) if verbose else None
        
        return classification
    
    
    def full_fb_detection(self, context_len=5, load_saved=True, verbose=False, aux=None, rag=False):
        if load_saved and os.path.exists(self.fb_detection_save_path):
            self.fb_detection = pd.read_csv(self.fb_detection_save_path)
            return self.fb_detection

        self.fb_detection = pd.DataFrame(columns=['full_clip_path', 'context_dialogue', 'context_times', 'phrase', 'pred_fb_instance'])
        client = OpenAI()
        
        # Add hms columns and remove unknown speakers from identifications
        def format_hms(str):
            if str == 0:
                return '00'
            if float(str) < 10:
                return f'0{str}'
            return str
        
        identifications = self.identifications.copy()
        if aux in [None, 'reduced hallucinations', 'temporal context']:
            identifications = identifications[identifications['se_speaker'] != 'unknown']
        identifications.reset_index(drop=True, inplace=True)
        
        # Get chunks of dialogues as contexts
        contexts_ids = []
        if aux != 'temporal context':
            for i in range(len(identifications)):
                if i < context_len:
                    continue
                
                se_speaker = identifications.loc[i, 'se_speaker']
                
                if aux in [None, 'reduced hallucinations'] and se_speaker in ['trainer', 'trainee']:
                    contexts_ids.append(list(range(i-context_len, i+1)))
                elif aux == 'dialogue':
                    contexts_ids.append(list(range(i-context_len, i+1)))
        else:
            for i in range(len(identifications)):
                if identifications.loc[i, 'start'] < context_len:
                    continue
                    
                context_ids = []
                start = identifications.loc[i, 'start']
                for j in range(i, -1, -1):
                    if start - identifications.loc[j, 'start'] < context_len:
                        context_ids.append(j)
                    else:
                        break
                contexts_ids.append(context_ids[::-1])
        
        print()
            
        identifications['start_hms'] = identifications['start'].apply(lambda x: f"{format_hms(round(x//3600))}:{format_hms(round((x%3600)//60))}:{format_hms(round(x%60))}")
        identifications['end_hms'] = identifications['end'].apply(lambda x: f"{format_hms(round(x//3600))}:{format_hms(round((x%3600)//60))}:{format_hms(round(x%60))}")
        
        print(f"Number of contexts: {len(contexts_ids)}") 
        print(f"aux: {aux}")

        # Classify each context
        for idx in tqdm(range(len(contexts_ids) - context_len)):
            context = identifications.loc[contexts_ids[idx], ['start_hms', 'end_hms', 'start', 'se_speaker', 'transcription']].reset_index(drop=True)
            phrase = identifications.loc[contexts_ids[idx][-1], ['start_hms', 'end_hms', 'start', 'se_speaker', 'transcription']]
            
            if aux in [None, 'temporal context']:
                if phrase['se_speaker'] == 'trainer':
                    clf = self.detect_feedback(client, phrase, context, verbose=verbose, rag=rag)
                else:
                    clf = {'feedback': 'no'}
            elif aux == 'reduced hallucinations':
                phrase['se_speaker'] = 'unknown'
                context['se_speaker'] = 'unknown'
                clf = self.detect_feedback(client, phrase, context, verbose=verbose, rag=rag)
            elif aux == 'dialogue':
                phrase['se_speaker'] = 'unknown'
                context['se_speaker'] = 'unknown'
                
                clf = self.detect_feedback(client, phrase, context, verbose=verbose, rag=rag)

            first_clip_start = identifications.loc[contexts_ids[idx][0], 'start_hms']
            last_clip_end = identifications.loc[contexts_ids[idx][-1], 'end_hms']
            start = first_clip_start.split(':')
            start = int(start[0])*3600 + int(start[1])*60 + int(start[2])
            end = last_clip_end.split(':')
            end = int(end[0])*3600 + int(end[1])*60 + int(end[2])
            sample_rate = 16000
            frame_offset = int(start*sample_rate)
            min_duration = 0.3
            num_frames = int(max((end-start)*sample_rate, min_duration * sample_rate))
            wav = self.full_wav[:, frame_offset : frame_offset + num_frames]
            clip_filename = os.path.basename(self.full_audio_path).replace('.wav', f'_{int(start)}_{int(end)}.wav')
            clip_save_path = os.path.join(self.audio_clips_dir, clip_filename)
            wavfile.write(clip_save_path, sample_rate, wav.numpy().T)
            
            phrase = f"'{phrase.loc['start_hms']}-{phrase.loc['end_hms']}': ['{phrase.loc['se_speaker']}': '{phrase.loc['transcription']}']"
            
            context_dialogue = "[\n"
            for i in range(len(context)):
                context_dialogue += f"  {i}: ['{context.loc[i, 'se_speaker']}': '{context.loc[i, 'transcription']}']\n"
            context_dialogue += "]"
            
            context_times = "[\n"
            for i in range(len(context)):
                context_times += f"  {i}: {context.loc[i, 'start_hms']}-{context.loc[i, 'end_hms']}\n"
            context_times += "]"
            
            pred_fb_instance = True if clf['feedback'] == 'yes' else False
            
            self.fb_detection.loc[len(self.fb_detection)] = [clip_save_path, context_dialogue, context_times, phrase, pred_fb_instance]
        
        self.fb_detection.to_csv(self.fb_detection_save_path, index=False)
        return self.fb_detection
    
    def _check_alignment(self, client, phrase, human_annotation):
        prompt = f"""\
Classify whether the following two strings have any alignment or not.

Format your response as follows. DO NOT DO ANY OTHER FORMATTING.:
{{'alignment': 'yes'}} if the two strings have any alignment
{{'alignment': 'no'}} if the two strings do not have any alignment

For example:
{{
    'alignment': 'yes',
}}

String 1:
{phrase}

String 2:
{human_annotation}
"""

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"""You are a binary classifier that determines whether two strings have any alignment or not. An alignment means that the two strings might have some common words or phrases that align with each other in terms of their order and/or meaning."""},
                {"role": "user", "content": prompt}
            ],
            seed=self.seed,
        )

        content = completion.choices[0].message.content
        
        try:
            classification = ast.literal_eval(content)
        except Exception as e:
            print(e)
            print(content)
            
            if 'yes' in content:
                classification = {'alignment': 'yes'}
            else:
                classification = {'alignment': 'no'}
        
        return classification
            
    
    # def align_human_fb_detection(self):
    #     if self.fb_detection is None:
    #         self.full_fb_detection()
        
    #     tolerance = 5 # sec
        
    #     self.fb_detection['human_annotations'] = None
    #     self.fb_detection['human_annotations_times'] = None
    #     self.fb_detection['true_fb_instance'] = None
        
    #     client = OpenAI()
        
    #     fb_annot = self.fb_annot[self.fb_annot['Case'] == self.case_id]
    #     fb_annot.reset_index(drop=True, inplace=True)
    #     for i in tqdm(range(len(self.fb_detection))):
    #         last_context_time = self.fb_detection.loc[i, 'context_times'].split('\n')[1:-1][-1].split(' ')[-1].split('-')
    #         start_sec = sum([a*b for a,b in zip([3600, 60, 1], map(int, last_context_time[0].split(':')))])
    #         end_sec = sum([a*b for a,b in zip([3600, 60, 1], map(int, last_context_time[1].split(':')))])
            
    #         k = 0
    #         human_annotations_str = "[\n"
    #         human_annotations_times_str = "[\n"
    #         for j in range(len(fb_annot)):
    #             fbk_secs = sum([a*b for a,b in zip([3600, 60, 1], map(int, fb_annot.loc[j, 'fbk_time'].split(':')))])
    #             if fbk_secs >= start_sec and fbk_secs <= end_sec:
    #                 time = fb_annot.loc[j, 'fbk_time']
    #                 time_str = ':'.join([f"{int(x):02d}" for x in time.split(':')])
                    
    #                 human_annotations_str += f"  {k}: {fb_annot.loc[j, 'Dialogue']}\n"
    #                 human_annotations_times_str += f"  {k}: {time_str}\n"
    #                 k += 1
    #             elif fbk_secs >= start_sec - tolerance and fbk_secs <= end_sec:
    #                 alignmend_check = self._check_alignment(client, self.fb_detection.loc[i, 'phrase'], fb_annot.loc[j, 'Dialogue'])
    #                 if alignmend_check['alignment'] == 'yes':
    #                     time = fb_annot.loc[j, 'fbk_time']
    #                     time_str = ':'.join([f"{int(x):02d}" for x in time.split(':')])
                        
    #                     human_annotations_str += f"  {k}: {fb_annot.loc[j, 'Dialogue']}\n"
    #                     human_annotations_times_str += f"  {k}: {time_str}\n"
    #                     k += 1
                    
    #         human_annotations_str += "]"
    #         human_annotations_times_str += "]"
    #         if k == 0:
    #             human_annotations_str = None
    #             human_annotations_times_str = None
    #             fb_instance = False
    #         else:
    #             fb_instance = True
                
    #         self.fb_detection.loc[i, 'human_annotations'] = human_annotations_str
    #         self.fb_detection.loc[i, 'human_annotations_times'] = human_annotations_times_str
    #         self.fb_detection.loc[i, 'true_fb_instance'] = fb_instance
        
    #     return self.fb_detection
    
    def full_aligned_fb_detection(self, load_saved=True):
        if load_saved and os.path.exists(self.aligned_fb_detection_save_path):
            self.aligned_fb_detection = pd.read_csv(self.aligned_fb_detection_save_path)
            return self.aligned_fb_detection
        
        if self.fb_detection is None:
            self.full_fb_detection()
        
        tolerance = 5 # sec
        
        self.fb_detection['human_annotations'] = None
        self.fb_detection['human_annotations_times'] = None
        self.fb_detection['true_fb_instance'] = None
        
        client = OpenAI()
        
        fb_annot = self.fb_annot[self.fb_annot['Case'] == self.case_id]
        fb_annot.reset_index(drop=True, inplace=True)
        for i in tqdm(range(len(self.fb_detection))):
            last_context_time = self.fb_detection.loc[i, 'context_times'].split('\n')[1:-1][-1].split(' ')[-1].split('-')
            start_sec = sum([a*b for a,b in zip([3600, 60, 1], map(int, last_context_time[0].split(':')))])
            end_sec = sum([a*b for a,b in zip([3600, 60, 1], map(int, last_context_time[1].split(':')))])
            
            k = 0
            human_annotations_str = "[\n"
            human_annotations_times_str = "[\n"
            for j in range(len(fb_annot)):
                fbk_secs = sum([a*b for a,b in zip([3600, 60, 1], map(int, fb_annot.loc[j, 'fbk_time'].split(':')))])
                if fbk_secs >= start_sec and fbk_secs <= end_sec:
                    time = fb_annot.loc[j, 'fbk_time']
                    time_str = ':'.join([f"{int(x):02d}" for x in time.split(':')])
                    
                    human_annotations_str += f"  {k}: {fb_annot.loc[j, 'Dialogue']}\n"
                    human_annotations_times_str += f"  {k}: {time_str}\n"
                    k += 1
                elif fbk_secs >= start_sec - tolerance and fbk_secs <= end_sec:
                    alignmend_check = self._check_alignment(client, self.fb_detection.loc[i, 'phrase'], fb_annot.loc[j, 'Dialogue'])
                    if alignmend_check['alignment'] == 'yes':
                        time = fb_annot.loc[j, 'fbk_time']
                        time_str = ':'.join([f"{int(x):02d}" for x in time.split(':')])
                        
                        human_annotations_str += f"  {k}: {fb_annot.loc[j, 'Dialogue']}\n"
                        human_annotations_times_str += f"  {k}: {time_str}\n"
                        k += 1
                    
            human_annotations_str += "]"
            human_annotations_times_str += "]"
            if k == 0 or 'trainee' in self.fb_detection.loc[i, 'phrase']:
                human_annotations_str = None
                human_annotations_times_str = None
                fb_instance = False
            else:
                fb_instance = True
                
            self.fb_detection.loc[i, 'human_annotations'] = human_annotations_str
            self.fb_detection.loc[i, 'human_annotations_times'] = human_annotations_times_str
            self.fb_detection.loc[i, 'true_fb_instance'] = fb_instance
        
        # Get specific console_time
        console_times = self.console_times[self.console_times['case_id'] == self.case_id]
        
        self.aligned_fb_detection = pd.DataFrame(columns=self.fb_detection.columns)
        
        fb_detection = self.fb_detection.copy()
        for i in range(len(fb_detection)):
            # start_sec = sum([a*b for a,b in zip([3600, 60, 1], map(int, self.fb_detection.loc[i, 'phrase'].split(' ')[0][:-1].split(':')))])
            start_sec = sum([a*b for a,b in zip([3600, 60, 1], map(int, fb_detection.loc[i, 'phrase'].split(' ')[0][:-1].replace('\'', '').split('-')[0].split(':')))])

            valid = True

            console_times_tmp = console_times.copy()
            console_times_tmp = console_times_tmp[console_times_tmp['is_eligible_time'] == 0]
            console_times_tmp.reset_index(drop=True, inplace=True)
            for j in range(len(console_times_tmp)):
                interval_start = console_times_tmp.loc[j, 'On time (secs)']
                interval_end = console_times_tmp.loc[j, 'Off time (secs)']
                if interval_start < start_sec and interval_end > start_sec:
                    valid = False
                    
            if valid:
                self.aligned_fb_detection.loc[len(self.aligned_fb_detection)] = fb_detection.loc[i]
        
        self.aligned_fb_detection.to_csv(self.aligned_fb_detection_save_path, index=False)
        return self.aligned_fb_detection
    
    def predict_trainee_behavior(self, client: OpenAI, phrase, context, verbose=False, use_human_annotations=False, use_clustering_defintion=False):
        if not use_human_annotations:
            context = context.reset_index(drop=True)
            
            phrase_str = f"{phrase.loc['start_hms']}: ['{phrase.loc['se_speaker']}': '{phrase.loc['transcription']}']"
            
            context_str = ""
            for i in range(len(context)):
                context_str += f"{context.loc[i, 'start_hms']}: ['{context.loc[i, 'se_speaker']}': '{context.loc[i, 'transcription']}']\n"
                
            prompt = f"""\
Classify whether the following feedback phrase will lead to a trainee response where a trainee response can be either 1) verbal acknowledgement, 2) behavioral change. Do this while considering the given context of the last couple turns in the dialogue where the phrase is the last entry in the context.

Context:
{context_str}

Phrase:
{phrase_str}

Format your response as follows. DO NOT DO ANY OTHER FORMATTING.:
'verbal acknowledgement': 'yes' if you predict the trainee to respond with a verbal acknowledgement otherwise 'no'
'behavioral change': 'yes' if you predict the trainee to respond with a behavioral change otherwise 'no'

Your output can be a combination of the two categories. For example:
{{
    'verbal acknowledgement': 'yes',
    'behavioral change': 'no',
}}
"""     
        else:
            annotations_str = phrase['human_annotations']
            
            prompt = f"""\
Classify whether the following feedback phrase will lead to a trainee response where a trainee response can be either 1) verbal acknowledgement, 2) behavioral change.

Feedback:
{annotations_str}

Format your response as follows. DO NOT DO ANY OTHER FORMATTING.:
'verbal acknowledgement': 'yes' if you predict the trainee to respond with a verbal acknowledgement otherwise 'no'
'behavioral change': 'yes' if you predict the trainee to respond with a behavioral change otherwise 'no'

Your output can be a combination of the two categories. For example:
{{
    'verbal acknowledgement': 'yes',
    'behavioral change': 'no',
}}
"""

        print(prompt) if verbose else None
        
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"""
You are an AI assistant specializing in predicting trainee responses during urology surgery training using the da Vinci robot. Your task is to analyze dialogue between a trainer and a trainee, focusing on the trainee's reactions to feedback. The dialogue may contain multiple consecutive turns by the same speaker due to missed responses or varying speech patterns.

You will categorize potential trainee responses into two types:

1. Verbal Acknowledgement: This includes any verbal or audible confirmation from the trainee indicating they have heard and understood the feedback. Examples include:
   - "Okay, I see"
   - "Uh-huh, got it"
   - "Understood"
   - "Yes, I'll do that"

2. Behavioral Change: This refers to any physical or observable adjustment made by the trainee that directly corresponds to the feedback received. For example:
   - If the trainer suggests tightening a suture, the trainee immediately pulls the suture thread more tightly.

Your role is to predict which type(s) of response the trainee is likely to give based on the specific feedback provided by the trainer. Consider the context of the surgical procedure and the nature of the feedback when making your predictions.
""" if not use_clustering_defintion else f"""\
You are an AI assistant specialized in predicting trainee responses during urology surgery training sessions using the da Vinci surgical system. Your role involves scrutinizing interactions between a trainer and a trainee, paying close attention to how the trainee reacts to verbal feedback during the procedure. This dialogue analysis can sometimes feature multiple consecutive statements by one party due to variations in communication style or pauses in response.

Your task is to classify potential trainee responses into two categories:

Verbal Acknowledgement: This includes any spoken affirmations from the trainee that signify comprehension and acknowledgment of the feedback. Common examples include:
"Okay, I understand."
"Yes, I'll adjust that."
"Got it, thanks."
"Sure, will do."

Behavioral Change: This category captures any tangible and immediate modifications in the trainee's actions as a direct result of the feedback, such as:
Adjusting the grip or technique as per the trainer's guidance.
Correcting the camera angle or the positioning of the surgical tools following a suggestion.
Your function is to predict which type of response, verbal or behavioral, the trainee is likely to exhibit based on the specific nuances of the feedback delivered by the trainer. Consider both the surgical context and the specific content of the feedback to make your predictions.

Examples of feedback that typically lead to verbal acknowledgments from the trainee include:

Direct questions about understanding or procedure ("Does that make sense?", "Okay, can I show you one thing here?")
Feedback that involves personal teaching preferences or critiques ("My pet peeve is...", "I'm not impressed", "The efficiency is pretty bad, okay?")
Constructive suggestions requiring verbal confirmation ("Stay there", "So now, you need to change your angle, yeah?", "I want you to be like that, okay?")
Examples of feedback not leading to verbal acknowledgment from the trainee tend to be more informational or directive without seeking a verbal response:

Simple directives ("Check on the left side", "Just give that a little tap there")
Observations or minor corrections that can be immediately acted upon without the need for verbal acknowledgment ("Just buzz that right there", "Open up wide, this is what I mean by not digging yourself in a hole")
Instructions that are followed by an immediate physical adjustment or check by the trainer ("So inch a little bit towards the prostate", "Okay, that's enough, you go too lateral you're going to get into the pedicles")
This nuanced categorization helps in understanding how feedback is typically received and acted upon, enhancing the training process by aligning trainer expectations with trainee responses.
"""},
                {"role": "user", "content": prompt}
            ],
            seed=self.seed,
        )

        content = completion.choices[0].message.content

        print(content) if verbose else None
        # classification = json.loads(content)
        
        classification = None
        try:
            classification = ast.literal_eval(content)
            classification['ask for clarification'] = None
        except Exception as e:
            print(e)
            print(content)
        
        
        return classification
    
    def full_behavior_prediction(self, load_saved=True, verbose=False, use_human_annotations=False, use_clustering_defintion=False):
        if load_saved and os.path.exists(self.behavior_prediction_save_path):
            self.behavior_prediction = pd.read_csv(self.behavior_prediction_save_path)
            return self.behavior_prediction

        if self.aligned_fb_detection is None:
            self.full_aligned_fb_detection()
        
        self.behavior_prediction = self.aligned_fb_detection.copy()
        self.behavior_prediction = self.behavior_prediction[self.behavior_prediction['pred_fb_instance'] == True]
        self.behavior_prediction = self.behavior_prediction[self.behavior_prediction['true_fb_instance'] == True]
        
        self.behavior_prediction['human_annotations_r_t'] = None
        self.behavior_prediction['true_r_t_verb'] = None
        self.behavior_prediction['true_r_t_beh'] = None
        self.behavior_prediction['true_r_t_clarify'] = None
        self.behavior_prediction['pred_r_t_verb'] = None
        self.behavior_prediction['pred_r_t_beh'] = None
        self.behavior_prediction['pred_r_t_clarify'] = None
        # self.behavior_prediction['pred_r_t'] = None
        # self.behavior_prediction['true_behavior_response'] = None
        # self.behavior_prediction['pred_behavior_response'] = None
        self.behavior_prediction.reset_index(drop=True, inplace=True)
        
        client = OpenAI()
        
        for i in tqdm(range(len(self.behavior_prediction))):
            context_dialogue, context_times, phrase_str = self.behavior_prediction.loc[i, 'context_dialogue'], self.behavior_prediction.loc[i, 'context_times'], self.behavior_prediction.loc[i, 'phrase']
            human_annotations, human_annotations_times = self.behavior_prediction.loc[i, 'human_annotations'], self.behavior_prediction.loc[i, 'human_annotations_times']
            context_dialogue = context_dialogue.split('\n')[1:-1]
            context_times = context_times.split('\n')[1:-1]
            
            # Reconstruct the context df
            context = pd.DataFrame(columns=['start_hms', 'se_speaker', 'transcription'])
            for j in range(len(context_dialogue)):
                start_hms = context_times[j].split(' ')[-1].split('-')[0]
                se_speaker = context_dialogue[j].strip().split(' ')[1][2:-2]
                transcription = ' '.join(context_dialogue[j].strip().split(' ')[2:])[1:-2]
                context.loc[j] = [start_hms, se_speaker, transcription]
            
            # Reconstruct the phrase series
            phrase = pd.Series(index=['start_hms', 'se_speaker', 'transcription'])
            phrase['start_hms'] = phrase_str[1:9]
            phrase['se_speaker'] = phrase_str.strip().split(' ')[1][2:-2]
            phrase['transcription'] = ' '.join(phrase_str.strip().split(' ')[2:])[1:-2]
            annotations = [x.split(':')[1] for x in human_annotations.split('\n')[-2:-1]]
            annotations_times = [':'.join(x.split(':')[1:]) for x in human_annotations_times.split('\n')[1:-1]]
            phrase['human_annotations'] = "\n".join([f"{annotations[i]}" for i in range(len(annotations))])
            
            clf = self.predict_trainee_behavior(client, phrase, context, verbose=verbose, use_human_annotations=use_human_annotations, use_clustering_defintion=use_clustering_defintion)
            
            fbk_times = self.behavior_prediction.loc[i, 'human_annotations_times'].split('\n')[1:-1]
            fbk_times = [x.split(' ')[-1] for x in fbk_times]
            fbk_times = [':'.join([str(int(y)) for y in x.split(':')]) for x in fbk_times]
            
            human_annotations_r_t = "[\n"
            true_behavior_response = False
            r_t_dict = {'r_t_verbs': [], 'r_t_behs': [], 'r_t_clarify': []}
            for j, fbk_time in enumerate(fbk_times):
                fb_annot_row = self.fb_annot[self.fb_annot['fbk_time'] == str(fbk_time)].iloc[0]
                
                r_t_verb, r_t_beh, r_t_clarify = fb_annot_row['r_t_verb'], fb_annot_row['r_t_beh'], fb_annot_row['r_t_clarify']
                
                true_r_t = f"['r_t_verb': {r_t_verb},  'r_t_beh': {r_t_beh},  'r_t_clarify': {r_t_clarify}]"
                true_behavior_response = True if r_t_verb or r_t_beh or r_t_clarify else true_behavior_response
                
                r_t_dict['r_t_behs'].append(True if r_t_beh else False)
                r_t_dict['r_t_clarify'].append(True if r_t_clarify else False)
                r_t_dict['r_t_verbs'].append(True if r_t_verb else False)
                
                human_annotations_r_t += f"  {j}: {true_r_t}\n"
            human_annotations_r_t += "]"
            if human_annotations_r_t == "[\n]":
                human_annotations_r_t = None
                
            self.behavior_prediction.loc[i, 'true_r_t_verb'] = max(r_t_dict['r_t_verbs'])
            self.behavior_prediction.loc[i, 'true_r_t_beh'] = max(r_t_dict['r_t_behs'])
            self.behavior_prediction.loc[i, 'true_r_t_clarify'] = max(r_t_dict['r_t_clarify'])
            
            self.behavior_prediction.loc[i, 'pred_r_t_verb'] = True if clf['verbal acknowledgement'] == 'yes' else False
            self.behavior_prediction.loc[i, 'pred_r_t_beh'] = True if clf['behavioral change'] == 'yes' else False
            self.behavior_prediction.loc[i, 'pred_r_t_clarify'] = True if clf['ask for clarification'] == 'yes' else False
            
            # self.behavior_prediction.loc[i, 'true_behavior_response'] = true_behavior_response
            # self.behavior_prediction.loc[i, 'pred_behavior_response'] = pred_behavior_response
            self.behavior_prediction.loc[i, 'human_annotations_r_t'] = human_annotations_r_t

        
        self.behavior_prediction.to_csv(self.behavior_prediction_save_path, index=False)

        return self.behavior_prediction
    
    def classify_components(self, client: OpenAI, phrase, context, verbose=False, rag=False, use_huamn_annotations=False):
        if not use_huamn_annotations:
            context = context.reset_index(drop=True)

            phrase_str = f"{phrase.loc['start_hms']}: ['{phrase.loc['se_speaker']}': '{phrase.loc['transcription']}']"

            context_str = ""
            for i in range(len(context)):
                context_str += f"{context.loc[i, 'start_hms']}: ['{context.loc[i, 'se_speaker']}': '{context.loc[i, 'transcription']}']\n"
            
            # Get RAG examples
            if rag:
                context_dialogue = [" ".join(context['transcription'])]
                embedding = self.sentence_transformer_model.encode(context_dialogue)[0]
                f_anatomic_examples, f_procedural_examples, f_technical_examples = rag_sample_examples_unseen_case_component(self.all_annotations_component, self.case_id, embedding, self.sentence_transformer_model)
                
                f_anatomic_examples_str = "\n".join(f_anatomic_examples['context_dialogue'].values)
                f_procedural_examples_str = "\n".join(f_procedural_examples['context_dialogue'].values)
                f_technical_examples_str = "\n".join(f_technical_examples['context_dialogue'].values)
                
                add_on = f"""\n\nExamples of dialogue interactions where the feedback the (last phrase) has anatomic components:
{f_anatomic_examples_str}

Examples of dialogue interactions where the feedback the (last phrase) has procedural components:
{f_procedural_examples_str}

Examples of dialogue interactions where the feedback the (last phrase) has technical components:
{f_technical_examples_str}"""
        
            prompt = f"""\
Classify the feedback phrase into one or more of the following categories: 1) anatomic, 2) procedural, 3) technical. Do this while considering the given context of the last couple turns in the dialogue where the phrase is the last entry in the context.

Context:
{context_str}

Phrase:
{phrase_str}

Format your response as follows. DO NOT DO ANY OTHER FORMATTING.:
'anatomic': 'yes' if the feedback is anatomic otherwise 'no'
'procedural': 'yes' if the feedback is procedural otherwise 'no'
'technical': 'yes' if the feedback is technical otherwise 'no'

Your output can be a combination of the three categories. For example:
{{
'anatomic': 'yes',
'procedural': 'no',
'technical': 'yes',
}}
"""
            if rag:
                prompt += add_on
        else:
            annotations_str = phrase['human_annotations']
            
            prompt = f"""\
Classify the feedback annotations into one or more of the following categories: 1) anatomic, 2) procedural, 3) technical. 

Feedback:
{annotations_str}

Format your response as follows. DO NOT DO ANY OTHER FORMATTING.:
'anatomic': 'yes' if the feedback is anatomic otherwise 'no'
'procedural': 'yes' if the feedback is procedural otherwise 'no'
'technical': 'yes' if the feedback is technical otherwise 'no'

Your output can be a combination of the three categories. For example:
{{
'anatomic': 'yes',
'procedural': 'no',
'technical': 'no',
}}
"""

            
        print(prompt) if verbose else None

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"""\
You are an AI assistant specializing in classifying feedback during urology surgery training using the da Vinci robot. Your task is to analyze dialogue between a trainer and a trainee, focusing on categorizing the feedback into anatomic, procedural, and/or technical. {'The dialogue may contain multiple consecutive turns by the same speaker due to missed responses or varying speech patterns.' if not use_huamn_annotations else ''}

You will categorize the feedback into three types:

1. Anatomic: Familiriaty with anatomic structures and landmarks. Examples include:
 - "Stay in the correct plane, between the 2 fascial layers."
 - "Avoid the blood vessels here."

2. Procedural: Pertains to the timing and sequence of surgical steps. Examples include:
 - "You need to suture this area first."
 - "You can switch to the left side now."

3. Technical: Performance of a discrete task with appropriate knowledge of factors including exposure, instruments, and traction. Examples include:
 - "Adjust the tension on the suture."
 - "Buzz it."

Your role is to predict which type(s) of feedback the phrase contains based on the specific feedback provided by the trainer. Consider the context of the surgical procedure and the nature of the feedback when making your predictions.
"""},
                    {"role": "user", "content": prompt}
                ],
                seed=self.seed,
            )

        content = completion.choices[0].message.content

        print("Content:\n", content) if verbose else None
        # classification = json.loads(content)

        classification = None
        try:
            classification = ast.literal_eval(content)
        except Exception as e:
            print("====================================")
            print(f"Error {e}")
            print("====================================")
            print()

        return classification
        
    
    def full_component_classification(self, load_saved=True, verbose=False, rag=False, use_huamn_annotations=False):
        if load_saved and os.path.exists(self.component_classification_save_path):
            self.component_classification = pd.read_csv(self.component_classification_save_path)
            return self.component_classification
        
        self.component_classification = self.aligned_fb_detection.copy()
        self.component_classification = self.component_classification[self.component_classification['pred_fb_instance'] == True]
        self.component_classification = self.component_classification[self.component_classification['true_fb_instance'] == True]
        
        self.component_classification['human_annotations_f'] = None
        self.component_classification['true_f_anatomic'] = None
        self.component_classification['true_f_procedural'] = None
        self.component_classification['true_f_technical'] = None
        self.component_classification['pred_f_anatomic'] = None
        self.component_classification['pred_f_procedural'] = None
        self.component_classification['pred_f_technical'] = None
        self.component_classification.reset_index(drop=True, inplace=True)
        
        client = OpenAI()
        
        for i in tqdm(range(len(self.component_classification))):
            context_dialogue, context_times, phrase_str = self.component_classification.loc[i, 'context_dialogue'], self.component_classification.loc[i, 'context_times'], self.component_classification.loc[i, 'phrase']
            human_annotations, human_annotations_times = self.component_classification.loc[i, 'human_annotations'], self.component_classification.loc[i, 'human_annotations_times']
            context_dialogue = context_dialogue.split('\n')[1:-1]
            context_times = context_times.split('\n')[1:-1]
            
            # Reconstruct the context df
            context = pd.DataFrame(columns=['start_hms', 'se_speaker', 'transcription'])
            for j in range(len(context_dialogue)):
                start_hms = context_times[j].split(' ')[-1].split('-')[0]
                se_speaker = context_dialogue[j].strip().split(' ')[1][2:-2]
                transcription = ' '.join(context_dialogue[j].strip().split(' ')[2:])[1:-2]
                context.loc[j] = [start_hms, se_speaker, transcription]
            
            # Reconstruct the phrase series
            phrase = pd.Series(index=['start_hms', 'se_speaker', 'transcription', 'human_annotations'])
            phrase['start_hms'] = phrase_str[1:9]
            phrase['se_speaker'] = phrase_str.strip().split(' ')[1][2:-2]
            phrase['transcription'] = ' '.join(phrase_str.strip().split(' ')[2:])[1:-2]
            annotations = [x.split(':')[1] for x in human_annotations.split('\n')[1:-1]]
            annotations_times = [':'.join(x.split(':')[1:]) for x in human_annotations_times.split('\n')[1:-1]]
            phrase['human_annotations'] = "\n".join([f"{annotations_times[i]}: {annotations[i]}" for i in range(len(annotations))])
        
            clf = self.classify_components(client, phrase, context, verbose=verbose, rag=rag, use_huamn_annotations=use_huamn_annotations)
            if clf is None:
                clf = {'anatomic': 'no', 'procedural': 'no', 'technical': 'no'}
            
            fbk_times = self.behavior_prediction.loc[i, 'human_annotations_times'].split('\n')[1:-1]
            fbk_times = [x.split(' ')[-1] for x in fbk_times]
            fbk_times = [':'.join([str(int(y)) for y in x.split(':')]) for x in fbk_times]
            
            human_annotations_f = "[\n"
            f_dict = {'f_anatomic': [], 'f_procedural': [], 'f_technical': []}
            for j, fbk_time in enumerate(fbk_times):
                fb_annot_row = self.fb_annot[self.fb_annot['fbk_time'] == str(fbk_time)].iloc[0]
                
                f_anatomic, f_procedural, f_technical = fb_annot_row['f_anatomic'], fb_annot_row['f_procedural'], fb_annot_row['f_technical']
                
                true_f = f"['f_anatomic': {int(f_anatomic)},  'f_procedural': {int(f_procedural)},  'f_technical': {int(f_technical)}]"
                
                f_dict['f_anatomic'].append(True if f_anatomic else False)
                f_dict['f_procedural'].append(True if f_procedural else False)
                f_dict['f_technical'].append(True if f_technical else False)
                                
                human_annotations_f += f"  {j}: {true_f}\n"
            human_annotations_f += "]"
            if human_annotations_f == "[\n]":
                human_annotations_f = None
                
            self.component_classification.loc[i, 'true_f_anatomic'] = max(f_dict['f_anatomic'])
            self.component_classification.loc[i, 'true_f_procedural'] = max(f_dict['f_procedural'])
            self.component_classification.loc[i, 'true_f_technical'] = max(f_dict['f_technical'])
            print(f"clf: {clf}") if verbose else None
            self.component_classification.loc[i, 'pred_f_anatomic'] = True if clf['anatomic'] == 'yes' else False
            self.component_classification.loc[i, 'pred_f_procedural'] = True if clf['procedural'] == 'yes' else False
            self.component_classification.loc[i, 'pred_f_technical'] = True if clf['technical'] == 'yes' else False
            
            self.component_classification.loc[i, 'human_annotations_f'] = human_annotations_f

        
        self.component_classification.to_csv(self.component_classification_save_path, index=False)

        return self.component_classification


    def evaluate_fb_detection(self, weighting='weighted'):
        pred = self.aligned_fb_detection['pred_fb_instance'].replace({True: 1, False: 0})
        true = self.aligned_fb_detection['true_fb_instance'].replace({True: 1, False: 0})
        
        accuracy = accuracy_score(true, pred)
        precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average=weighting)
        try:
            roc_auc = roc_auc_score(true, pred, average='weighted')
        except ValueError:
            roc_auc = None
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'accuracy': accuracy
        }
        
        return metrics
    def evaluate_behavior_prediction(self, weighting='weighted'):
        metrics = {}
        for response_type in ['r_t_verb', 'r_t_beh', 'r_t_clarify']:
            pred = self.behavior_prediction[f'pred_{response_type}']
            true = self.behavior_prediction[f'true_{response_type}']

            accuracy = accuracy_score(true, pred)
            precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average=weighting)
            try:
                roc_auc = roc_auc_score(true, pred, average='weighted')
            except ValueError:
                roc_auc = None
            
            for metric_type in ['precision', 'recall', 'f1', 'accuracy', 'roc_auc']:
                metrics[f"{metric_type}_{response_type}"] = locals()[metric_type]
            
        return metrics
    
    def evaluate_component_classification(self, weighting='weighted'):
        metrics = {}
        for component in ['f_anatomic', 'f_procedural', 'f_technical']:
            pred = self.component_classification[f'pred_{component}']
            true = self.component_classification[f'true_{component}']

            accuracy = accuracy_score(true, pred)
            precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average=weighting)
            try:
                roc_auc = roc_auc_score(true, pred, average='weighted')
            except ValueError:
                roc_auc = None
            
            for metric_type in ['precision', 'recall', 'f1', 'accuracy', 'roc_auc']:
                metrics[f"{metric_type}_{component}"] = locals()[metric_type]
            
        return metrics        
    
    def evaluate(self, weighting='weighted', model_type='fb'):
        if model_type == 'fb':
            return self.evaluate_fb_detection(weighting)
        elif model_type == 'behavior':
            return self.evaluate_behavior_prediction(weighting)
        elif model_type == 'component':
            return self.evaluate_component_classification(weighting)
        else:
            return ValueError("Model type must be 'fb', 'behavior', or 'component'.")