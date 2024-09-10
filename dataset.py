from datasets import Dataset
import torch
import numpy as np
import os
from transformers import AutoFeatureExtractor, AutoTokenizer

from utils import (
    get_waveforms
)

class TextDataset(Dataset):
    def __init__(self, transcriptions_df, tokenizer, text_col, label_col, file_col):
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
        self.text_col  = text_col
        self.label_col = label_col
        self.file_col  = file_col
        
        self.transcriptions_df = transcriptions_df.copy()
        self.transcriptions_df.reset_index(drop=True, inplace=True)
    
    def __len__(self):
        return len(self.transcriptions_df)

    def __getitem__(self, idx):
        transcriptions = self.transcriptions_df.iloc[idx]
        
        texts, labels, files = None, None, None
        if isinstance(idx, int):
            texts = '' if transcriptions[self.text_col] is np.nan else transcriptions[self.text_col]
            labels = transcriptions[self.label_col]
            files = transcriptions[self.file_col]
        else:
            texts = transcriptions[self.text_col].replace(np.nan, '').to_list()
            labels = transcriptions[self.label_col].to_list()
            files = transcriptions[self.file_col].to_list()

        encodings = self.tokenizer(texts, truncation=True, padding=True)
        
        data = {
            # inputs
            'input_ids': torch.tensor(encodings['input_ids']),
            'token_type_ids': torch.tensor(encodings['token_type_ids']),
            'attention_mask': torch.tensor(encodings['attention_mask']),
            
            # output
            'label': torch.tensor(labels),
            
            # misc
            'file': files,
            'text': texts,
        }
        return data

    def __repr__(self):
        return f"TextDataset(num_samples={len(self)}, input_features=['input_ids', 'token_type_ids', 'attention_mask'], output_features=['label'], misc_features=['file', 'text'])"


class AudioDataset(Dataset):
    def __init__(self, wavs_df, feature_extractor, channel, file_col, label_col, features_dir=None):
        if isinstance(feature_extractor, str):
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor)
        else:
            self.feature_extractor = feature_extractor
        self.file_col     = file_col
        self.label_col    = label_col
        self.channel      = channel
        self.features_dir = features_dir
        os.makedirs(self.features_dir, exist_ok=True)
        
        self.wavs_df = wavs_df.copy()
        self.wavs_df.reset_index(drop=True, inplace=True)
    
    def __len__(self):
        return len(self.wavs_df)

    def __getitem__(self, idx):
        wavs = self.wavs_df.iloc[idx]
        
        all_waveforms = None
        if isinstance(idx, int):
            waveforms, sampling_rate = get_waveforms(wavs[self.file_col], include_channels=[self.channel])
            labels = [wavs[self.label_col]]
            files = [wavs[self.file_col]]
            all_waveforms = {self.channel: {wavs[self.file_col]: waveforms[self.channel]}}
        else:
            all_waveforms = {self.channel: {}}
            iterator = range(idx.start, idx.stop) if isinstance(idx, slice) else idx
            for i in iterator:
                waveforms, sampling_rate = get_waveforms(self.wavs_df[self.file_col].iloc[i], include_channels=[self.channel])
                all_waveforms[self.channel][self.wavs_df[self.file_col].iloc[i]] = waveforms[self.channel]
            
            labels = wavs[self.label_col].to_list()
            files = wavs[self.file_col].to_list()
        
        ## CACHING
        if self.features_dir is not None:
            features = {'input_values': [], 'attention_mask': []}
            for file in files:
                filename = file.split('/')[-1].replace('.wav', f'_{self.channel}.pt')
                features_file = f"{self.features_dir}/{filename}"
                if os.path.exists(features_file):
                    ind_features = torch.load(features_file)
                    features['input_values'].append(ind_features['input_values'])
                    features['attention_mask'].append(ind_features['attention_mask']) if 'attention_mask' in ind_features else None
                else:
                    ind_features = self.feature_extractor(all_waveforms[self.channel][file], sampling_rate=16000, return_tensors='pt')
                    ind_features = {k: v.squeeze(0) for k, v in ind_features.items()}

                    features['input_values'].append(ind_features['input_values'])
                    features['attention_mask'].append(ind_features['attention_mask']) if 'attention_mask' in ind_features else None
                    torch.save(ind_features, features_file)
            features = {k: torch.stack(v).squeeze(1) for k, v in features.items() if len(v) > 0}      
        else:
            features = self.feature_extractor(list(all_waveforms[self.channel].values()), sampling_rate=16000, return_tensors='pt')
        
        data = {
            # inputs
            'input_values': torch.tensor(features['input_values'], dtype=torch.float32),
            'attention_mask': torch.tensor(features['attention_mask'], dtype=torch.int16) if 'attention_mask' in features else None,
            
            # output
            'labels': torch.tensor(labels, dtype=torch.int64),
            
            # misc
            'file': files,
        }
        if 'attention_mask' not in features:
            data['attention_mask'] = torch.ones(data['input_values'].shape, dtype=torch.int16)
        
        return data

    def __repr__(self):
        return f"AudioDataset(num_samples={len(self)}, input_features=['input_values', 'attention_mask'], output_features=['labels'], misc_features=['file'])"


class AudioTextDataset(Dataset):
    def __init__(self, transcriptions_df, tokenizer, feature_extractor, text_col, channel, label_col, file_col, audio_features_dir=None):
        wavs_df = transcriptions_df.copy()[[file_col, label_col]]
        
        self.text_dataset = TextDataset(
            transcriptions_df=transcriptions_df,
            tokenizer=tokenizer,
            text_col=text_col,
            label_col=label_col,
            file_col=file_col,
        )
        self.audio_dataset = AudioDataset(
            wavs_df=wavs_df,
            feature_extractor=feature_extractor,
            channel=channel,
            file_col=file_col,
            label_col=label_col,
            features_dir=audio_features_dir,
        )
        
        
        # self.text_dataset = TextDataset(
        #     transcriptions_df=text_dataset_args['transcriptions_df'],
        #     tokenizer=text_dataset_args['tokenizer'],
        #     text_col=text_dataset_args['text_col'],
        #     label_col=text_dataset_args['label_col'],
        #     file_col=text_dataset_args['file_col'],
        # )
        # self.audio_dataset = AudioDataset(
        #     wavs_df=audio_dataset_args['wavs_df'],
        #     feature_extractor=audio_dataset_args['feature_extractor'],
        #     channel=audio_dataset_args['channel'],
        #     file_col=audio_dataset_args['file_col'],
        #     label_col=audio_dataset_args['label_col'],
        #     features_dir=audio_dataset_args['features_dir'],
        # )

    def __len__(self):
        return len(self.text_dataset)
    
    def __getitem__(self, idx):
        text_data = self.text_dataset[idx]
        audio_data = self.audio_dataset[idx]

        data = {
            # text inputs
            'text_input_ids': text_data['input_ids'],
            'text_token_type_ids': text_data['token_type_ids'],
            'text_attention_mask': text_data['attention_mask'],
            
            # audio inputs
            'audio_input_values': audio_data['input_values'],
            'audio_attention_mask': audio_data['attention_mask'],
            
            # outputs
            'labels': audio_data['labels'],
            
            # misc
            'file': text_data['file'],
            'text': text_data['text'],
        }
        
        return data

    def __repr__(self):
        return f"AudioTextDataset(num_samples={len(self)}, text_dataset={self.text_dataset}, audio_dataset={self.audio_dataset})"