import wandb
from transformers import set_seed, TrainingArguments, Trainer, BertForSequenceClassification, AutoTokenizer, AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from dataset import TextDataset, AudioDataset, AudioTextDataset
from CustomTrainer import CustomTrainer
from models import TextModel, AudioModel, AudioTextFusionModel

def compute_metrics(eval_pred):
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    roc_auc = roc_auc_score(labels, preds)
    
    metrics =  {'accuracy': accuracy,
                'roc_auc': roc_auc,
                'precision': precision,
                'recall': recall,
                'f1': f1}
    return metrics

def train_text_model(text_model, train_transcriptions_df, eval_transcriptions_df, num_classes, class_weights, device, output_dir, wandb_project_name=None, seed=42, epochs=5, batch_size=32, lr_scheduler_type='linear', lr_scheduler_kwargs={}, warmup_steps=500, weight_decay=0.1, save_steps=100, eval_steps=100, eval_save_strategy='steps', metric_for_best_model='eval_roc_auc', report_to='wandb'):
    params_model = {
        'text_model': text_model,
        'num_classes': num_classes,
        'class_weights': class_weights
    }
    model = TextModel(params_model, device)
    # model = BertForSequenceClassification.from_pretrained(text_model, num_labels=num_classes, output_attentions=False, output_hidden_states=False)
    
    
    train_dataset = TextDataset(train_transcriptions_df, text_model, 'whisper_transcription', 'fb_instance', 'file')
    eval_dataset = TextDataset(eval_transcriptions_df, text_model, 'whisper_transcription', 'fb_instance', 'file')
    
    # Init wandb
    if wandb_project_name is not None:
        wandb.init(project=wandb_project_name)
    
    set_seed(seed)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        eval_strategy=eval_save_strategy,
        save_strategy=eval_save_strategy,
        save_steps=save_steps,
        eval_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        report_to=report_to,
        seed=seed,
        lr_scheduler_type=lr_scheduler_type,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        save_total_limit=5,
        remove_unused_columns=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()
    
    return trainer.model

def train_audio_model(audio_model, train_wavs_df, eval_wavs_df, channel, num_classes, class_weights, device, output_dir, audio_features_dir=None, wandb_project_name=None, seed=42, epochs=5, batch_size=2, lr_scheduler_type='linear', lr_scheduler_kwargs={}, warmup_steps=500, weight_decay=0.1, save_steps=100, eval_steps=100, eval_save_strategy='steps', metric_for_best_model='eval_roc_auc', report_to='wandb'):
    params_model = {
        'audio_model': audio_model,
        'num_classes': num_classes,
        'class_weights': class_weights
    }
    model = AudioModel(params_model, device)

    train_dataset = AudioDataset(train_wavs_df, audio_model, channel, file_col='file', label_col='fb_instance', features_dir=audio_features_dir)
    eval_dataset = AudioDataset(eval_wavs_df, audio_model, channel, file_col='file', label_col='fb_instance', features_dir=audio_features_dir)
    
    # Init wandb
    if wandb_project_name is not None:
        wandb.init(project=wandb_project_name)
    
    set_seed(seed)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        eval_strategy=eval_save_strategy,
        save_strategy=eval_save_strategy,
        save_steps=save_steps,
        eval_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        report_to=report_to,
        seed=seed,
        lr_scheduler_type=lr_scheduler_type,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        save_total_limit=5,
        remove_unused_columns=False
    )
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()
    
    return trainer.model

import os, json
from safetensors import safe_open

def get_best_checkpoint_dir(checkpoint_dir):
    checkpoints = os.listdir(checkpoint_dir)
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))
    last_checkpoint = checkpoints[-1]
    last_trainer_state = json.load(open(f'{checkpoint_dir}/{last_checkpoint}/trainer_state.json'))
    best_checkpoint_dir = last_trainer_state['best_model_checkpoint']
    
    return best_checkpoint_dir

def load_clf_model(model_dir, model_class, device, text_model, audio_model):
    params_model = {
        "num_classes": 2,
        "class_weights": None,
        "num_features": 256,
        "text_model": text_model,
        "audio_model": audio_model,
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
    with safe_open(f'{best_checkpoint_dir}/model.safetensors', framework='pt') as f:
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

def train_multimodal_model(text_model, audio_model, train_transcriptions_df, eval_transcriptions_df, num_classes, num_features, class_weights, device, output_dir, audio_features_dir=None, wandb_project_name=None, seed=42, epochs=5, batch_size=2, lr_scheduler_type='linear', lr_scheduler_kwargs={}, warmup_steps=500, weight_decay=0.1, save_steps=100, eval_steps=100, eval_save_strategy='steps', metric_for_best_model='eval_roc_auc', report_to='wandb', bert_pretrained_dir=None, wav2vec2_pretrained_dir=None):
    params_model = {
        'text_model': text_model,
        'audio_model': audio_model,
        'num_classes': num_classes,
        'class_weights': class_weights,
        'num_features': num_features,
    }
    model = AudioTextFusionModel(params_model, device)
    
    if bert_pretrained_dir is not None:
        text_clf = load_clf_model(bert_pretrained_dir, 'TextModel', device, text_model, audio_model)
        model.text_model.text_model.bert = text_clf.text_model.bert
    
    if wav2vec2_pretrained_dir is not None:
        audio_clf = load_clf_model(wav2vec2_pretrained_dir, 'AudioModel', device, text_model, audio_model)
        model.audio_model.audio_model.wav2vec2 = audio_clf.audio_model.wav2vec2
    
    train_dataset = AudioTextDataset(
        transcriptions_df=train_transcriptions_df,
        tokenizer=text_model,
        feature_extractor=audio_model,
        text_col='whisper_transcription',
        channel='both',
        audio_features_dir=audio_features_dir,
        label_col='fb_instance',
        file_col='file',
    )
    eval_dataset = AudioTextDataset(
        transcriptions_df=eval_transcriptions_df,
        tokenizer=text_model,
        feature_extractor=audio_model,
        text_col='whisper_transcription',
        channel='both',
        audio_features_dir=audio_features_dir,
        label_col='fb_instance',
        file_col='file',
    )
    
    # Init wandb
    if wandb_project_name is not None:
        wandb.init(project=wandb_project_name)
    
    set_seed(seed)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        eval_strategy=eval_save_strategy,
        save_strategy=eval_save_strategy,
        save_steps=save_steps,
        eval_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        report_to=report_to,
        seed=seed,
        lr_scheduler_type=lr_scheduler_type,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        save_total_limit=5,
        remove_unused_columns=False
    )
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()
    
    return trainer.model
        