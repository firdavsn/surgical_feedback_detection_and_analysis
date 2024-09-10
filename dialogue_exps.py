from utils import *
import torch
from models import ExtractDialogueModel
from transcribe import whisper_transcribe
from utils import set_openai_key
torch.cuda.empty_cache()

def get_metrics(case_id, aux=None):
    openai_key_path = 'openai_api_key.txt'

    if case_id not in [4, 11]:
        full_audio_path = f'../../full_audios/LFB{case_id}_full.wav'
    else:
        full_audio_path = f'../../full_audios/LFB{case_id}_full_1.wav'
    vad_activity_path = f'../../full_VADs/LFB{case_id}_full_activity.csv'
    diarization_save_path = f'results/extract_dialogue/diarizations/LFB{case_id}_full.csv'
    transcriptions_save_path = f'results/extract_dialogue/transcriptions/LFB{case_id}_full.csv'
    identification_save_path = f'results/extract_dialogue/identifications/LFB{case_id}_full.csv'
    fb_detection_save_path = f'results/extract_dialogue/fb_detection/LFB{case_id}_full{" " + aux if aux else ""}.csv'
    aligned_fb_detection_save_path = f'results/extract_dialogue/aligned_fb_detection/LFB{case_id}_full{" " + aux if aux else ""}.csv'
    behavior_prediction_save_path = f'results/extract_dialogue/behavior_prediction/LFB{case_id}_full{" " + aux if aux else ""} clustering_defintion.csv'
    component_classification_save_path = f'results/extract_dialogue/component_classification/LFB{case_id}_full{" " + aux if aux else ""}.csv'
    
    device = torch.device("cuda")
    params_extract_dialogue = {
        'speaker_diarization_model': 'pyannote/speaker-diarization-3.1',
        'speaker_embedding_model': 'pyannote/embedding',
        'hf_token_path': 'huggingface_token.txt',
        'openai_key_path': openai_key_path, 
        'transcribe_fn': whisper_transcribe,
        'full_audio_path': full_audio_path,
        'interval': 180,
        'console_times_path': '../../annotations/console_times/combined_console_times_secs.csv',
        'fb_annot_path': '../../clips_no_wiggle/fbk_cuts_no_wiggle_0_4210.csv',
        'vad_activity_path': vad_activity_path,
        'diarizations_save_path': diarization_save_path,
        'transcriptions_save_path': transcriptions_save_path,
        'identifications_save_path': identification_save_path,
        'fb_detection_save_path': fb_detection_save_path,
        'aligned_fb_detection_save_path': aligned_fb_detection_save_path,
        'behavior_prediction_save_path': behavior_prediction_save_path,
        'component_classification_save_path': component_classification_save_path,
        'audio_clips_dir': 'results/extract_dialogue/audio_clips',
        'trainer_anchors_dir': 'results/extract_dialogue/anchors/trainer',
        'trainee_anchors_dir': 'results/extract_dialogue/anchors/trainee',
        'rag_embeddings_dir': 'results/extract_dialogue/rag_embeddings/context+phrase',
        'tmp_dir': 'tmp',
        'seed': 42,
        'min_n_speakers': 2,
        'max_n_speakers': 2,
        'embedding_dist_thresh': 0.8
    }
    set_openai_key(openai_key_path)
    print(params_extract_dialogue)
    
    model = ExtractDialogueModel(params_extract_dialogue, device)
    print("Extracting dialogue...")
    print("Diarization...")
    model.full_diarization(load_saved=True)
    print("Transcription...")
    model.full_transcription(load_saved=True)
    print("Identification...")
    model.full_identification(load_saved=True)
    
    print("Detecting feedback...")
    aux = None if aux not in ["'dialogue'", "'reduced hallucinations'", "'temporal context'"] else aux[1:-1]
    model.full_fb_detection(context_len=5, load_saved=True, aux=aux)
    model.full_aligned_fb_detection(load_saved=True)
    
    print("Predicting trainee behavior...")
    model.full_behavior_prediction(load_saved=True, use_human_annotations=False, use_clustering_defintion=True, verbose=True)
    
    print("Classifying components...")
    model.full_component_classification(load_saved=True, use_huamn_annotations=False, verbose=False)
    
    metrics_dict = {
        'binary_fb': model.evaluate(weighting='binary', model_type='fb'),
        'binary_behavior': model.evaluate(weighting='binary', model_type='behavior'),
        'binary_component': model.evaluate(weighting='binary', model_type='component'),
        'weighted_fb': model.evaluate(weighting='weighted', model_type='fb'),
        'weighted_behavior': model.evaluate(weighting='weighted', model_type='behavior'),
        'weighted_component': model.evaluate(weighting='weighted', model_type='component')
    }
    
    return metrics_dict

def main():
    fb_metrics_df = pd.DataFrame(columns=['Case', 'Model Type', 'Binary/Weighted', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUROC'])
    behavior_metrics_df = pd.DataFrame(columns=['Case', 'Model Type', 'Binary/Weighted', 
                                                'Accuracy r_t_verb', 'Precision r_t_verb', 'Recall r_t_verb', 'F1 r_t_verb', 'AUROC r_t_verb',
                                                'Accuracy r_t_beh', 'Precision r_t_beh', 'Recall r_t_beh', 'F1 r_t_beh', 'AUROC r_t_beh',
                                                'Accuracy r_t_clarify', 'Precision r_t_clarify', 'Recall r_t_clarify', 'F1 r_t_clarify', 'AUROC r_t_clarify',])
    component_metrics_df = pd.DataFrame(columns=['Case', 'Model Type', 'Binary/Weighted',
                                                 'Accuracy f_anatomic', 'Precision f_anatomic', 'Recall f_anatomic', 'F1 f_anatomic', 'AUROC f_anatomic',
                                                 'Accuracy f_procedural', 'Precision f_procedural', 'Recall f_procedural', 'F1 f_procedural', 'AUROC f_procedural',
                                                 'Accuracy f_technical', 'Precision f_technical', 'Recall f_technical', 'F1 f_technical', 'AUROC f_technical',])
    aux = "'all phrases'"
    # the original 5
    for case_id in [1, 2, 9, 10, 18]:
    
    # all identifiable (with anchors)
    # for case_id in [1, 2, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 28, 29, 33]:
        
    # After removing outliers
    # for case_id in [1, 2, 8, 9, 10, 12, 13, 16, 17, 18, 20, 21, 22, 25, 26, 28, 29, 33]:
    
    # dialogue
    # for case_id in [10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 28, 29, 33]:
    
    # reduced hallucinations
    # for case_id in [18, 19, 20, 21, 22, 23, 25, 26, 28, 29, 33]:
    
    # temporal context
    # for case_id in [1, 2, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 28, 29, 33]: 
    
        print(f"Case {case_id}")
        try: 
            metrics = get_metrics(case_id, aux=aux)
            print(metrics)
            
            fb_metrics_df.loc[len(fb_metrics_df)] = pd.Series({
                'Case': case_id,
                'Model Type': 'Feedback Detection',
                'Binary/Weighted': 'Binary',
                'Accuracy': metrics['binary_fb']['accuracy'],
                'Precision': metrics['binary_fb']['precision'],
                'Recall': metrics['binary_fb']['recall'],
                'F1': metrics['binary_fb']['f1'],
                'AUROC': metrics['binary_fb']['roc_auc']
            })
            fb_metrics_df.loc[len(fb_metrics_df)] = pd.Series({
                'Case': case_id,
                'Model Type': 'Feedback Detection',
                'Binary/Weighted': 'Weighted',
                'Accuracy': metrics['weighted_fb']['accuracy'],
                'Precision': metrics['weighted_fb']['precision'],
                'Recall': metrics['weighted_fb']['recall'],
                'F1': metrics['weighted_fb']['f1'],
                'AUROC': metrics['weighted_fb']['roc_auc']
            })
            
            
            behavior_metrics_df.loc[len(behavior_metrics_df)] = pd.Series({
                'Case': case_id,
                'Model Type': 'Trainee Behavior Prediction',
                'Binary/Weighted': 'Binary',
                
                'Accuracy r_t_verb': metrics['binary_behavior']['accuracy_r_t_verb'],
                'Precision r_t_verb': metrics['binary_behavior']['precision_r_t_verb'],
                'Recall r_t_verb': metrics['binary_behavior']['recall_r_t_verb'],
                'F1 r_t_verb': metrics['binary_behavior']['f1_r_t_verb'],
                'AUROC r_t_verb': metrics['binary_behavior']['roc_auc_r_t_verb'],
                
                'Accuracy r_t_beh': metrics['binary_behavior']['accuracy_r_t_beh'],
                'Precision r_t_beh': metrics['binary_behavior']['precision_r_t_beh'],
                'Recall r_t_beh': metrics['binary_behavior']['recall_r_t_beh'],
                'F1 r_t_beh': metrics['binary_behavior']['f1_r_t_beh'],
                'AUROC r_t_beh': metrics['binary_behavior']['roc_auc_r_t_beh'],
                
                'Accuracy r_t_clarify': metrics['binary_behavior']['accuracy_r_t_clarify'],
                'Precision r_t_clarify': metrics['binary_behavior']['precision_r_t_clarify'],
                'Recall r_t_clarify': metrics['binary_behavior']['recall_r_t_clarify'],
                'F1 r_t_clarify': metrics['binary_behavior']['f1_r_t_clarify'],
                'AUROC r_t_clarify': metrics['binary_behavior']['roc_auc_r_t_clarify'],
            })
            behavior_metrics_df.loc[len(behavior_metrics_df)] = pd.Series({
                'Case': case_id,
                'Model Type': 'Trainee Behavior Prediction',
                'Binary/Weighted': 'Weighted',
                
                'Accuracy r_t_verb': metrics['weighted_behavior']['accuracy_r_t_verb'],
                'Precision r_t_verb': metrics['weighted_behavior']['precision_r_t_verb'],
                'Recall r_t_verb': metrics['weighted_behavior']['recall_r_t_verb'],
                'F1 r_t_verb': metrics['weighted_behavior']['f1_r_t_verb'],
                'AUROC r_t_verb': metrics['weighted_behavior']['roc_auc_r_t_verb'],
                
                'Accuracy r_t_beh': metrics['weighted_behavior']['accuracy_r_t_beh'],
                'Precision r_t_beh': metrics['weighted_behavior']['precision_r_t_beh'],
                'Recall r_t_beh': metrics['weighted_behavior']['recall_r_t_beh'],
                'F1 r_t_beh': metrics['weighted_behavior']['f1_r_t_beh'],
                'AUROC r_t_beh': metrics['weighted_behavior']['roc_auc_r_t_beh'],
                
                'Accuracy r_t_clarify': metrics['weighted_behavior']['accuracy_r_t_clarify'],
                'Precision r_t_clarify': metrics['weighted_behavior']['precision_r_t_clarify'],
                'Recall r_t_clarify': metrics['weighted_behavior']['recall_r_t_clarify'],
                'F1 r_t_clarify': metrics['weighted_behavior']['f1_r_t_clarify'],
                'AUROC r_t_clarify': metrics['weighted_behavior']['roc_auc_r_t_clarify'],
            })

            component_metrics_df.loc[len(component_metrics_df)] = pd.Series({
                'Case': case_id,
                'Model Type': 'Component Prediction',
                'Binary/Weighted': 'Binary',
                
                'Accuracy f_anatomic': metrics['binary_component']['accuracy_f_anatomic'],
                'Precision f_anatomic': metrics['binary_component']['precision_f_anatomic'],
                'Recall f_anatomic': metrics['binary_component']['recall_f_anatomic'],
                'F1 f_anatomic': metrics['binary_component']['f1_f_anatomic'],
                'AUROC f_anatomic': metrics['binary_component']['roc_auc_f_anatomic'],
                
                'Accuracy f_procedural': metrics['binary_component']['accuracy_f_procedural'],
                'Precision f_procedural': metrics['binary_component']['precision_f_procedural'],
                'Recall f_procedural': metrics['binary_component']['recall_f_procedural'],
                'F1 f_procedural': metrics['binary_component']['f1_f_procedural'],
                'AUROC f_procedural': metrics['binary_component']['roc_auc_f_procedural'],
                
                'Accuracy f_technical': metrics['binary_component']['accuracy_f_technical'],
                'Precision f_technical': metrics['binary_component']['precision_f_technical'],
                'Recall f_technical': metrics['binary_component']['recall_f_technical'],
                'F1 f_technical': metrics['binary_component']['f1_f_technical'],
                'AUROC f_technical': metrics['binary_component']['roc_auc_f_technical'],
            })
            
            component_metrics_df.loc[len(component_metrics_df)] = pd.Series({
                'Case': case_id,
                'Model Type': 'Component Prediction',
                'Binary/Weighted': 'Weighted',
                
                'Accuracy f_anatomic': metrics['weighted_component']['accuracy_f_anatomic'],
                'Precision f_anatomic': metrics['weighted_component']['precision_f_anatomic'],
                'Recall f_anatomic': metrics['weighted_component']['recall_f_anatomic'],
                'F1 f_anatomic': metrics['weighted_component']['f1_f_anatomic'],
                'AUROC f_anatomic': metrics['weighted_component']['roc_auc_f_anatomic'],
                
                'Accuracy f_procedural': metrics['weighted_component']['accuracy_f_procedural'],
                'Precision f_procedural': metrics['weighted_component']['precision_f_procedural'],
                'Recall f_procedural': metrics['weighted_component']['recall_f_procedural'],
                'F1 f_procedural': metrics['weighted_component']['f1_f_procedural'],
                'AUROC f_procedural': metrics['weighted_component']['roc_auc_f_procedural'],
                
                'Accuracy f_technical': metrics['weighted_component']['accuracy_f_technical'],
                'Precision f_technical': metrics['weighted_component']['precision_f_technical'],
                'Recall f_technical': metrics['weighted_component']['recall_f_technical'],
                'F1 f_technical': metrics['weighted_component']['f1_f_technical'],
                'AUROC f_technical': metrics['weighted_component']['roc_auc_f_technical'],
            })
            
        except Exception as e:
            print(f"An error occurred: {e}.")
        
        print("=========================================")
        print("\n\n")

        fb_metrics_df.to_csv(f'results/extract_dialogue/metrics/fb {aux}.csv', index=False)
        behavior_metrics_df.to_csv(f'results/extract_dialogue/metrics/behavior {aux} clustering_definition.csv', index=False)
        component_metrics_df.to_csv(f'results/extract_dialogue/metrics/component {aux}.csv', index=False)

if __name__ == '__main__':
    main()