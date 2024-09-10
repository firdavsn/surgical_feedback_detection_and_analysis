import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput

from .TextModel import TextModel
from .AudioModel import AudioModel

class AudioTextFusionModel(nn.Module):
    def __init__(self, params_model, device):
        super(AudioTextFusionModel, self).__init__()

        self.device = device
        num_features = params_model['num_features']
        num_classes = params_model["num_classes"]
        class_weights = params_model["class_weights"]
        
        text_params = {
            'text_model': params_model['text_model'],
            'num_classes': num_features,
            'class_weights': class_weights,
        }
        audio_params = {
            'audio_model': params_model['audio_model'],
            'num_classes': num_features,
            'class_weights': class_weights
        }
        
        self.text_model = TextModel(text_params, device)
        self.audio_model = AudioModel(audio_params, device)

        # Custom loss
        if class_weights != None:
            print(f"Using class weighting: {class_weights}")
            self.loss_func = nn.CrossEntropyLoss(reduction="mean", weight = torch.tensor(class_weights))
        else:
            print(f"No class weighting!")
            self.loss_func = nn.CrossEntropyLoss(reduction="mean")

        self.fc1 = nn.Linear(num_features * 2, (num_features//8) * 2)  
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear((num_features//8) * 2, num_classes)
        
    def forward(self, 
                # text inputs
                text_input_ids,
                text_attention_mask,
                text_token_type_ids,
                
                # audio inputs
                audio_input_values,
                audio_attention_mask,
                
                # output
                labels=None):
        self.fc1.to(self.device)
        self.fc2.to(self.device)
        
        text_outputs = self.text_model(
            input_ids=text_input_ids.to(self.device),
            attention_mask=text_attention_mask.to(self.device),
            token_type_ids=text_token_type_ids.to(self.device),
            labels=None
        )
        audio_outputs = self.audio_model(
            input_values=audio_input_values.to(self.device),
            attention_mask=audio_attention_mask.to(self.device),
            labels=None
        )
        
        text_logits = text_outputs.logits
        audio_logits = audio_outputs.logits
        
        ccat_logits = torch.cat((text_logits, audio_logits), 1)
        logits = F.relu(self.fc1(ccat_logits))
        logits = self.dropout(self.fc2(logits))
        
        loss = None
        if labels is not None:
            loss = self.loss_func(logits, labels)
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )