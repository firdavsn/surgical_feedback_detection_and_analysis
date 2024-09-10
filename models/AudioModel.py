import torch
import torch.nn as nn
from transformers import Wav2Vec2ForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

class AudioModel(nn.Module):
    def __init__(self, params_model, device):
        super(AudioModel, self).__init__()

        self.device = device
        self.num_classes = params_model["num_classes"]
        class_weights = params_model["class_weights"]
        audio_model = params_model['audio_model']

        self.audio_model = None
        if audio_model != None:
            print("Initializing Audio Model!")
            self.audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(audio_model, 
                                                                                 num_labels=self.num_classes, 
                                                                                 ignore_mismatched_sizes=True)

        # Custom loss
        if class_weights != None:
            print(f"Using class weighting: {class_weights}")
            self.loss_func = nn.CrossEntropyLoss(reduction="mean", weight = torch.tensor(class_weights))
        else:
            print(f"No class weighting!")
            self.loss_func = nn.CrossEntropyLoss(reduction="mean")
    
    def forward(self, input_values, attention_mask, labels=None) -> SequenceClassifierOutput:
        self.audio_model.to(self.device)
        self.audio_model.eval()
        
        outputs = self.audio_model(input_values=input_values.to(self.device), attention_mask=attention_mask.to(self.device), labels=labels.to(self.device) if labels is not None else None)
        
        loss = None
        if labels is not None:
            loss = self.loss_func(outputs.logits.view(-1, self.num_classes), labels.view(-1))
            outputs.loss = loss
        
        return outputs

    def predict(self, input_ids, attention_mask):
        self.eval()
        with torch.no_grad():
            outputs = self.audio_model(input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device))
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
        return predictions