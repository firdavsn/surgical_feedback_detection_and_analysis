import torch
import torch.nn as nn
from transformers import BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

class TextModel(nn.Module):
    def __init__(self, params_model, device):
        super(TextModel, self).__init__()

        self.device = device
        self.num_classes = params_model["num_classes"]
        class_weights = params_model["class_weights"]
        text_model = params_model['text_model']

        self.text_model = None
        if text_model != None:
            print("Initializing Text Model!")
            self.text_model = BertForSequenceClassification.from_pretrained(text_model,
                                                                            num_labels=self.num_classes,
                                                                            output_attentions=False,
                                                                            output_hidden_states=False,
                                                                            )

        # Custom loss
        if class_weights != None:
            print(f"Using class weighting: {class_weights}")
            self.loss_func = nn.CrossEntropyLoss(reduction="mean", weight = torch.tensor(class_weights))
        else:
            print(f"No class weighting!")
            self.loss_func = nn.CrossEntropyLoss(reduction="mean")
    
    def forward(self, input_ids, attention_mask, token_type_ids, labels=None) -> SequenceClassifierOutput:
        self.text_model.to(self.device)
        self.text_model.eval()
        
        outputs = self.text_model(input_ids=input_ids.to(self.device), token_type_ids=token_type_ids.to(self.device), attention_mask=attention_mask.to(self.device), labels=labels.to(self.device) if labels is not None else None)
        
        loss = None
        if labels is not None:
            loss = self.loss_func(outputs.logits.view(-1, self.num_classes), labels.view(-1))
            outputs.loss = loss
            
        return outputs

    def predict(self, input_ids, attention_mask):
        self.eval()
        with torch.no_grad():
            outputs = self.text_model(input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device))
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
        return predictions