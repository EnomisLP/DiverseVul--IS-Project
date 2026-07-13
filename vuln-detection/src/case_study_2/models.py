import torch
import torch.nn as nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model, PeftModel
import pyreft

class NeoBertSequenceClassifier(nn.Module):
    def __init__(self, model_name="Arize-ai/NeoBERT-250M", freeze_backbone=True):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        hidden_size = self.backbone.config.hidden_size
        self.classification_head = nn.Linear(hidden_size, 1)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        pooled_output = hidden_states[:, 0, :]
        logits = self.classification_head(pooled_output)
        return logits.squeeze(-1)

def get_exp4_lora_model(model_name="Arize-ai/NeoBERT-250M", rank=8, lora_alpha=16):
    base_classifier = NeoBertSequenceClassifier(model_name=model_name, freeze_backbone=False)
    peft_config = LoraConfig(
        task_type="SEQ_CLS",
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    lora_model = get_peft_model(base_classifier, peft_config)
    return lora_model

def get_exp5_heft_model(lora_model_path, model_name="Arize-ai/NeoBERT-250M", reft_rank=4):
    base_classifier = NeoBertSequenceClassifier(model_name=model_name, freeze_backbone=False)
    lora_model = PeftModel.from_pretrained(base_classifier, lora_model_path)
    
    for param in lora_model.parameters():
        param.requires_grad = False
        
    reft_config = pyreft.ReftConfig(
        intervention_type=pyreft.PositionControlledIntervention,
        embed_dim=lora_model.config.hidden_size,
        low_rank_dimension=reft_rank,
        intervened_layers=[12, 16, 20, 24],
        positions="prefix", 
        num_prefix_tokens=2
    )
    reft_model = pyreft.get_reft_model(lora_model, reft_config)
    return reft_model