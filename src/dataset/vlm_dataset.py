import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.dataset.annotation_loader import load_annotations, FOUL_LABELS, SEVERITY_LABELS, ACTION_CLASS_LABELS

# Inverse mappings to get text back
INV_FOUL = {v: k for k, v in FOUL_LABELS.items()}
INV_SEV = {
    0: "No Card",
    1: "No Card", 
    2: "Yellow Card",
    3: "Red Card"
}
INV_ACTION = {v: k for k, v in ACTION_CLASS_LABELS.items()}

class RefereeVLMDataset(Dataset):
    def __init__(self, data_root="data", features_root="data/features", split="train", max_length=128):
        """
        Loads precomputed visual features and formats text for LLM causal training.
        """
        self.features_root = features_root
        self.max_length = max_length
        self.split = split
        
        print(f"Loading {split} split for VLM...")
        all_samples = load_annotations() # Loads all splits
        
        self.items = []
        missing = 0
        
        for sample in all_samples:
            if sample.get("split") != self.split:
                continue
                
            feat_path = (
                sample["clip_path"]
                .replace("data/frames",  features_root)
                .replace("data\\frames", features_root)
                .replace(".npy", ".pt")
                .replace("\\", "/")
            )
            if os.path.exists(feat_path):
                # Format the target text
                foul_text = "Foul" if sample['foul_label'] == 1 else "No Foul"
                sev_text = INV_SEV.get(sample['sev_label'], "No Card")
                action_text = INV_ACTION.get(sample['action_class'], "Unknown Action")
                
                if foul_text == "No Foul":
                    reasoning = f"The player did not commit an offence."
                    sev_text = "No Card"
                else:
                    reasoning = f"The player committed a foul (Action: {action_text})."

                prompt = (
                    "<|user|>\n"
                    "You are an expert VAR referee. Based on the video, explain the decision.\n"
                    "<|end|>\n"
                    "<|assistant|>\n"
                )
                
                response = (
                    f"Decision: {foul_text}\n"
                    f"Severity: {sev_text}\n"
                    f"Reasoning: {reasoning}<|end|>"
                )
                
                self.items.append({
                    "feat_path": feat_path,
                    "prompt": prompt,
                    "response": response,
                    "foul_label": sample['foul_label'],
                    "sev_label": sample['sev_label']
                })
            else:
                missing += 1
                
        print(f"Loaded {len(self.items)} samples for VLM (Missing features: {missing})")
        
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token
            
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        
        data = torch.load(item["feat_path"], map_location="cpu", weights_only=True)
        visual_features = data["cls"] # (16, 1152)
        
        prompt_tokens = self.tokenizer(
            item["prompt"], 
            add_special_tokens=False, 
            return_tensors="pt"
        )["input_ids"][0]
        
        response_tokens = self.tokenizer(
            item["response"], 
            add_special_tokens=False, 
            return_tensors="pt"
        )["input_ids"][0]
        
        input_ids = torch.cat([prompt_tokens, response_tokens])
        
        labels = torch.cat([
            torch.full_like(prompt_tokens, -100),  # Ignore prompt in loss
            response_tokens                        # Calculate loss on response
        ])
        
        attention_mask = torch.cat([
            torch.ones_like(prompt_tokens),
            torch.ones_like(response_tokens)
        ])
        
        # Truncate or Pad
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
        else:
            padding_len = self.max_length - len(input_ids)
            input_ids = F.pad(input_ids, (0, padding_len), value=self.tokenizer.pad_token_id)
            labels = F.pad(labels, (0, padding_len), value=-100)
            attention_mask = F.pad(attention_mask, (0, padding_len), value=0)
            
        return {
            "visual_features": visual_features,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "foul_label": item["foul_label"],
            "sev_label": item["sev_label"]
        }
