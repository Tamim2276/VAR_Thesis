import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig

class RefereeVLM(nn.Module):
    def __init__(self, visual_dim=1152, llm_name="microsoft/Phi-3-mini-4k-instruct"):
        super().__init__()
        print(f"Loading LLM: {llm_name}...")
        
        # Load the LLM in bfloat16 to save memory
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto" # Automatically uses GPU if available
        )
        
        # Freeze the LLM entirely
        for param in self.llm.parameters():
            param.requires_grad = False
            
        # Get LLM hidden size (Phi-3-mini is 3072)
        self.llm_hidden_size = self.llm.config.hidden_size
        
        print(f"Building Projection Layer ({visual_dim} -> {self.llm_hidden_size})")
        # Trainable projection layer
        self.projection = nn.Sequential(
            nn.Linear(visual_dim, self.llm_hidden_size),
            nn.GELU(),
            nn.Linear(self.llm_hidden_size, self.llm_hidden_size)
        ).to(torch.bfloat16) # Match LLM dtype
        
    def forward(self, visual_features, input_ids, attention_mask=None, labels=None):
        """
        visual_features: (Batch, T, 1152) — T is typically 16 frames
        input_ids: (Batch, SeqLen)
        attention_mask: (Batch, SeqLen)
        labels: (Batch, SeqLen)
        """
        # Ensure visual features match projection layer dtype
        visual_features = visual_features.to(self.projection[0].weight.dtype)
        
        # 1. Project visual features to LLM embedding space -> (Batch, T, 3072)
        visual_embeds = self.projection(visual_features)
        
        # Dynamically read the number of visual tokens from the tensor itself
        num_visual_tokens = visual_embeds.size(1)
        
        # 2. Get text embeddings from the frozen LLM -> (Batch, SeqLen, 3072)
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # 3. Concatenate visual embeddings BEFORE text embeddings
        # Shape: (Batch, T + SeqLen, 3072)
        combined_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        
        # Combine attention masks
        if attention_mask is not None:
            batch_size = attention_mask.size(0)
            visual_mask = torch.ones((batch_size, num_visual_tokens), dtype=attention_mask.dtype, device=attention_mask.device)
            combined_mask = torch.cat([visual_mask, attention_mask], dim=1)
        else:
            combined_mask = None
            
        if labels is not None:
            # Prepend -100 to labels for the visual tokens so loss isn't computed on them
            batch_size = labels.size(0)
            visual_labels = torch.full((batch_size, num_visual_tokens), -100, dtype=labels.dtype, device=labels.device)
            combined_labels = torch.cat([visual_labels, labels], dim=1)
            
            # 4. Forward pass through LLM
            outputs = self.llm(inputs_embeds=combined_embeds, attention_mask=combined_mask, labels=combined_labels)
            return outputs
        else:
            # Inference mode
            outputs = self.llm(inputs_embeds=combined_embeds, attention_mask=combined_mask)
            return outputs
            
    def generate(self, visual_features, input_ids, attention_mask=None, 
                 max_new_tokens=50, tokenizer=None, temperature=1.0):
        """
        Helper for generation during evaluation.
        temperature: Controls randomness. 1.0 = normal, <1.0 = more confident, >1.0 = more creative
        """
        visual_features = visual_features.to(self.projection[0].weight.dtype)
        visual_embeds = self.projection(visual_features)
        
        # Dynamically read visual token count
        num_visual_tokens = visual_embeds.size(1)
        
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        combined_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        batch_size = attention_mask.size(0)
        visual_mask = torch.ones((batch_size, num_visual_tokens), dtype=attention_mask.dtype, device=attention_mask.device)
        combined_mask = torch.cat([visual_mask, attention_mask], dim=1)
        
        outputs = self.llm.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id if tokenizer else self.llm.config.pad_token_id,
            eos_token_id=tokenizer.eos_token_id if tokenizer else self.llm.config.eos_token_id,
            do_sample=False, # greedy for factual extraction
            temperature=temperature,
        )
        return outputs