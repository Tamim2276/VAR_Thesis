import os
import torch
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.dataset.vlm_dataset import RefereeVLMDataset
from src.models.vlm_model import RefereeVLM

def collate_fn(batch):
    visual_features = torch.stack([item['visual_features'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        "visual_features": visual_features,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def main():
    # To fix VRAM spilling, we lower physical batch size to 2, 
    # but accumulate gradients twice to keep the effective batch size at 4!
    BATCH_SIZE = 2 
    ACCUMULATION_STEPS = 2
    NUM_EPOCHS = 3
    LR = 2e-4
    
    print("Initializing Datasets...")
    train_dataset = RefereeVLMDataset(split="train")
    valid_dataset = RefereeVLMDataset(split="valid")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    print("\nInitializing Model...")
    model = RefereeVLM()
    # The LLM is already loaded to device_map="auto" by HuggingFace
    # We just need to move the projection layer to the LLM's device
    device = model.llm.device
    model.projection = model.projection.to(device)
    
    optimizer = torch.optim.AdamW(model.projection.parameters(), lr=LR)
    
    print("\nStarting Training...")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        
        # Zero gradients at the start of the epoch
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_pbar):
            visual_features = batch["visual_features"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(visual_features=visual_features, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            # Scale the loss down because we are accumulating over multiple steps
            loss = outputs.loss / ACCUMULATION_STEPS
            loss.backward()
            
            # Only step the optimizer and clear grads every 2 batches
            if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            
            # Multiply back to display the true loss value on the progress bar
            true_loss = loss.item() * ACCUMULATION_STEPS
            train_loss += true_loss
            train_pbar.set_postfix({"loss": f"{true_loss:.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(valid_loader, desc=f"Epoch {epoch} [Valid]")
        
        with torch.no_grad():
            for batch in val_pbar:
                visual_features = batch["visual_features"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(visual_features=visual_features, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(valid_loader)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Valid Loss: {avg_val_loss:.4f}")
        
        # Generation test on first valid sample
        print("\n--- Example Generation Test ---")
        test_batch = next(iter(valid_loader))
        test_vf = test_batch["visual_features"][0:1].to(device)
        
        # For generation, we just pass the user prompt without the assistant response
        prompt = (
            "<|user|>\n"
            "You are an expert VAR referee. Based on the video, explain the decision.\n"
            "<|end|>\n"
            "<|assistant|>\n"
        )
        test_input_ids = train_dataset.tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        
        # 1. Clear out the leftover training memory to prevent OOM crashes
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            torch.xpu.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 2. Tell PyTorch we are just testing, DO NOT track gradients!
        with torch.inference_mode():
            gen_outputs = model.generate(
                visual_features=test_vf, 
                input_ids=test_input_ids, 
                tokenizer=train_dataset.tokenizer,
                max_new_tokens=50  # Limit tokens to save memory
            )
            
        gen_text = train_dataset.tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
        print(gen_text)
        print("-------------------------------\n")
        
        os.makedirs("models/vlm", exist_ok=True)
        torch.save(model.projection.state_dict(), f"models/vlm/projection_ep{epoch}.pt")

if __name__ == "__main__":
    main()