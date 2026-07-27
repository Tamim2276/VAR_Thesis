import os
import sys
import json
import math
import torch
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
    SAVE_DIR = "models/vlm"
    LOG_DIR = "logs"
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    print("Initializing Datasets...")
    train_dataset = RefereeVLMDataset(split="train")
    valid_dataset = RefereeVLMDataset(split="valid")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    print(f"\n  Train samples: {len(train_dataset)} | Valid samples: {len(valid_dataset)}")
    print(f"  Train batches: {len(train_loader)} | Valid batches: {len(valid_loader)}")
    print(f"  Effective batch size: {BATCH_SIZE * ACCUMULATION_STEPS} (physical={BATCH_SIZE} x accumulation={ACCUMULATION_STEPS})")
    
    print("\nInitializing Model...")
    model = RefereeVLM()
    # The LLM is already loaded to device_map="auto" by HuggingFace
    # We just need to move the projection layer to the LLM's device
    device = model.llm.device
    model.projection = model.projection.to(device)
    
    # Count trainable params (only the projection layer)
    trainable_params = sum(p.numel() for p in model.projection.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable_params:,} / {total_params:,} total")
    
    optimizer = torch.optim.AdamW(model.projection.parameters(), lr=LR, weight_decay=1e-4)
    
    # Cosine annealing scheduler for smoother convergence
    total_steps = len(train_loader) * NUM_EPOCHS // ACCUMULATION_STEPS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    
    print(f"\n  Config: LR={LR} | WeightDecay=1e-4 | Scheduler=CosineAnnealing")
    print("=" * 70)
    print("Starting Training...\n")
    
    history = []
    best_val_loss = float("inf")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # ── Training ──
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
            
            # Only step the optimizer and clear grads every ACCUMULATION_STEPS batches
            if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.projection.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Multiply back to display the true loss value on the progress bar
            true_loss = loss.item() * ACCUMULATION_STEPS
            train_loss += true_loss
            train_pbar.set_postfix({"loss": f"{true_loss:.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        train_perplexity = math.exp(min(avg_train_loss, 20))  # Cap to avoid overflow
        
        # ── Validation ──
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
                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                
        avg_val_loss = val_loss / len(valid_loader)
        val_perplexity = math.exp(min(avg_val_loss, 20))  # Cap to avoid overflow
        current_lr = optimizer.param_groups[0]['lr']
        
        # ── Print BOTH train and validation side by side ──
        print(f"\n  Epoch {epoch}/{NUM_EPOCHS} Summary:")
        print(f"  Train — loss: {avg_train_loss:.4f} | perplexity: {train_perplexity:.2f}")
        print(f"  Valid — loss: {avg_val_loss:.4f} | perplexity: {val_perplexity:.2f}")
        print(f"  LR: {current_lr:.6f}")
        
        # Save history
        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_perplexity": train_perplexity,
            "val_loss": avg_val_loss,
            "val_perplexity": val_perplexity,
            "lr": current_lr,
        })
        
        # ── Save best model based on validation loss ──
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.projection.state_dict(), f"{SAVE_DIR}/best_projection.pt")
            print(f"  ✓ Saved new best VLM Projection (val_loss: {best_val_loss:.4f})")
        
        # Always save per-epoch checkpoint too
        torch.save(model.projection.state_dict(), f"{SAVE_DIR}/projection_ep{epoch}.pt")
        
        # ── Generation Test ──
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
        
        # Clear out the leftover training memory to prevent OOM crashes
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            torch.xpu.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Tell PyTorch we are just testing, DO NOT track gradients!
        with torch.inference_mode():
            gen_outputs = model.generate(
                visual_features=test_vf, 
                input_ids=test_input_ids, 
                tokenizer=train_dataset.tokenizer,
                max_new_tokens=50
            )
            
        gen_text = train_dataset.tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
        print(gen_text)
        print("-------------------------------\n")

    # Save training history for plotting later
    with open(f"{LOG_DIR}/vlm_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("=" * 70)
    print("VLM TRAINING COMPLETE")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Best model saved to: {SAVE_DIR}/best_projection.pt")
    print(f"  History saved to: {LOG_DIR}/vlm_training_history.json")
    print("=" * 70)

if __name__ == "__main__":
    main()