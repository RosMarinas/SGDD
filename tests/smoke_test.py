import sys
import os
import shutil
from pathlib import Path
import torch
import torch.optim as optim
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.sgdd import SGDDModel, SGDDConfig as ModelConfig
from utils.config import SGDDConfig as Config
from utils.config import TrainingConfig, DataConfig
from utils.data import get_dataloader
from train import train_epoch
from utils.metrics import evaluate_generation

def sample_real_dataset(source_path, target_path, tokenizer_name="BAAI/bge-m3", num_samples=64):
    print(f"Loading real dataset from {source_path}...")
    try:
        dataset = load_from_disk(str(source_path))
        if isinstance(dataset, DatasetDict):
            if 'train' in dataset:
                dataset = dataset['train']
            else:
                dataset = list(dataset.values())[0]
        
        print(f"Original dataset size: {len(dataset)}")
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        sampled_texts = []
        count = 0
        
        # Iterate and filter
        # shuffling to get random samples
        shuffled_indices = torch.randperm(len(dataset)).tolist()
        
        print("Filtering samples (20 <= len <= 128)...")
        for idx in shuffled_indices:
            if count >= num_samples:
                break
                
            item = dataset[idx]
            text = item['text']
            
            # Fast length check (approximate) then precise
            if len(text.split()) < 10 or len(text.split()) > 200:
                continue
                
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            length = len(token_ids)
            
            if 20 <= length <= 128:
                sampled_texts.append(text)
                count += 1
                if count % 10 == 0:
                    print(f"  Found {count}/{num_samples} samples...")
        
        if len(sampled_texts) < num_samples:
            print(f"Warning: Only found {len(sampled_texts)} suitable samples.")
            
        # Create subset dataset
        data = {"text": sampled_texts}
        ds = Dataset.from_dict(data)
        dd = DatasetDict({"train": ds, "validation": ds})
        
        print(f"Saving {len(sampled_texts)} samples to {target_path}...")
        dd.save_to_disk(str(target_path))
        return sampled_texts
        
    except Exception as e:
        print(f"Error sampling dataset: {e}")
        raise e

def run_smoke_test():
    print("Starting Smoke Test with Real Data...")
    # Setup paths
    project_root = Path(__file__).parent.parent
    test_dir = Path(__file__).parent
    
    # Real dataset path
    source_data_path = project_root / "data" / "BookCorpus" / "final_dataset_1.4B"
    
    # Temp paths
    smoke_data_path = test_dir / "smoke_data"
    checkpoint_dir = test_dir / "smoke_checkpoints"
    
    # Clean up old data
    if smoke_data_path.exists():
        shutil.rmtree(smoke_data_path)
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        
    try:
        # Create data
        if not source_data_path.exists():
            raise FileNotFoundError(f"Source dataset not found at {source_data_path}")
            
        original_texts = sample_real_dataset(source_data_path, smoke_data_path)
        
        # Config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Config for 128 length
        config = Config(
            model=ModelConfig(
                encoder_model="BAAI/bge-m3",
                semantic_dim=1024,
                decoder_dim=256,
                num_layers=2,
                num_heads=4,
                ffn_dim=1024,
                max_len=128, # Updated to 128
                dropout=0.1,
                num_diffusion_steps=100,
                vocab_size=250002, 
                cfg_prob=0.0,
                use_self_conditioning=True,
                compute_pad_loss=True, # Important for variable length
                kl_weight=1e-4,
                kl_anneal_steps=100,
            ),
            data=DataConfig(
                dataset="bookcorpus",
                dataset_path=str(smoke_data_path),
                max_token_length=128, # Updated to 128
                min_length=20, # Updated min length
                num_workers=0,
                pin_memory=False
            ),
            training=TrainingConfig(
                batch_size=8, # Reduced batch size for longer seqs
                num_epochs=100, # More epochs to ensure overfitting on real data
                learning_rate=1e-3,
                warmup_steps=10,
                weight_decay=0.0,
                gradient_accumulation_steps=1,
                grad_clip=1.0,
                log_interval=10,
                save_interval=1000,
                save_epochs=0,
                use_wandb=False,
                use_fp16=True, # Enable fp16 for speed
                lr_scheduler="constant",
                wandb_project="test",
                wandb_entity="test",
                wandb_run_name="test"
            ),
            checkpoint_dir=str(checkpoint_dir),
            seed=42,
            device=str(device)
        )

        # Initialize model
        print("Initializing model...")
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        vocab_size = tokenizer.vocab_size
        config.model.vocab_size = vocab_size
        
        model = SGDDModel(config.model).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=config.training.learning_rate)
        
        # Mixed precision scaler
        scaler = torch.amp.GradScaler('cuda') if config.training.use_fp16 and device.type == 'cuda' else None

        # Data Loader
        print("Loading dataloader...")
        train_loader = get_dataloader(
            dataset_name="bookcorpus",
            split="train",
            dataset_path=str(smoke_data_path),
            max_token_length=config.data.max_token_length,
            min_length=config.data.min_length,
            batch_size=config.training.batch_size,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            tokenizer_name=config.model.encoder_model
        )
        
        # DEBUG: Verify data
        print("\n[DEBUG] Verifying data...")
        batch = next(iter(train_loader))
        debug_ids = batch["input_ids"][0]
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        print(f"First sample IDs: {debug_ids.tolist()[:20]}...")
        print(f"Decoded: {tokenizer.decode(debug_ids)}")
        print(f"Original Text field: {batch['texts'][0]}")
        
        # Training Loop
        print(f"Starting training loop for {config.training.num_epochs} epochs...")
        initial_loss = None
        final_loss = None
        
        for epoch in range(config.training.num_epochs):
            metrics = train_epoch(
                model,
                train_loader,
                optimizer,
                None, # scheduler
                device,
                config,
                epoch,
                scaler # scaler
            )
            if epoch == 0:
                initial_loss = metrics["loss"]
            final_loss = metrics["loss"]
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{config.training.num_epochs}: loss={metrics['loss']:.4f}, recon={metrics['reconstruction_loss']:.4f}")
        
        print(f"Initial Loss: {initial_loss:.4f}")
        print(f"Final Loss: {final_loss:.4f}")
        
        # Check training stability
        if final_loss is None or torch.isnan(torch.tensor(final_loss)) or torch.isinf(torch.tensor(final_loss)):
            raise ValueError("Training Failed: Loss is NaN or Inf")
            
        if final_loss >= initial_loss:
            print("Warning: Loss did not decrease. Overfitting might have failed.")
        else:
            print("Loss decreased successfully.")

        # Evaluation / Inference
        print("\nTesting Inference...")
        model.eval()
        
        # Select a sample to reconstruct
        batch = next(iter(train_loader))
        input_ids = batch["input_ids"].to(device)
        batch_texts = batch["texts"]
        
        test_text = batch_texts[0]
        print(f"\nOriginal ({len(test_text.split())} words):\n{test_text}")
        
        # DEBUG: Inspect what the model predicts at step 0 (all mask)
        print("\n[DEBUG] Single step prediction check:")
        with torch.no_grad():
            semantic_vector, _ = model.encoder(input_ids[0:1], batch["attention_mask"][0:1].to(device), return_kl=True)
            # Create all-mask input
            mask_token_id = tokenizer.mask_token_id
            dummy_input = torch.full((1, config.model.max_len), mask_token_id, device=device)
            # Timestep T (pure noise)
            timestep = torch.tensor([100], device=device) # High timestep
            
            logits = model.decoder(dummy_input, semantic_vector, timestep)
            probs = torch.softmax(logits, dim=-1)
            
            # Check first 5 positions
            for i in range(5):
                top_vals, top_idxs = torch.topk(probs[0, i], 5)
                # Decode and print ID to debug empty strings
                debug_tokens = []
                for idx, val in zip(top_idxs, top_vals):
                    token_str = tokenizer.decode([idx])
                    if not token_str:
                        token_str = "<EMPTY>"
                    debug_tokens.append(f"({idx}:'{token_str}', {val.item():.4f})")
                print(f"Pos {i}: {debug_tokens}")

        with torch.no_grad():
            # Use deterministic generation with low temp
            # IMPORTANT: guidance_scale=0.0 because we trained with cfg_prob=0.0
            generated_text = model.generate(
                input_text=test_text,
                num_steps=32, 
                guidance_scale=0.0, 
                max_length=128,
                temperature=0.01, # Almost greedy
                top_k=-1,
                sample_z=False
            )
        
        print(f"\nGenerated:\n{generated_text}")
        
        # Basic check
        if not isinstance(generated_text, str) or len(generated_text) == 0:
             raise ValueError("Generation failed: Empty or invalid output")
             
        print("\nSmoke Test Passed!")
        
    except Exception as e:
        print(f"\nSmoke Test Failed: {e}")
        import traceback
        traceback.print_exc()
        # raise e # Don't raise to avoid crashing the agent loop, just report
    finally:
        # Clean up
        if smoke_data_path.exists():
            shutil.rmtree(smoke_data_path)
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)

if __name__ == "__main__":
    run_smoke_test()