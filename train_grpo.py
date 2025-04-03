from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import torch
import gc
import os
import shutil
from transformers import logging, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import time

# Check TRL version
import trl
print(f"TRL version: {trl.__version__}")

# Enable more detailed logging
logging.set_verbosity_info()

def clear_memory():
    """Clear GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        with torch.cuda.device('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

def clear_previous_run():
    """Clear previous training outputs"""
    output_dir = "phi2-grpo-output"
    final_model = "phi2-grpo-final"
    
    if os.path.exists(output_dir):
        print(f"Removing previous output directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    if os.path.exists(final_model):
        print(f"Removing previous model: {final_model}")
        shutil.rmtree(final_model)

def reward_len(completions, **kwargs):
    """A simple reward function that rewards completions close to 150 characters."""
    return [-abs(150 - len(completion)) for completion in completions]

def print_gpu_utilization():
    """Print GPU memory usage"""
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

def prepare_tokenizer_and_model():
    """Create a phi-2 tokenizer with padding token and model together"""
    print("Setting up phi-2 tokenizer with padding token...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    
    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Print tokenizer info
    print(f"Padding token: {tokenizer.pad_token}")
    print(f"Padding token ID: {tokenizer.pad_token_id}")
    print(f"Padding side: {tokenizer.padding_side}")
    
    # Load model with special token handling
    print("\nLoading model and applying LoRA...")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        torch_dtype=torch.float16,
        device_map="auto",
        pad_token_id=tokenizer.pad_token_id  # Set pad token ID during model loading
    )
    
    # Make sure model config has padding token
    model.config.pad_token_id = tokenizer.pad_token_id
    print(f"Model pad token ID set to: {model.config.pad_token_id}")
    
    # Configure LoRA (Parameter-Efficient Fine-Tuning)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["fc1", "fc2"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    print(f"Trainable parameters: {model.print_trainable_parameters()}")
    
    # Create output directory and save tokenizer for later reference
    os.makedirs("phi2-grpo-output", exist_ok=True)
    tokenizer.save_pretrained("phi2-grpo-output")
    
    return model, tokenizer

def main():
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU.")
    else:
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    start_time = time.time()
    
    # Clear previous run and memory
    clear_previous_run()
    clear_memory()

    print("Loading dataset...")
    dataset = load_dataset("trl-lib/tldr", split="train")
    # Using a smaller portion of the dataset to fit in GPU memory
    dataset = dataset.select(range(min(5000, len(dataset))))
    print(f"Dataset size: {len(dataset)} examples")
    print(f"Example: {dataset[0]}")

    print("\nInitial GPU state:")
    print_gpu_utilization()
    
    # Create and save tokenizer with padding token
    model, tokenizer = prepare_tokenizer_and_model()

    # Training configuration with lower memory usage
    print("\nSetting up training configuration...")
    training_args = GRPOConfig(
        output_dir="phi2-grpo-output",
        num_train_epochs=1,  # Single epoch for testing
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        logging_steps=5,
        num_generations=2,
        max_prompt_length=128,
        max_completion_length=64,
        fp16=True,
        gradient_checkpointing=True,
        report_to="none",
        log_level="info"
    )

    print("\nInitializing trainer...")
    print("This may take a few minutes...")
    
    # Try a different approach - patch the model to return the tokenizer
    model.get_tokenizer = lambda: tokenizer
    model.tokenizer = tokenizer  # Add tokenizer as attribute
    
    # Initialize the trainer with the prepared model
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_len,
        args=training_args,
        train_dataset=dataset
    )

    print("\nGPU state after trainer initialization:")
    print_gpu_utilization()
    
    print("\nStarting training...")
    print("Time since start:", time.time() - start_time, "seconds")
    try:
        # Try monkey-patching the token before training starts
        # This ensures any internal tokenizer creation uses our prepared tokenizer
        if hasattr(trainer, 'tokenizer'):
            trainer.tokenizer = tokenizer
        
        # Apply the pad token to the internal tokenizer if it exists
        if hasattr(trainer, '_tokenizer'):
            trainer._tokenizer = tokenizer
            print("Applied prepared tokenizer to trainer's internal tokenizer")
            
        trainer.train()
        trainer.save_model("phi2-grpo-final")
        # Save tokenizer with final model
        tokenizer.save_pretrained("phi2-grpo-final")
        print("Training completed successfully!")
        print("Total time:", time.time() - start_time, "seconds")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        print("Time until error:", time.time() - start_time, "seconds")
        # Print error type
        print(f"Error type: {type(e)}")
        
        # If it's a padding token error, try to print more details
        if "padding token" in str(e):
            print("\nPadding token debug information:")
            print(f"Model pad_token_id: {model.config.pad_token_id}")
            print(f"Tokenizer pad_token: {tokenizer.pad_token}")
            print(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")
            
            # Try to access trainer's tokenizer
            if hasattr(trainer, 'tokenizer'):
                print("\nTrainer tokenizer information:")
                print(f"Trainer tokenizer pad_token: {trainer.tokenizer.pad_token}")
                print(f"Trainer tokenizer pad_token_id: {trainer.tokenizer.pad_token_id}")
        
        raise
    finally:
        clear_memory()
        print("\nFinal GPU state:")
        print_gpu_utilization()

if __name__ == "__main__":
    main()
