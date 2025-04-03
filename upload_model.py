from huggingface_hub import HfApi, login
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def upload_model_to_hub(
    model_path="phi2-grpo-final",
    repo_name="phi2-grpo",  # This will create username/phi2-grpo on HF
    token=None  # Your HF token
):
    if token is None:
        token = os.getenv("HF_TOKEN")
        if token is None:
            raise ValueError(
                "Please provide your Hugging Face token either as an argument or set it as HF_TOKEN environment variable"
            )
    
    # Login to Hugging Face
    print("Logging in to Hugging Face...")
    login(token=token)
    
    # Create the repository
    api = HfApi()
    
    try:
        print(f"Creating repository: {repo_name}")
        api.create_repo(repo_name, private=True)
    except Exception as e:
        print(f"Repository might already exist: {e}")
    
    # Upload the model and tokenizer files
    print("Uploading model and tokenizer...")
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_name,
        repo_type="model"
    )
    
    print(f"Model uploaded successfully to: https://huggingface.co/{repo_name}")
    print("Note: The repository is private by default. You can make it public from the Hugging Face website if desired.")

if __name__ == "__main__":
    # You can either set HF_TOKEN environment variable or pass it directly here
    upload_model_to_hub(
        model_path="phi2-grpo-final",  # Path to your saved model
        repo_name="your-username/phi2-grpo",  # Replace with your username
        token=None  # Add your token here or set HF_TOKEN environment variable
    ) 