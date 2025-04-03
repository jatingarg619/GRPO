import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def load_quantized_model():
    model_name = "microsoft/phi-2"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model in 4-bit precision...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True
    )
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=512):
    # Format the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
    
    end_time = time.time()
    
    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generation_time = end_time - start_time
    
    return response, generation_time

def main():
    # Load model and tokenizer
    model, tokenizer = load_quantized_model()
    print("Model loaded successfully!")
    
    # Example prompts to test
    test_prompts = [
        "Write a short poem about artificial intelligence:",
        "Explain quantum computing in simple terms:",
        "Write a function to calculate fibonacci numbers in Python:"
    ]
    
    print("\nStarting inference tests...")
    print("-" * 50)
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        response, gen_time = generate_response(model, tokenizer, prompt)
        print(f"\nResponse:\n{response}")
        print(f"\nGeneration time: {gen_time:.2f} seconds")
        print("-" * 50)
    
    # Interactive mode
    print("\nEntering interactive mode (type 'quit' to exit)")
    while True:
        user_input = input("\nEnter your prompt: ")
        if user_input.lower() == 'quit':
            break
            
        response, gen_time = generate_response(model, tokenizer, user_input)
        print(f"\nResponse:\n{response}")
        print(f"\nGeneration time: {gen_time:.2f} seconds")

if __name__ == "__main__":
    main() 