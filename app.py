import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load model and tokenizer
def load_model(model_id):
    # First load the base model
    base_model_id = "microsoft/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Ensure tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load and merge the LoRA adapter
    model = PeftModel.from_pretrained(base_model, model_id)
    return model, tokenizer

def generate_response(instruction, model, tokenizer, max_length=200, temperature=0.7, top_p=0.9):
    # Format the input text
    input_text = instruction.strip()
    
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True
        )
    
    # Decode and return the response
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the response part (what comes after the instruction)
    if len(input_text) < len(full_text):
        response = full_text[len(input_text):].strip()
        return response
    return full_text.strip()

def create_demo(model_id):
    # Load model and tokenizer
    model, tokenizer = load_model(model_id)
    
    # Define the interface
    def process_input(instruction, max_length, temperature, top_p):
        try:
            return generate_response(
                instruction,
                model,
                tokenizer,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p
            )
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    # Create the interface
    demo = gr.Interface(
        fn=process_input,
        inputs=[
            gr.Textbox(
                label="Input Text",
                placeholder="Enter your text here...",
                lines=4
            ),
            gr.Slider(
                minimum=50,
                maximum=500,
                value=150,
                step=10,
                label="Maximum Length"
            ),
            gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.7,
                step=0.1,
                label="Temperature"
            ),
            gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.9,
                step=0.1,
                label="Top P"
            )
        ],
        outputs=gr.Textbox(label="Completion", lines=8),
        title="Phi-2 GRPO Model Demo",
        description="""This is a generative model trained using GRPO (Generative Reinforcement from Preference Optimization) 
        on the TLDR dataset. The model was trained to generate completions of around 150 characters.
        
        You can adjust the generation parameters:
        - **Maximum Length**: Controls the maximum length of the generated response
        - **Temperature**: Higher values make the output more random, lower values make it more focused
        - **Top P**: Controls the cumulative probability threshold for token sampling
        """,
        examples=[
            ["The quick brown fox jumps over the lazy dog."],
            ["In this tutorial, we will explore how to build a neural network for image classification."],
            ["The best way to prepare for an interview is to"],
            ["Python is a popular programming language because"]
        ]
    )
    return demo

if __name__ == "__main__":
    # Use your model ID
    model_id = "jatingocodeo/phi2-grpo"
    demo = create_demo(model_id)
    demo.launch() 