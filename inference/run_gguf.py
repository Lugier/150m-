'''
llama.cpp Python bindings wrapper.
Provides access to testing logic via extreme precision llama.cpp backend mechanisms.
'''
import argparse
import sys

def run_gguf_inference(model_path: str, prompt: str, max_tokens: int = 128):
    '''
    Initializes llama-cpp-python and runs the generation loop.
    '''
    try:
        from llama_cpp import Llama
    except ImportError:
        print("ERROR: llama-cpp-python is not installed. Please install it to use GGUF local inference.")
        print("Run: pip install llama-cpp-python")
        sys.exit(1)

    print(f"Loading GGUF Model from {model_path} via llama.cpp execution engine...")
    llm = Llama(model_path=model_path, n_ctx=8192, verbose=False)
    
    print(f"Evaluating Prompt: {prompt}")
    output = llm(prompt, max_tokens=max_tokens, echo=True)
    
    print("\\n--- GENERATION ---")
    generation = output["choices"][0]["text"]
    print(generation)
    print("------------------")
    return generation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to the GGUF model")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=128)
    args = parser.parse_args()
    
    run_gguf_inference(args.model, args.prompt, args.max_tokens)
