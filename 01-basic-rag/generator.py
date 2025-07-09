from transformers import pipeline

class Generator:
    def __init__(self, model_name="gpt2"):
        self.pipeline = pipeline("text-generation", model=model_name, max_new_tokens=100)

    def generate(self, context, question):
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        return self.pipeline(prompt)[0]['generated_text']
