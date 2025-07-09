from ragas.metrics import (
    context_precision,
    answer_relevance,
    faithfulness,
)
from ragas import evaluate
from datasets import Dataset

# Example data
samples = [
    {
        "question": "Where is the Eiffel Tower?",
        "contexts": ["The Eiffel Tower is in Paris.", "Paris is the capital of France."],
        "answer": "The Eiffel Tower is in Paris.",
        "ground_truth": "The Eiffel Tower is located in Paris, France."
    },
    {
        "question": "What is France known for?",
        "contexts": ["France is known for cheese and wine.", "Paris is the capital."],
        "answer": "France is famous for cheese and wine.",
        "ground_truth": "France is known for its cheese and wine."
    },
]

# Convert to HuggingFace Dataset
dataset = Dataset.from_list(samples)

# Run evaluation
results = evaluate(dataset, metrics=[context_precision, faithfulness, answer_relevance])
print("Evaluation Metrics:\n", results)
