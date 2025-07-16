from langchain.evaluation import QA
evaluator = QA()
score = evaluator.evaluate(reference="truth", prediction="model_answer")
