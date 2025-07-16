from langchain.evaluation.qa import QAEvalChain
from langchain.prompts import PromptTemplate

if st.button("Evaluate Response"):
    eval_prompt = PromptTemplate(
        input_variables=["question", "answer", "context"],
        template="""
        Given the question: {question}
        And the provided answer: {answer}
        Evaluate how well the answer matches the context:
        {context}
        Return a score 1-5 and a justification.
        """,
    )
    eval_chain = QAEvalChain.from_llm(OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY))
    result = eval_chain.evaluate(
        predictions=[{"question": query, "answer": result}],
        references=[{"context": "\n".join([doc.page_content for doc in docs])}]
    )
    st.info(result[0]["text"])
