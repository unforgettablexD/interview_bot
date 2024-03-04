import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import SVMRetriever
import streamlit as st

# Replace this with your actual Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'


# Check for uniqueness of the generated question
def is_unique_question(question, questions_set):
    return question not in questions_set

# Load and split the document
loader = PyPDFLoader("document.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(data)

# Prepare for retrieval
embedding = HuggingFaceEmbeddings()
retriever = SVMRetriever.from_documents(documents=splits, embeddings=embedding)

# This should be replaced with the ID of a model that you intend to use from Hugging Face
repo_id = 'mistralai/Mixtral-8x7B-Instruct-v0.1' # Example placeholder

# Store generated questions to ensure uniqueness
generated_questions = set()
content_excerpt = "Ask me question from product design"
# st.title("Interview Question Generator")
# content_excerpt = st.text_input("Tell me which field you want an interview for:", "")

try:
    while True:
        
        result = retriever.get_relevant_documents(content_excerpt)
        formatted_generated_questions = '\n'.join([f"- {question}" for question in generated_questions])

        question_template = """
        Instruction: Generate only one question based on the provided content,data "{result}" and avoid repeating the question present in "{formatted_generated_questions}":
        Content: "{content_excerpt}"
        
        """

        prompt = PromptTemplate.from_template(question_template)

        llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 1})
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        question_result = llm_chain.run(result=result, content_excerpt=content_excerpt,formatted_generated_questions=formatted_generated_questions)

        # Extract the generated question
        generated_question = question_result.split("Question: ")[-1].strip()

        # print("\nGenerated Question_new:",generated_question)
        # print("\nGenerated Questions_set:",generated_questions)
        # if is_unique_question(generated_question, generated_questions):

        generated_questions.add(generated_question)
        print(f"\nGenerated Question: {generated_question}")

        response = input("\nEnter your response: ")

        evaluation_template = """
        Instruction: Evaluate the given answer as good, bad, or neutral based on the question{generated_question},and data{result}:
        Question: "{generated_question}"
        Answer: "{user_answer}"
        Evaluation:
        """

        prompt_e = PromptTemplate.from_template(evaluation_template) 

        llm = HuggingFaceHub(
            repo_id=repo_id, model_kwargs={"temperature": 0.5}
        )
        llm_chain_1 = LLMChain(prompt=prompt_e, llm=llm)

        evaluation_result=llm_chain_1.run(result=result,generated_question=generated_question,user_answer=response)
        #print(evaluation_result)
        # generated_evaluation=evaluation_result.split("Evaluation: ")[-1].strip()
        generated_evaluation = evaluation_result.find("Evaluation:") + len("Evaluation:")
        generated_evaluation = evaluation_result[generated_evaluation:].strip()
        print(generated_evaluation )
        
except KeyboardInterrupt:
    print("\nProgram terminated by user.")
