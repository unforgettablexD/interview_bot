from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import SVMRetriever
import os
import streamlit as st

os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'

# Load blog post

loader = PyPDFLoader("document.pdf")
data = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(data)

# VectorDB
embedding = HuggingFaceEmbeddings()
retriever = SVMRetriever.from_documents(documents=splits, embeddings=embedding)
content_excerpt = "Ask me question from product design "
result=retriever.get_relevant_documents(content_excerpt)

question_template = """
Instruction: Generate a question based on the provided content and data{result}:
Content: "{content_excerpt}"
"""

prompt = PromptTemplate.from_template(question_template) 
repo_id='mistralai/Mixtral-8x7B-Instruct-v0.1'


llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 1}
)

llm_chain = LLMChain(prompt=prompt, llm=llm)
question_result=llm_chain.run(result=result,content_excerpt=content_excerpt)


generated_question = question_result.rfind("Question: ") +len("Question: ")
generated_question= question_result[generated_question:]

response=input("Enter your response:")

evaluation_template = """
Instruction: Evaluate the given answer as good, bad, or neutral based on the content and data{result}:
Content: "{content_excerpt}"
Question: "{generated_question}"
Answer: "{user_answer}"
Evaluation:
"""

prompt_e = PromptTemplate.from_template(evaluation_template) 

llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.7}
)
llm_chain_1 = LLMChain(prompt=prompt_e, llm=llm)

print(llm_chain_1.run(result=result,content_excerpt=content_excerpt,generated_question=generated_question,user_answer=response))
