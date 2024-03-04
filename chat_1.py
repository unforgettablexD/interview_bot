import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import SVMRetriever
import streamlit as st
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
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

        template = """You are a interviewer chatbot taking interview of a human.

        Context={context}
        Question:{question}
        Human: {human_input}
        Chatbot:"""

        prompt_c = PromptTemplate.from_template(template)

#         conversation_buffer_window = ConversationChain(
#                 llm=llm,
#                 memory=ConversationBufferMemory(ai_prefix="Chatbot"),
#                 prompt=prompt_c,
#                 verbose=True
# )
        llm_chain_c = LLMChain(prompt=prompt_c, llm=llm)

        result=llm_chain_c.run(human_input=response,context=content_excerpt,question=generated_question)
        print(result)
        
except KeyboardInterrupt:
    print("\nProgram terminated by user.")
