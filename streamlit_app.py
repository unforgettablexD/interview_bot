import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import SVMRetriever

# Assuming Hugging Face API token is set in Streamlit secrets or environment variable
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'

# Initialize a set to store generated questions to ensure uniqueness
generated_questions = set()


# Load and split the document
loader = PyPDFLoader("document.pdf")  # Ensure 'document.pdf' is in your project folder or provide the correct path
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(data)

# Prepare for retrieval
embedding = HuggingFaceEmbeddings()
retriever = SVMRetriever.from_documents(documents=splits, embeddings=embedding)

# Set the repository ID for the language model
repo_id = 'mistralai/Mixtral-8x7B-Instruct-v0.1'

st.title("Product Design Interview Question Generator")

# User inputs for the content excerpt
content_excerpt = st.text_input("Enter a topic related to ask:")

if st.button("Generate Question") and content_excerpt:
    try:
        result = retriever.get_relevant_documents(content_excerpt)
        formatted_generated_questions = '\n'.join([f"- {question}" for question in generated_questions])

        question_template = f"""
        Instruction: Generate only one question based on the provided context and data "{result}" and avoid repeating the question present in "{formatted_generated_questions}":
        Context: "{content_excerpt}"
        """

        prompt = PromptTemplate.from_template(question_template)

        llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 1})
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        question_result = llm_chain.run(result=result, content_excerpt=content_excerpt, formatted_generated_questions=formatted_generated_questions)

        # Extract and display the generated question
        generated_question = question_result.split("Question: ")[-1].strip()

        generated_questions.add(generated_question)
        st.write(f"Generated Question: {generated_question}")
    
        # Allow users to enter a response
        user_response = st.text_area("Enter your response to the question:")

        if st.button("Evaluate Response", key="evaluate"):
            evaluation_template = f"""
            Instruction: Evaluate the given answer as good, bad, or neutral based on the question "{generated_question}", content, and data "{result}":
            Question: "{generated_question}"
            Answer: "{user_response}"
            Evaluation:
            """

            prompt_e = PromptTemplate.from_template(evaluation_template)

            llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.5})
            llm_chain_1 = LLMChain(prompt=prompt_e, llm=llm)

            evaluation_result = llm_chain_1.run(result=result, generated_question=generated_question, user_answer=user_response)
            st.write(evaluation_result)
        else:
          st.error("Please enter a topic to generate a question.")
    except Exception as e:
        st.error(f"An error occurred:{str(e)}")
