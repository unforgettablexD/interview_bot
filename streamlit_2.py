import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import SVMRetriever

# Set up your Hugging Face API token securely
# It's assumed that the HUGGINGFACEHUB_API_TOKEN is set in your environment for security
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = 'your_hf_api_token_here'
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'
repo_id = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
# Initialize or load necessary data and models once using Streamlit's session state
if 'data_loaded' not in st.session_state:
    # Load and split the document
    loader = PyPDFLoader("document.pdf")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    splits = text_splitter.split_documents(data)
    
    # Prepare for retrieval
    embedding = HuggingFaceEmbeddings()
    retriever = SVMRetriever.from_documents(documents=splits, embeddings=embedding)
    
    st.session_state['retriever'] = retriever
    st.session_state['data_loaded'] = True
    st.session_state['generated_questions'] = set()
    st.session_state.last_generated_question = ""

# Streamlit UI Elements
st.title("Product Design Question Generator")

# Field for user to input their topic of interest
content_excerpt = st.text_input("Enter a topic related to product design:", "Ask me a question from product design")
result = st.session_state.retriever.get_relevant_documents(content_excerpt)
# Generate Question Button
if st.button('Generate Question'):
    
    
    # Format previously generated questions for inclusion in the prompt
    formatted_generated_questions = '\\n'.join([f"- {question}" for question in st.session_state['generated_questions']])

    question_template = """
        Instruction: Generate only one question based on the provided content,data "{result}" and avoid repeating the question present in "{formatted_generated_questions}":
        Content: "{content_excerpt}"
        
        """

    # Set the repo_id for your Hugging Face model
    # repo_id = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
    prompt = PromptTemplate.from_template(question_template)
    llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 1})
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    question_result = llm_chain.run(result=result, content_excerpt=content_excerpt,formatted_generated_questions=formatted_generated_questions)

    # Extract the generated question
    generated_question = question_result.split("Question: ")[-1].strip()
    st.session_state['generated_questions'].add(generated_question)
    st.session_state['last_generated_question'] = generated_question  
    st.write("Generated Question:", st.session_state.last_generated_question)

# Placeholder for user response - For demonstration purposes
user_response = st.text_area("Your response to the question:", "")

if st.button('Evaluate Response') and user_response:
    with st.spinner('Evaluating response...'):

        evaluation_template = """
        Instruction: Evaluate the given answer as good, bad, or neutral based on the question,and data{result}:
        Question: "{generated_question}"
        Answer: "{user_answer}"
        Evaluation:
        """

        prompt_e = PromptTemplate.from_template(evaluation_template)
        llm_chain_1 = LLMChain(prompt=prompt_e, llm=HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.5}))
        evaluation_result = llm_chain_1.run(result=result,generated_question=st.session_state['last_generated_question'],user_answer=user_response)
        evaluation_text = evaluation_result.split("Evaluation: ")[-1].strip()
        st.write("Evaluation:", evaluation_text)