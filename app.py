import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# Load environment variables
load_dotenv()

# Function to initialize session state
def init_state():
    st.session_state.messages = []
    st.session_state.history = []
    st.session_state.session_id = "unique_session_id"

# Function to select and initialize the LLM model
def select_llm_model(model_name, temperature):
    model_mapping = {
        "Gemma-7b-IT": "gemma-7b-it",
        "Llama3–70b-8192": "llama3-70b-8192",
        "Llama3–8b-8192": "llama3-8b-8192",
        "Mixtral-8x7b-32768": "mixtral-8x7b-32768"
    }
    selected_model = model_mapping.get(model_name)
    groq_api = os.environ.get('GROQ_API_KEY')
    if not groq_api:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
    llm = ChatGroq(temperature=temperature, model=selected_model, api_key=groq_api)
    return llm

# Function to convert Excel file to DataFrame
def convert_excel_to_df(uploaded_file):
    df = pd.read_excel(uploaded_file)
    return df

# Function to create a pandas dataframe agent
def create_pandas_agent(llm, df):
    agent_executor = create_pandas_dataframe_agent(
        llm,
        df,
        agent_type="tool-calling",
        verbose=True,
        prefix='Answer detailedly and in markdown format. Name of the data file is Drug.csv',
        suffix='If there is a need or has been asked for a graph then only generate a python code with import streamlit and at the end streamlit.pyplot(matplotlib.pyplot).',
        allow_dangerous_code=True,
        include_df_in_prompt=True
    )
    return agent_executor

# Function to query the agent and extract output
def query_data(agent, query):
    response = agent.invoke(query)
    output_value = response.get('output', 'No output found')
    graph_code = response.get('graph_code', '').strip()
    return output_value, graph_code

# Set up the Streamlit page
st.set_page_config(page_title="Generative AI Based Data Analyzer", page_icon="🤖", layout="wide")
st.title("Data Analyser with LangChain")

# Initialize session state if not already present
if 'messages' not in st.session_state:
    init_state()

# Sidebar for model selection and file upload
with st.sidebar:
    st.subheader("Select LLM Model")
    selected_model = st.radio(
        "Choose a model:",
        ("Gemma-7b-IT", "Llama3–70b-8192", "Llama3–8b-8192", "Mixtral-8x7b-32768")
    )
    
    # Add temperature slider
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

    uploaded_file = st.file_uploader("Upload Excel or CSV", type=["csv", "xlsx"])

# Main content area
if uploaded_file is not None:
    # Convert uploaded file to DataFrame
    if uploaded_file.name.endswith('.xlsx'):
        df = convert_excel_to_df(uploaded_file)
    elif uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        st.error("Only Excel or CSV files are allowed!")

    if selected_model and uploaded_file:
        llm = select_llm_model(selected_model, temperature)
        agent = create_pandas_agent(llm, df)

        # Display existing messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=True)

        # Handle user input and agent response
        if prompt := st.chat_input("Type your message here..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                placeholder.markdown("...")

                try:
                    output_value, graph_code = query_data(agent, prompt)
                    placeholder.markdown(output_value, unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": output_value})

                    # Execute the graph code if it exists
                    if graph_code:
                        exec(graph_code)
                        st.pyplot(plt.gcf())
                    
                except KeyError as e:
                    st.error(f"KeyError: {str(e)}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Display chat history if any
if st.session_state.history:
    for chat in st.session_state.history:
        st.write(f"**You:** {chat['query']}")
        st.write(f"**Agent:** {chat['response']}")
