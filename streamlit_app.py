import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import requests
import cohere
import pandas as pd

# from dotenv import load_dotenv
# Load environment variables from .env file
# load_dotenv()
# Access the API key from the environment
# API_KEY = os.getenv("API_KEY")

# Load Streamlit Secrets
API_KEY = st.secrets["API_KEY"]

url = "https://api.cohere.ai/v1/check-api-key"

headers = {
    "accept": "application/json",
    "authorization": f"Bearer {API_KEY}"
}

response = requests.post(url, headers=headers)


# Check API Status
if response.json()['valid'] != True:
    with st.sidebar:
        st.error("Status: Error connecting to Cohere!")
else:
    with st.sidebar:
        st.success("Status: Live!")

        selected = option_menu(
            menu_title="Implementing Cohere's API Endpoints", #required
            options=["Chatbot", "Rerank", "Summarization", "Text Generator"],
        )
    

    co = cohere.Client(API_KEY)

    # Initiating Chatbot's message history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if selected == "Chatbot":
        st.title("Coral ChatBot")

        # Generating the response from the bot
        def simulate_conversation(messages):
            client = cohere.Client(API_KEY)

            # Concatenating all previous messages into a single string
            chat_history = "\n".join(message["content"] for message in messages)

            # Sending the chat history to the Cohere API and getting the response
            response = client.chat(
                message=chat_history,
            )

            # Extracting the latest response from the chat history
            reply = response.chat_history[-1].message
            return reply
        
        # Displaying previous messages in the chat interface
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Allowing the user to input a message
        if prompt := st.chat_input("Message CoralAI"):

            # Appending the user's message to the message history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generating and displaying the bot's response
            with st.chat_message("assistant"):
                response = simulate_conversation(st.session_state.messages)
                st.markdown(response)

                # Appending the bot's response to the message history
                st.session_state.messages.append({"role": "assistant", "content": response})

    if selected == "Rerank":
        st.title("Animal Fact Finder!")

        # User Input
        user_query = st.text_area("Explore facts related to animals:")

        # Loading the Documents/Information
        df = pd.read_csv("./animal_facts.csv", header=None)
        docs = df[0].tolist()

        # Button
        if st.button("Find it!"):
            # Checking length of the User Query
            if len(user_query) > 1:
                # Generate text based on user prompt
                response = co.rerank(
                    model="rerank-english-v3.0",
                    query=user_query,
                    documents=docs,
                    top_n=1,
                )

                index_value = response.results[0].index
                score = round(response.results[0].relevance_score, 2)
                generated_text = docs[index_value]
                
                # Display generated text in a scrollable text box
                st.text_area("Fact:", value=generated_text, height=180)
                if score >= 0.5:
                    st.success(f"Confidence: {score}")
                else:
                    st.error(f"Confidence: {score}")
            else:
                st.error("Enter text before proceeding!")


    if selected =="Summarization":
        st.title("Summarization")

        # User Input
        user_text = st.text_area("Enter your text here:")

        # Button to generate text
        if st.button("Summarize"):
            if len(user_text) > 1:
                # Generate text based on user prompt
                response = co.summarize(text=user_text)

                # Retrieve the Text
                summarized_text = response.summary
                
                # Display in a text box
                st.text_area("Summarized Text:", value=summarized_text, height=200)
            else:
                st.error("Enter text before proceeding!")

    if selected =="Text Generator":
        st.title("Text Generator")

        # User Input
        user_prompt = st.text_area("Type your question here:")

        # Button to generate text
        if st.button("Generate"):
            # Checking length of the User Query
            if len(user_prompt) > 1:
                # Generate text based on user prompt
                response = co.generate(prompt=user_prompt)

                # Retrieve the Text
                generated_text = response.generations[0].text
                
                # Display in a text box
                st.text_area("Generated Text:", value=generated_text, height=400)
            else:
                st.error("Enter text before proceeding!")