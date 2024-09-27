import streamlit as st
import time
from components import load_and_create_embeddings, generate_response

if 'docs' not in st.session_state:
    st.session_state.docs = None
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'document_embeddings' not in st.session_state:
    st.session_state.document_embeddings = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

st.title("PDF Query System")


def parse_groq_stream(stream):
    word = ""
    for chunk in stream:
        for char in chunk:
            if char.isspace():  # Check if the character is a space
                if word:
                    yield word
                    word = ""
            else:
                word += char
    if word:  # Yield the last word if there is no space after it
        yield word


def display_messages():
    for sender, message in st.session_state.messages:
        with st.chat_message(sender):
            st.markdown(message)


uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None and st.session_state.docs is None:
    with st.spinner('Processing the PDF...'):
        docs, document_embeddings, faiss_index = load_and_create_embeddings(
            uploaded_file)
        st.session_state.docs = docs
        st.session_state.faiss_index = faiss_index
        st.session_state.document_embeddings = document_embeddings
    st.success('PDF processing complete!')

if st.session_state.docs is not None:
    query = st.chat_input("Enter your query here...")

    if query:
        st.session_state.messages.append(('user', query))

        display_messages()

        bot_response_container = st.empty()
        bot_response = ""

        temp_bot_response = []

        for word in parse_groq_stream(generate_response(query, st.session_state.docs, st.session_state.faiss_index)):
            bot_response += word + " "
            temp_bot_response.append(word)
            bot_response_container.markdown(
                ' '.join(temp_bot_response))
            time.sleep(0.2)

        # with st.spinner('Generating response...'):
        #     response = generate_response(
        #         query, st.session_state.docs, st.session_state.faiss_index)

        st.session_state.messages.append(('system', bot_response.strip()))

# display_messages()
