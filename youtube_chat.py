import streamlit as st
import os

from dotenv import find_dotenv, load_dotenv
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from youtube_transcript_api.formatters import TextFormatter
from googletrans import Translator  # Translation library

# Load environment variables
load_dotenv(find_dotenv())
my_api_key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=my_api_key)

# Function to get embeddings using Google Gemini
def get_embeddings(text):
    model = 'models/embedding-001'
    embedding = genai.embed_content(
        model=model,
        content=text,
        task_type="retrieval_document"
    )
    return embedding['embedding']

# Create a database from a YouTube video transcript
# def create_db_from_youtube_video_url(video_url):
#     loader = YoutubeLoader.from_youtube_url(video_url)
#     transcript = loader.load()

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
#     docs = text_splitter.split_documents(transcript)

#     # Create embeddings for each chunk
#     content_list = [doc.page_content for doc in docs]
#     embeddings = [get_embeddings(content) for content in content_list]

#     # Store embeddings and documents in a DataFrame
#     dataframe = pd.DataFrame({
#         'page_content': content_list,
#         'embeddings': embeddings
#     })

#     return dataframe


# Function to create a database from YouTube video transcript
def create_db_from_youtube_video_url(video_url):
    try:
        # Extract video ID from the URL
        video_id = video_url.split("v=")[1]

        # Attempt to fetch transcripts in English
        transcript = None
        available_transcripts = YouTubeTranscriptApi.list_transcripts(video_id)

        # Look for an English transcript
        try:
            transcript = available_transcripts.find_transcript(['en'])
        except:
            st.warning("English transcript not found. Trying auto-generated transcripts...")

        # Fallback to auto-generated transcript
        if transcript is None:
            transcript = available_transcripts.find_generated_transcript(['hi'])  # Example: Hindi

        # Convert transcript to plain text
        formatter = TextFormatter()
        transcript_text = formatter.format_transcript(transcript.fetch())

        # Optionally, translate to English if it's not in English
        if transcript.language_code != 'en':
            st.info("Translating transcript to English...")
            translator = Translator()
            transcript_text = translator.translate(transcript_text, src='hi', dest='en').text

        # Split transcript into chunks and generate embeddings
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        docs = text_splitter.split_text(transcript_text)

        # Create embeddings for each chunk
        embeddings = [get_embeddings(doc) for doc in docs]

        # Store embeddings and documents in a DataFrame
        dataframe = pd.DataFrame({
            'page_content': docs,
            'embeddings': embeddings
        })

        return dataframe

    except TranscriptsDisabled:
        raise Exception("Transcripts are disabled for this video.")
    except Exception as e:
        raise Exception(f"Error processing video: {e}")





# Retrieve relevant documents using cosine similarity
def get_relevant_docs(dataframe, query, top_k=4):
    query_embedding = get_embeddings(query)

    # Calculate cosine similarity
    dataframe['similarity'] = dataframe['embeddings'].apply(
        lambda x: sum(a * b for a, b in zip(x, query_embedding))
    )
    top_docs = dataframe.nlargest(top_k, 'similarity')
    return top_docs['page_content'].tolist()

# Create the RAG prompt
def make_rag_prompt(query, relevant_passages):
    relevant_text = " ".join(relevant_passages)
    prompt = (
        f"You are a helpful and informative chatbot that answers questions using text from the reference passage below in list format and with headings and explainable format.\n"
        f"QUESTION: {query}\n"
        f"PASSAGE: {relevant_text}\n\n"
        f"ANSWER:"
    )
    return prompt

# Generate a response using Google Gemini
def generate_response(prompt):
    model = genai.GenerativeModel('gemini-pro')
    answer = model.generate_content(prompt)
    return answer.text

# Generate an answer based on the query
def generate_answer(dataframe, query):
    relevant_passages = get_relevant_docs(dataframe, query)
    prompt = make_rag_prompt(query, relevant_passages)
    answer = generate_response(prompt)
    return answer

# Streamlit App
import streamlit as st
import textwrap  # For formatting text


# Page configuration
st.set_page_config(page_title="YouTube Video Chatbot", layout="wide")

st.title("📺 YouTube Video Chatbot")
st.markdown("### Upload a YouTube URL and chat with the video's content!")

st.markdown("""
<style>
    .user-message {
        background-color: #f6f6f6;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .bot-message {
        background-color: #F2F4F6;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)
# App state management
if "db" not in st.session_state:
    st.session_state.db = None

if "chat_active" not in st.session_state:
    st.session_state.chat_active = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Initialize empty chat history

# Upload button at the top-right corner when the chatbot is active
if st.session_state.chat_active:
    st.sidebar.button("📤 Upload New URL", key="upload_button", on_click=lambda: st.session_state.update({
        "db": None,
        "chat_active": False,
        "chat_history": []
    }))

# Initial UI for Upload Section
if st.session_state.db is None or st.session_state.db.empty:
    video_url = st.text_input("Enter YouTube Video URL:")
    if st.button("Process Video"):
        if video_url.strip() == "":
            st.warning("Please enter a valid YouTube URL.")
        else:
            st.write("Processing video... This might take a while.")
            try:
                with st.spinner("Analyzing video content..."):
                    db = create_db_from_youtube_video_url(video_url)  # Your function here
                st.session_state.db = db
                st.session_state.chat_active = True
                st.session_state.chat_history.clear()  # Reset chat history
                st.success("Video processed successfully! Start chatting below.")
                st.rerun()
            except Exception as e:
                st.error(f"Error processing video: {e}")

# Chat Section
if st.session_state.db is not None and not st.session_state.db.empty:
    st.subheader("💬 Chat with the content")
    
    # Display chat history in reverse order
    for message in reversed(st.session_state.chat_history):
        if message["role"] == "User":
            st.markdown(f'<div class="user-message"><strong>🧑 User:</strong> {message["text"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message"><strong>🤖 Bot:</strong> {message["text"]}</div>', unsafe_allow_html=True)

    # Chat input field
    query = st.text_input("Ask something about the video:")
    if st.button("Submit Query"):
        if query.strip() == "":
            st.warning("Please enter a valid query.")
        else:
            # Save user message to history
            st.session_state.chat_history.insert(0, {"role": "User", "text": query})
            
            # Generate answer with a spinner
            try:
                with st.spinner("Generating response..."):
                    answer = generate_answer(st.session_state.db, query)  # Your function here
                st.session_state.chat_history.insert(0, {"role": "Bot", "text": answer})
                st.rerun()
            except Exception as e:
                st.error(f"Error generating response: {e}")
