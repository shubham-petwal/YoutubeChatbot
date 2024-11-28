# YouTube Video Chatbot 🎥🤖

A Streamlit-based chatbot application that allows you to interact with the content of YouTube videos by processing transcripts, creating embeddings, and using them to generate intelligent responses to user queries. Powered by LangChain, Google Generative AI (Gemini), and Streamlit.

---

## Features 🚀

- Extract transcripts from YouTube videos.
- Automatically translate non-English transcripts to English (if required).
- Split transcripts into meaningful chunks and generate embeddings using Google Gemini.
- Retrieve relevant chunks and answer queries with AI-generated responses.
- User-friendly chat interface built with Streamlit.

---

## Getting Started 🛠️

### Prerequisites

- Python 3.8 or later
- [Streamlit](https://streamlit.io/)
- A valid Google API key (stored in `.env` file)

---

### Installation Steps

1. **Clone the Repository**  
   Clone the project from the GitHub repository using the following command:
   ```bash
   git clone https://github.com/shubham-petwal/YoutubeChatbot.git
   ```
   
2. **Navigate to the Project Directory**  
   Use the cd command to navigate to the project directory:
   ```bash
   cd YoutubeChatbot
   ```
3. **Create and Activate a Virtual Environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
4. **Install Required Dependencies**  
   Install all required Python dependencies listed in requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```
5. **Add Your Google API Key**  
   Create a .env file in the root project directory.
   Add your Google API key to the .env file:
   ```bash
   GOOGLE_API_KEY=your_google_api_key
   ```
6. **Run the Application**  
    Start the Streamlit app by running:
   ```bash
   streamlit run app.py
   ```
5. **Usage 📖**  
    Open the application in your default web browser (Streamlit will provide a link in the terminal).
    Enter the YouTube video URL into the input field and click Process Video.
    Once the video is processed, ask questions about the video content using the chat interface.
