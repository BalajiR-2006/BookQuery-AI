import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import time
from datetime import datetime

# Load API Keys
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("❌ Google API Key is missing! Please check your .env file.")
    st.stop()

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY
)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_pdfs" not in st.session_state:
    st.session_state.processed_pdfs = False

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files"""
    if not pdf_docs:
        raise ValueError("No PDF files provided")
    
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text
        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {str(e)}")
            continue
    
    if not text.strip():
        raise ValueError("No text could be extracted from the PDF files")
    
    return text

def get_text_chunks(text):
    """Split text into manageable chunks for processing"""
    if not text or not text.strip():
        raise ValueError("Empty text provided for chunking")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, 
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    
    if not chunks:
        raise ValueError("No text chunks generated from the provided text")
    
    return chunks

def get_vector_store(text_chunks):
    """Create and save FAISS vector store from text chunks"""
    try:
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        
        vector_store = FAISS.from_texts(
            texts=text_chunks,
            embedding=embeddings_model
        )
        
        vector_store.save_local("faiss_index")
        return True
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return False

def get_response(user_question):
    """Get response from the AI model"""
    if not os.path.exists("faiss_index"):
        return "❌ No processed PDFs found. Please upload and process PDF files first."
    
    try:
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        
        # Load FAISS Vector Store
        new_db = FAISS.load_local(
            "faiss_index", 
            embeddings_model, 
            allow_dangerous_deserialization=True
        )
        
        # Search for relevant documents
        docs = new_db.similarity_search(user_question, k=3)
        
        # Extract context from documents
        context = "\n\n".join([doc.page_content for doc in docs]) if docs else ""
        
        if not context:
            return "⚠️ No relevant context found in the uploaded documents for your question."
        
        # Construct prompt
        prompt = f"""
        You are an expert assistant providing detailed answers based on the provided context.
        Use only the provided context for answering.
        If the answer is not available in the context, respond with "The answer is not available in the provided context."

        Context:
        {context}

        Question:
        {user_question}

        Answer:
        """
        
        # Get response from Gemini
        response = llm.invoke(prompt)
        result = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        
        return result
        
    except Exception as e:
        return f"Error processing question: {str(e)}"

def display_chat_message(message, is_user=True):
    """Display a chat message with styling"""
    if is_user:
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
            <div style="background-color: #007bff; color: white; padding: 10px 15px; 
                        border-radius: 15px 15px 5px 15px; max-width: 70%; 
                        word-wrap: break-word;">
                <strong>You:</strong><br>{message}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">
            <div style="background-color: #f1f3f4; color: #333; padding: 10px 15px; 
                        border-radius: 15px 15px 15px 5px; max-width: 70%; 
                        word-wrap: break-word;">
                <strong>🤖 AI Assistant:</strong><br>{message}
            </div>
        </div>
        """, unsafe_allow_html=True)

def clear_chat_history():
    """Clear the chat history"""
    st.session_state.chat_history = []
    st.rerun()

def main():
    st.set_page_config(
        page_title="Chat with PDF 🤖", 
        page_icon="📄", 
        layout="wide", 
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better chat styling
    st.markdown("""
    <style>
    .chat-container {
                width:100%;
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        background-color: #fafafa;
        margin-bottom: 20px;
    }
    
    .stTextInput > div > div > input {
        border-radius: 20px;
        padding: 10px 15px;
        border: 2px solid #e0e0e0;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #007bff;
        box-shadow: 0 0 0 0.2rem rgba(0,123,255,.25);
    }
    
    .main-header {
        font-size:500px;
        color: #2c3e50;
        margin-bottom: 30px;
    }
    
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-ready {
        background-color: #28a745;
    }
    
    .status-not-ready {
        background-color: #dc3545;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">BookQuery AI</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="main-header">chat with your PDFs</h2>', unsafe_allow_html=True)
    
    # Status indicator
    status_color = "status-ready" if st.session_state.processed_pdfs else "status-not-ready"
    status_text = "Ready to chat!" if st.session_state.processed_pdfs else "Upload and process PDFs to start"
    
    st.markdown(f"""
    <div style=" margin-bottom: 20px;">
        <span class="status-indicator {status_color}"></span>
        <span style="color: #666;">{status_text}</span>
    </div>
    """, unsafe_allow_html=True)

    # Main chat area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Chat history container
        chat_container = st.container()
        
        with chat_container:
            if st.session_state.chat_history:
                st.markdown('<div class="chat-container">', unsafe_allow_html=True)
                for message in st.session_state.chat_history:
                    display_chat_message(message["content"], message["is_user"])
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("👋 Welcome! Upload your PDFs and start asking questions about them.")

        # Clear chat button (outside form)
        col_clear_btn, col_spacer = st.columns([1, 4])
        with col_clear_btn:
            if st.button("🗑️ Clear Chat", help="Clear chat history"):
                clear_chat_history()

        # Chat input form - THIS IS THE KEY FIX
        with st.form(key='chat_form', clear_on_submit=True):
            col_input, col_send = st.columns([4, 1])
            
            with col_input:
                user_message = st.text_input(
                    "Type your message here...",
                    placeholder="Ask me anything about your PDFs...",
                    label_visibility="collapsed"
                )
            
            with col_send:
                send_button = st.form_submit_button("📤 Send", help="Send message")

        # Handle user input - SIMPLIFIED LOGIC
        if send_button and user_message and user_message.strip():
            if not st.session_state.processed_pdfs:
                st.warning("⚠️ Please upload and process PDF files first before asking questions.")
            else:
                # Add user message to history
                st.session_state.chat_history.append({
                    "content": user_message,
                    "is_user": True,
                    "timestamp": datetime.now()
                })
                
                # Show typing indicator
                with st.spinner("🤖 AI is thinking..."):
                    # Get AI response
                    ai_response = get_response(user_message)
                
                # Add AI response to history
                st.session_state.chat_history.append({
                    "content": ai_response,
                    "is_user": False,
                    "timestamp": datetime.now()
                })
                
                # Rerun to update the chat display
                st.rerun()

    # Sidebar for PDF management
    with st.sidebar:
        st.title("📚 Document Management")
        
        # PDF upload section
        st.markdown("### Upload Documents")
        pdf_docs = st.file_uploader(
            "Choose PDF Files", 
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF files to analyze"
        )
        
        # Processing section
        if st.button("🚀 Process Documents", type="primary", use_container_width=True):
            if not pdf_docs:
                st.error("❌ Please upload at least one PDF file")
            else:
                try:
                    with st.status("Processing documents...", expanded=True) as status:
                        # Step 1: Extract text
                        st.write("📥 Extracting text from PDFs...")
                        raw_text = get_pdf_text(pdf_docs)
                        st.write("✅ Text extraction completed!")

                        # Step 2: Create chunks
                        st.write("🔍 Splitting text into chunks...")
                        text_chunks = get_text_chunks(raw_text)
                        st.write(f"✅ Created {len(text_chunks)} text chunks!")

                        # Step 3: Create embeddings
                        st.write("🧠 Creating vector embeddings...")
                        success = get_vector_store(text_chunks)
                        
                        if success:
                            st.write("✅ Vector embeddings created!")
                            st.session_state.processed_pdfs = True
                            status.update(label="🎉 Processing Complete!", state="complete")
                            st.success("✅ Documents processed successfully! Start chatting now.")
                            
                            # Add welcome message to chat
                            welcome_msg = f"📚 Successfully processed {len(pdf_docs)} PDF file(s). You can now ask me questions about the content!"
                            st.session_state.chat_history.append({
                                "content": welcome_msg,
                                "is_user": False,
                                "timestamp": datetime.now()
                            })
                        else:
                            status.update(label="❌ Processing Failed", state="error")
                            
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Chat statistics
        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown("### 📊 Chat Statistics")
            user_messages = len([msg for msg in st.session_state.chat_history if msg["is_user"]])
            ai_messages = len([msg for msg in st.session_state.chat_history if not msg["is_user"]])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Your messages", user_messages)
            with col2:
                st.metric("AI responses", ai_messages)
        
        # Help section
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.info("""
        **How to get better answers:**
        - Be specific in your questions
        - Ask about topics covered in your PDFs
        - Try rephrasing if you don't get the expected answer
        - Use follow-up questions for clarification
        """)
        
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #666; font-size: 12px;'>
            Made with ❤️ by Balaji | 2025
            </div>
            """,
            unsafe_allow_html=True,
        )

if __name__ == "__main__":
    main()