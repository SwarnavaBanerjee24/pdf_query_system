import os
import streamlit as st
import hashlib
from utils.file_processing import extract_text_from_pdf, save_uploaded_file
from utils.vector_store import init_astra, process_and_store_text
from components.query import query_pdf_component
from components.summary import summarize_document
from components.flashcards import generate_flashcards_component
from config import Config

def main():
    st.title("📄 Document Intelligence System")
    
    try:
        Config.validate()
    except ValueError as e:
        st.error(str(e))
        st.stop()
    
    astra_vector_store = init_astra()
    
    uploaded_file = st.file_uploader("📤 Upload a PDF", type="pdf")
    
    if "doc_processed" not in st.session_state:
        st.session_state.doc_processed = False
    if "pdf_text" not in st.session_state:
        st.session_state.pdf_text = None
    if "temp_pdf_path" not in st.session_state:
        st.session_state.temp_pdf_path = None
    if "chunk_count" not in st.session_state:
        st.session_state.chunk_count = 0
    if "current_file_hash" not in st.session_state:
        st.session_state.current_file_hash = None

    if uploaded_file:
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        
        if file_hash != st.session_state.current_file_hash:
            st.session_state.doc_processed = False
            st.session_state.pdf_text = None
            if st.session_state.temp_pdf_path and os.path.exists(st.session_state.temp_pdf_path):
                os.unlink(st.session_state.temp_pdf_path)
            st.session_state.temp_pdf_path = None
            st.session_state.chunk_count = 0
            st.session_state.current_file_hash = file_hash
            
            astra_vector_store.clear()
            
            if "flashcard_session" in st.session_state:
                st.session_state.flashcard_session = None

        if not st.session_state.doc_processed:
            with st.spinner("⏳ Processing document..."):
                try:
                    st.session_state.temp_pdf_path = save_uploaded_file(uploaded_file)
                    raw_text = extract_text_from_pdf(st.session_state.temp_pdf_path)
                    
                    if not raw_text:
                        st.error("Failed to extract text from PDF")
                        return
                    
                    st.session_state.pdf_text = raw_text
                    st.session_state.chunk_count = process_and_store_text(
                        raw_text, 
                        astra_vector_store
                    )
                    st.session_state.doc_processed = True
                    st.success(f"✅ Document processed successfully! Added {st.session_state.chunk_count} chunks to database.")
                    
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
                    if st.session_state.temp_pdf_path and os.path.exists(st.session_state.temp_pdf_path):
                        os.unlink(st.session_state.temp_pdf_path)

    if st.session_state.doc_processed:
        st.subheader("🔎 Document Analysis Options")
        option = st.radio("Select an action:", 
                         ["Query PDF", "Summarize Document", "Generate Flashcards"],
                         horizontal=True)
        
        if option == "Query PDF":
            query_pdf_component(astra_vector_store)
        elif option == "Summarize Document":
            summarize_document(astra_vector_store)
        elif option == "Generate Flashcards":
            generate_flashcards_component(astra_vector_store)

    if st.session_state.temp_pdf_path and os.path.exists(st.session_state.temp_pdf_path):
        try:
            os.unlink(st.session_state.temp_pdf_path)
        except:
            pass

if __name__ == "__main__":
    main()