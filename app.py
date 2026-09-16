import streamlit as st
import cassio
import tempfile
import os
import json
import random
import time
import re
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from PIL import Image,ImageEnhance
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
import textwrap
import hashlib

# Configuration Class
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
    
    @classmethod
    def validate(cls):
        if not cls.OPENAI_API_KEY:
            st.error("OpenAI API key not found in environment variables")
            st.stop()
        if not cls.ASTRA_DB_APPLICATION_TOKEN or not cls.ASTRA_DB_ID:
            st.error("Astra DB credentials not found in environment variables")
            st.stop()

Config.validate()

# Initialize Astra DB connection
@st.cache_resource
def init_astra():
    cassio.init(token=Config.ASTRA_DB_APPLICATION_TOKEN, database_id=Config.ASTRA_DB_ID)
    embedding = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)
    vector_store = Cassandra(
        embedding=embedding,
        table_name="qa_mini_demo",
        session=None,
        keyspace=None
    )
    return vector_store

astra_vector_store = init_astra()
astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

# Text Processing Functions
def extract_text_from_pdf(pdf_path: str) -> str:
    """Enhanced PDF text extraction with multiple fallback strategies"""
    try:
        # Try PyPDF2 first
        text = ""
        with open(pdf_path, 'rb') as f:
            pdf_reader = PdfReader(f)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        # If minimal text extracted, try OCR
        if len(text.strip().split()) < 100:  # Threshold for considering OCR
            try:
                images = convert_from_path(
                    pdf_path,
                    dpi=300,  # Higher DPI for better OCR
                    grayscale=True,  # Often improves OCR accuracy
                    thread_count=4  # Parallel processing
                )
                ocr_text = ""
                for img in images:
                    # Preprocess image for better OCR
                    img = img.convert('L')  # Grayscale
                    img = ImageEnhance.Contrast(img).enhance(2.0)  # Increase contrast
                    ocr_text += pytesseract.image_to_string(img) + "\n"
                
                if len(ocr_text.strip().split()) > len(text.strip().split()):
                    text = ocr_text
            except Exception as ocr_error:
                st.warning(f"OCR processing had issues: {str(ocr_error)}")
        
        return text if text.strip() else None
    except Exception as e:
        st.error(f"PDF processing failed: {str(e)}")
        return None

def format_answer(answer: str) -> str:
    """Format answer text for better readability"""
    return textwrap.fill(answer, width=80)

def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts using word overlap.
    Returns a score between 0 and 1.
    """
    def preprocess(text: str) -> set:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return set(text.split())
    
    words1 = preprocess(text1)
    words2 = preprocess(text2)
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    return intersection / union if union > 0 else 0.0

def questions_too_similar(question1: str, question2: str, threshold: float = 0.5) -> bool:
    """
    Check if two questions are too similar.
    Returns True if similarity is above threshold.
    """
    return calculate_similarity(question1, question2) > threshold

def query_pdf_component():
    st.subheader("💡 Ask Questions from the Document")
    query_text = st.text_input("🔍 Enter your question", key="query_input")

    if st.button("Get Answer") and query_text:
        with st.spinner("⏳ Searching for answer..."):
            try:
                # Get documents with scores
                retrieved_docs_with_scores = astra_vector_store.similarity_search_with_score(
                    query_text, 
                    k=6
                )
                
                if not retrieved_docs_with_scores:
                    # Fallback to keyword search if no results
                    keywords = re.findall(r'\b\w+\b', query_text.lower())
                    if keywords:
                        fallback_query = " ".join([w for w in keywords if len(w) > 3][:3])
                        retrieved_docs_with_scores = astra_vector_store.similarity_search_with_score(
                            fallback_query, 
                            k=6
                        )
                
                if not retrieved_docs_with_scores:
                    st.error("No relevant information found in document")
                    return
                
                # Sort by score (lower is better)
                retrieved_docs_with_scores.sort(key=lambda x: x[1])
                
                # Prepare context from top 4 documents
                top_docs_with_scores = retrieved_docs_with_scores[:4]
                context = "\n\n".join([doc.page_content for doc, score in top_docs_with_scores])
                
                prompt_template = PromptTemplate.from_template(
                    """You are a document question answering system. Answer the question using ONLY the following context. 
                    If the answer cannot be found in the context, respond with "The document doesn't contain this information."
                    
                    Rules:
                    1. Only use facts explicitly stated in the context
                    2. Never make up information or use prior knowledge
                    3. Be concise and specific
                    4. If the answer is partially available, provide the partial information
                    5. Do not say "based on the context" or "according to the document" - just answer directly
                    
                    Context:
                    {context}
                    
                    Question: {question}
                    Answer:"""
                )
                
                chat_model = ChatOpenAI(
                    model_name="gpt-3.5-turbo",
                    temperature=0.0,
                    max_tokens=500,
                    openai_api_key=Config.OPENAI_API_KEY
                )
                
                response = chat_model.invoke(prompt_template.format(
                    context=context,
                    question=query_text
                ))
                
                ai_answer = response.content
                if "document doesn't contain" not in ai_answer.lower():
                    answer_words = set(re.findall(r'\b\w{4,}\b', ai_answer.lower()))
                    context_words = set(re.findall(r'\b\w{4,}\b', context.lower()))
                    
                    if len(answer_words) > 0 and len(answer_words & context_words) / len(answer_words) < 0.4:
                        ai_answer = "The document doesn't contain this information."
                
                with st.expander("📖 Answer", expanded=True):
                    st.markdown(format_answer(ai_answer))
                
                with st.expander("🔍 Top Relevant Document Sections"):
                    for i, (doc, score) in enumerate(top_docs_with_scores):
                        st.markdown(f"**Section {i+1} [Score: {score:.4f}]**")
                        st.markdown(format_answer(doc.page_content))
                        st.divider()
                
            except Exception as e:
                st.error(f"Error finding answer: {str(e)}")

def generate_flashcards_component():
    st.subheader("🗂️ Generate Flashcards")
    
    # Initialize session state variables
    if "flashcards" not in st.session_state:
        st.session_state.flashcards = []
    
    if "used_questions" not in st.session_state:
        st.session_state.used_questions = set()
        
    if "document_topics" not in st.session_state:
        st.session_state.document_topics = []
    
    if "current_question" not in st.session_state:
        st.session_state.current_question = 0
    
    if "show_answer" not in st.session_state:
        st.session_state.show_answer = False
    
    if "quiz_score" not in st.session_state:
        st.session_state.quiz_score = 0
    
    if "quiz_completed" not in st.session_state:
        st.session_state.quiz_completed = False
    
    if "quiz_started" not in st.session_state:
        st.session_state.quiz_started = False
    
    if "flashcard_answer" not in st.session_state:
        st.session_state.flashcard_answer = ""
    
    if "answered_questions" not in st.session_state:
        st.session_state.answered_questions = set()
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        num_cards = st.slider("Number of flashcards to generate", 1, 20, 5)
    
    with col2:
        if st.button("Generate Cards"):
            with st.spinner("Creating flashcards..."):
                try:
                    if not st.session_state.document_topics:
                        try:
                            all_docs = astra_vector_store.similarity_search("important concepts", k=10)
                            if all_docs:
                                topic_context = "\n\n".join([doc.page_content for doc in all_docs])
                                
                                topic_prompt = PromptTemplate.from_template(
                                    """Extract 10-15 distinct major topics from the text below. 
                                    Each topic should be a different subject area or concept from the document.
                                    Do not repeat similar topics.
                                    
                                    Text content:
                                    {context}
                                    
                                    Format as a JSON array of topic strings.
                                    """
                                )
                                
                                chat_model = ChatOpenAI(
                                    model_name="gpt-3.5-turbo",
                                    temperature=0.2,
                                    max_tokens=1000,
                                    openai_api_key=Config.OPENAI_API_KEY
                                )
                                
                                topics_response = chat_model.invoke(topic_prompt.format(context=topic_context))
                                
                                json_match = re.search(r'\[.*\]', topics_response.content, re.DOTALL)
                                if json_match:
                                    st.session_state.document_topics = json.loads(json_match.group(0))
                        except Exception as e:
                            st.warning(f"Could not extract document topics: {str(e)}")
                            st.session_state.document_topics = ["General"]
                    
                    all_docs = astra_vector_store.similarity_search("important concepts", k=20)
                    
                    if not all_docs:
                        st.error("No document content available for flashcard generation")
                        return
                    
                    context = "\n\n".join([doc.page_content for doc in all_docs])
                    
                    topic_count = len(st.session_state.document_topics) if st.session_state.document_topics else 1
                    
                    prompt_template = PromptTemplate.from_template(
                        """Generate {num_cards} diverse educational flashcards about key concepts from the following text.
                        Each flashcard should have a question and answer about an important concept.
                        
                        CRITICAL REQUIREMENTS:
                        1. Make each question about a DIFFERENT topic or concept
                        2. Cover a wide range of topics from the document
                        3. AVOID creating questions that are variations of the same concept
                        4. Each question must be substantially different from others
                        5. Ensure questions test different knowledge domains
                        6. Make questions challenging but answerable using the text provided
                        
                        Text content:
                        {context}
                        
                        Available topics to cover (distribute questions across these):
                        {topics}
                        
                        Format as a JSON array with each flashcard containing "question", "answer", and "topic" fields.
                        """
                    )
                    
                    chat_model = ChatOpenAI(
                        model_name="gpt-3.5-turbo",
                        temperature=0.7,
                        max_tokens=2000,
                        openai_api_key=Config.OPENAI_API_KEY
                    )
                    
                    response = chat_model.invoke(prompt_template.format(
                        context=context,
                        num_cards=num_cards*2,
                        topics=", ".join(st.session_state.document_topics) if st.session_state.document_topics else "General"
                    ))
                    
                    json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
                    if not json_match:
                        st.error("Failed to generate valid flashcards")
                        return
                    
                    try:
                        all_flashcards = json.loads(json_match.group(0))
                        
                        new_flashcards = []
                        
                        for card in all_flashcards:
                            question = card["question"].strip()
                            
                            skip = False
                            for used_q in st.session_state.used_questions:
                                if questions_too_similar(question, used_q):
                                    skip = True
                                    break
                            
                            if not skip:
                                new_flashcards.append(card)
                                if len(new_flashcards) >= num_cards:
                                    break
                        
                        if len(new_flashcards) < num_cards:
                            remaining = [c for c in all_flashcards if c not in new_flashcards]
                            new_flashcards.extend(remaining[:num_cards-len(new_flashcards)])
                        
                        st.session_state.flashcards = new_flashcards[:num_cards]
                        
                        for card in st.session_state.flashcards:
                            st.session_state.used_questions.add(card["question"].strip())
                        
                        st.session_state.current_question = 0
                        st.session_state.show_answer = False
                        st.session_state.quiz_score = 0
                        st.session_state.quiz_completed = False
                        st.session_state.quiz_started = True
                        st.session_state.answered_questions = set()
                        st.session_state.flashcard_answer = ""
                        
                        random.shuffle(st.session_state.flashcards)
                        
                        st.success(f"✅ Generated {len(st.session_state.flashcards)} unique flashcards!")
                        
                    except json.JSONDecodeError:
                        st.error("Failed to parse flashcards format")
                        return
                        
                except Exception as e:
                    st.error(f"Error generating flashcards: {str(e)}")
    
    if st.session_state.flashcards:
        if st.session_state.quiz_completed:
            st.subheader("📊 Final Quiz Results")
            total_flashcards = len(st.session_state.flashcards)
            score_percentage = (st.session_state.quiz_score / total_flashcards) * 100
            st.success(f"Final Score: {st.session_state.quiz_score}/{total_flashcards} ({score_percentage:.1f}%)")
            
            if st.button("Start New Quiz"):
                random.shuffle(st.session_state.flashcards)
                st.session_state.current_question = 0
                st.session_state.show_answer = False
                st.session_state.quiz_score = 0
                st.session_state.quiz_completed = False
                st.session_state.quiz_started = True
                st.session_state.answered_questions = set()
                st.session_state.flashcard_answer = ""
                st.rerun()
            
            if st.button("Clear Question History"):
                st.session_state.used_questions = set()
                st.success("Question history cleared! You can now generate previously seen questions.")
        
        elif st.session_state.quiz_started:
            st.subheader("📚 Flashcard Quiz")
            
            progress_text = f"Question {st.session_state.current_question + 1} of {len(st.session_state.flashcards)}"
            progress_bar = st.progress(st.session_state.current_question / len(st.session_state.flashcards))
            st.write(progress_text)
            
            total_flashcards = len(st.session_state.flashcards)
            score_percentage = (st.session_state.quiz_score / total_flashcards) * 100
            st.info(f"Current Score: {st.session_state.quiz_score}/{total_flashcards} ({score_percentage:.1f}%)")
            
            current_card = st.session_state.flashcards[st.session_state.current_question]
            
            if "topic" in current_card:
                st.markdown(f"**Topic: {current_card['topic']}**")
            
            st.markdown("### Question:")
            st.markdown(f"**{current_card['question']}**")
            
            user_answer = st.text_area("Your answer:", height=100, key="flashcard_answer", value=st.session_state.flashcard_answer)
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if st.button("Check Answer"):
                    st.session_state.show_answer = True
                    st.session_state.answered_questions.add(st.session_state.current_question)
                    
                    if user_answer.strip():
                        similarity_score = calculate_similarity(user_answer, current_card['answer'])
                        
                        if similarity_score >= 0.1:
                            st.session_state.quiz_score += 1
                            st.success(f"✅ Correct answer! (Similarity: {similarity_score:.1%})")
                        else:
                            st.error(f"❌ Not correct. (Similarity: {similarity_score:.1%})")
                    else:
                        st.error("❌ No answer provided - counted as wrong")
            
            with col2:
                def next_question():
                    st.session_state.current_question += 1
                    st.session_state.show_answer = False
                    st.session_state.flashcard_answer = ""
                    if st.session_state.current_question >= len(st.session_state.flashcards):
                        st.session_state.quiz_completed = True
                
                if st.button("Next Question", on_click=next_question):
                    st.rerun()
            
            with col3:
                def end_quiz():
                    st.session_state.quiz_completed = True
                    total_flashcards = len(st.session_state.flashcards)
                    unanswered = total_flashcards - len(st.session_state.answered_questions)
                    st.session_state.quiz_score = max(0, st.session_state.quiz_score - unanswered)
                
                if st.button("End Quiz", on_click=end_quiz):
                    st.rerun()
            
            if st.session_state.show_answer:
                st.markdown("### Correct Answer:")
                st.markdown(f"{current_card['answer']}")

def main():
    st.title("📄 Document Intelligence System")
    
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
            
            if "flashcards" in st.session_state:
                st.session_state.flashcards = []
            if "used_questions" in st.session_state:
                st.session_state.used_questions = set()
            if "document_topics" in st.session_state:
                st.session_state.document_topics = []

        if not st.session_state.doc_processed:
            with st.spinner("⏳ Processing document..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                        temp_pdf.write(uploaded_file.getbuffer())
                        st.session_state.temp_pdf_path = temp_pdf.name
                    
                    raw_text = extract_text_from_pdf(st.session_state.temp_pdf_path)
                    if not raw_text:
                        st.error("Failed to extract text from PDF")
                        return
                    
                    st.session_state.pdf_text = raw_text
                    text_splitter = CharacterTextSplitter(
                        separator="\n",
                        chunk_size=800,
                        chunk_overlap=200,
                        length_function=len
                    )
                    texts = text_splitter.split_text(raw_text)
                    
                    chunks_to_add = min(50, len(texts))
                    
                    progress_bar = st.progress(0)
                    chunk_step = 1.0 / chunks_to_add if chunks_to_add > 0 else 1.0
                    
                    for i in range(0, chunks_to_add, 5):
                        batch = texts[i:min(i+5, chunks_to_add)]
                        astra_vector_store.add_texts(batch)
                        progress_bar.progress(min(1.0, (i + len(batch)) * chunk_step))
                        time.sleep(0.1)
                    
                    st.session_state.chunk_count = chunks_to_add
                    st.session_state.doc_processed = True
                    st.success(f"✅ Document processed successfully! Added {chunks_to_add} chunks to database.")
                    
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
            query_pdf_component()
        elif option == "Summarize Document":
            st.subheader("📝 Document Summary")
            
            # Summary type selection
            summary_type = st.radio(
                "Select summary length:",
                ["Short", "Detailed"],
                horizontal=True,
                key="summary_type"
            )
            
            if st.button("Generate Summary"):
                with st.spinner("🔍 Analyzing document..."):
                    try:
                        # Adjust the number of chunks based on summary type
                        k_value = 5 if summary_type.startswith("Short") else 10
                        
                        key_sections = astra_vector_store.similarity_search(
                            "important information main points summary", 
                            k=k_value
                        )
                        
                        if key_sections:
                            summary_context = "\n\n".join([doc.page_content for doc in key_sections])
                            
                            # Different prompts for short vs long summaries
                            if summary_type.startswith("Short"):
                                summary_prompt = PromptTemplate.from_template(
                                    """Generate a concise bullet-point summary of the key points from this document.
                                    - Focus only on the most important information
                                    - Use brief phrases (no full sentences)
                                    - Maximum 5-7 bullet points
                                    - Skip examples and minor details
                                    
                                    Document content:
                                    {context}
                                    
                                    Format as markdown bullet points:"""
                                )
                            else:
                                summary_prompt = PromptTemplate.from_template(
                                    """Generate a comprehensive summary of this document including:
                                    1. Main topics and themes (2-3 sentences)
                                    2. Key findings/conclusions (bulleted list)
                                    3. Important supporting details (short paragraphs)
                                    4. Any notable examples or evidence
                                    
                                    Maintain a professional tone and organize information logically.
                                    Length: approximately 1-2 paragraphs plus bullet points.
                                    
                                    Document content:
                                    {context}"""
                                )
                            
                            chat_model = ChatOpenAI(
                                model_name="gpt-3.5-turbo",
                                temperature=0.3,
                                max_tokens=1000 if summary_type.startswith("Short") else 1500,
                                openai_api_key=Config.OPENAI_API_KEY
                            )
                            
                            summary_response = chat_model.invoke(summary_prompt.format(context=summary_context))
                            
                            with st.expander(f"📋 {summary_type} Summary", expanded=True):
                                st.markdown(summary_response.content)
                                
                            # Add download option
                            summary_text = summary_response.content
                            st.download_button(
                                label="Download Summary",
                                data=summary_text,
                                file_name=f"{'short' if summary_type.startswith('Short') else 'detailed'}_summary.txt",
                                mime="text/plain"
                            )
                        else:
                            st.warning("Could not extract enough content for a good summary")
                            
                    except Exception as e:
                        st.error(f"Summary generation failed: {str(e)}")
        elif option == "Generate Flashcards":
            generate_flashcards_component()

    if st.session_state.temp_pdf_path and os.path.exists(st.session_state.temp_pdf_path):
        try:
            os.unlink(st.session_state.temp_pdf_path)
        except:
            pass

if __name__ == "__main__":
    main()