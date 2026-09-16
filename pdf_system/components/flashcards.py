import json
import random
import re
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from utils.text_utils import questions_too_similar, calculate_similarity, format_answer

class FlashcardSession:
    def __init__(self):
        self.flashcards = []
        self.used_questions = set()
        self.document_topics = []
        self.current_question = 0
        self.show_answer = False
        self.quiz_score = 0
        self.quiz_completed = False
        self.quiz_started = False
        self.flashcard_answer = ""
        self.answered_questions = set()

def generate_flashcards_component(vector_store):
    st.subheader("🗂️ Generate Flashcards")
    
    if "flashcard_session" not in st.session_state:
        st.session_state.flashcard_session = FlashcardSession()
    
    session = st.session_state.flashcard_session
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        num_cards = st.slider("Number of flashcards to generate", 1, 20, 5)
    
    with col2:
        if st.button("Generate Cards"):
            _generate_flashcards(vector_store, session, num_cards)
    
    if session.flashcards:
        if session.quiz_completed:
            _show_quiz_results(session)
        elif session.quiz_started:
            _show_flashcard_quiz(session)

def _generate_flashcards(vector_store, session, num_cards):
    with st.spinner("Creating flashcards..."):
        try:
            if not session.document_topics:
                _extract_document_topics(vector_store, session)
            
            all_docs = vector_store.similarity_search("important concepts", k=20)
            
            if not all_docs:
                st.error("No document content available for flashcard generation")
                return
            
            context = "\n\n".join([doc.page_content for doc in all_docs])
            topic_count = len(session.document_topics) if session.document_topics else 1
            
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
                topics=", ".join(session.document_topics) if session.document_topics else "General"
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
                    for used_q in session.used_questions:
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
                
                session.flashcards = new_flashcards[:num_cards]
                
                for card in session.flashcards:
                    session.used_questions.add(card["question"].strip())
                
                session.current_question = 0
                session.show_answer = False
                session.quiz_score = 0
                session.quiz_completed = False
                session.quiz_started = True
                session.answered_questions = set()
                session.flashcard_answer = ""
                
                random.shuffle(session.flashcards)
                
                st.success(f"✅ Generated {len(session.flashcards)} unique flashcards!")
                
            except json.JSONDecodeError:
                st.error("Failed to parse flashcards format")
                return
                
        except Exception as e:
            st.error(f"Error generating flashcards: {str(e)}")

def _extract_document_topics(vector_store, session):
    try:
        all_docs = vector_store.similarity_search("important concepts", k=10)
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
                session.document_topics = json.loads(json_match.group(0))
    except Exception as e:
        st.warning(f"Could not extract document topics: {str(e)}")
        session.document_topics = ["General"]

def _show_quiz_results(session):
    st.subheader("📊 Final Quiz Results")
    total_flashcards = len(session.flashcards)
    score_percentage = (session.quiz_score / total_flashcards) * 100
    st.success(f"Final Score: {session.quiz_score}/{total_flashcards} ({score_percentage:.1f}%)")
    
    if st.button("Start New Quiz"):
        random.shuffle(session.flashcards)
        session.current_question = 0
        session.show_answer = False
        session.quiz_score = 0
        session.quiz_completed = False
        session.quiz_started = True
        session.answered_questions = set()
        session.flashcard_answer = ""
        st.rerun()
    
    if st.button("Clear Question History"):
        session.used_questions = set()
        st.success("Question history cleared! You can now generate previously seen questions.")

def _show_flashcard_quiz(session):
    st.subheader("📚 Flashcard Quiz")
    
    progress_text = f"Question {session.current_question + 1} of {len(session.flashcards)}"
    progress_bar = st.progress(session.current_question / len(session.flashcards))
    st.write(progress_text)
    
    total_flashcards = len(session.flashcards)
    score_percentage = (session.quiz_score / total_flashcards) * 100
    st.info(f"Current Score: {session.quiz_score}/{total_flashcards} ({score_percentage:.1f}%)")
    
    current_card = session.flashcards[session.current_question]
    
    if "topic" in current_card:
        st.markdown(f"**Topic: {current_card['topic']}**")
    
    st.markdown("### Question:")
    st.markdown(f"**{current_card['question']}**")
    
    user_answer = st.text_area("Your answer:", height=100, key="flashcard_answer", value=session.flashcard_answer)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("Check Answer"):
            session.show_answer = True
            session.answered_questions.add(session.current_question)
            
            if user_answer.strip():
                similarity_score = calculate_similarity(user_answer, current_card['answer'])
                
                if similarity_score >= 0.1:
                    session.quiz_score += 1
                    st.success(f"✅ Correct answer! (Similarity: {similarity_score:.1%})")
                else:
                    st.error(f"❌ Not correct. (Similarity: {similarity_score:.1%})")
            else:
                st.error("❌ No answer provided - counted as wrong")
    
    with col2:
        def next_question():
            session.current_question += 1
            session.show_answer = False
            session.flashcard_answer = ""
            if session.current_question >= len(session.flashcards):
                session.quiz_completed = True
        
        if st.button("Next Question", on_click=next_question):
            st.rerun()
    
    with col3:
        def end_quiz():
            session.quiz_completed = True
            total_flashcards = len(session.flashcards)
            unanswered = total_flashcards - len(session.answered_questions)
            session.quiz_score = max(0, session.quiz_score - unanswered)
        
        if st.button("End Quiz", on_click=end_quiz):
            st.rerun()
    
    if session.show_answer:
        st.markdown("### Correct Answer:")
        st.markdown(f"{current_card['answer']}")