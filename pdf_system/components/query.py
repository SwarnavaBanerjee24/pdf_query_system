from config import Config
import re
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from utils.text_analysis import format_answer

def query_pdf_component(vector_store):
    st.subheader("💡 Ask Questions from the Document")
    query_text = st.text_input("🔍 Enter your question", key="query_input")

    if st.button("Get Answer") and query_text:
        with st.spinner("⏳ Searching for answer..."):
            try:
                retrieved_docs_with_scores = vector_store.similarity_search_with_score(
                    query_text, 
                    k=6
                )
                
                if not retrieved_docs_with_scores:
                    keywords = re.findall(r'\b\w+\b', query_text.lower())
                    if keywords:
                        fallback_query = " ".join([w for w in keywords if len(w) > 3][:3])
                        retrieved_docs_with_scores = vector_store.similarity_search_with_score(
                            fallback_query, 
                            k=6
                        )
                
                if not retrieved_docs_with_scores:
                    st.error("No relevant information found in document")
                    return
                
                retrieved_docs_with_scores.sort(key=lambda x: x[1])
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