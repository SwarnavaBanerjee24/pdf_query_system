import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

def summarize_document(vector_store):
    st.subheader("📝 Document Summary")
    
    summary_type = st.radio(
        "Select summary length:",
        ["Short", "Detailed"],
        horizontal=True,
        key="summary_type"
    )
    
    if st.button("Generate Summary"):
        with st.spinner("🔍 Analyzing document..."):
            try:
                k_value = 5 if summary_type.startswith("Short") else 10
                key_sections = vector_store.similarity_search(
                    "important information main points summary", 
                    k=k_value
                )
                
                if key_sections:
                    summary_context = "\n\n".join([doc.page_content for doc in key_sections])
                    
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
                        
                    st.download_button(
                        label="Download Summary",
                        data=summary_response.content,
                        file_name=f"{'short' if summary_type.startswith('Short') else 'detailed'}_summary.txt",
                        mime="text/plain"
                    )
                else:
                    st.warning("Could not extract enough content for a good summary")
                    
            except Exception as e:
                st.error(f"Summary generation failed: {str(e)}")