import re
import textwrap

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