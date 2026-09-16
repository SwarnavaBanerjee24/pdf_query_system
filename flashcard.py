import os
import fitz  # PyMuPDF
import numpy as np
import json
import random
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy


class PDFFlashcardGenerator:
    def __init__(self):
        """Initialize NLP components and models lazily to optimize performance."""
        self.nlp = None  # Lazy loading of Spacy
        self.summarizer = None
        self.importance_tokenizer = None
        self.importance_model = None

        # Regex pattern for cleaning text
        self.clean_pattern = re.compile(r'[^a-zA-Z0-9\s]')

    def load_models(self):
        """Lazy-load NLP models when needed."""
        if self.nlp is None:
            self.nlp = spacy.load("en_core_web_sm")

        if self.summarizer is None:
            self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

        if self.importance_tokenizer is None:
            self.importance_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.importance_model = AutoModelForSequenceClassification.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2"
            )

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file."""
        text_content = []
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text = page.get_text() or ""
                    text_content.append(text)
            return "\n".join(text_content)
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")

    def preprocess_text(self, text):
        """Clean and preprocess extracted text."""
        text = self.clean_pattern.sub(' ', text)  # Remove special characters
        text = ' '.join(text.split())  # Normalize whitespace

        self.load_models()
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]

    def identify_key_concepts(self, sentences):
        """Identify important words using TF-IDF."""
        vectorizer = TfidfVectorizer(max_features=100)
        tfidf_matrix = vectorizer.fit_transform([' '.join(sentences)])

        feature_names = vectorizer.get_feature_names_out()
        importance_scores = np.asarray(tfidf_matrix.sum(axis=0)).ravel()
        return dict(zip(feature_names, importance_scores))

    def generate_flashcard_content(self, sentence, important_terms):
        """Generate a high-quality multiple-choice flashcard from a sentence."""
        self.load_models()
        doc = self.nlp(sentence)
        entities = [ent.text for ent in doc.ents]

        # If Named Entities exist, replace with blank
        if entities:
            entity = random.choice(entities)  # Choose a meaningful entity
            question = sentence.replace(entity, "___")
            correct_answer = entity
        else:
            # Pick an important word from TF-IDF scores
            words = sentence.split()
            important_words = [word for word in words if word.lower() in important_terms and len(word) > 3]

            if important_words:
                correct_answer = random.choice(important_words)
                question = sentence.replace(correct_answer, "___")
            else:
                # As a last resort, generate a conceptual question from summarization
                summary = self.summarizer(sentence, max_length=60, min_length=10, num_beams=4)[0]['summary_text']
                question = f"What does this mean? {summary}"
                correct_answer = summary

        # Generate multiple-choice options
        distractors = random.sample(list(important_terms.keys()), min(3, len(important_terms)))
        if correct_answer in distractors:
            distractors.remove(correct_answer)

        options = [correct_answer] + distractors[:3]
        random.shuffle(options)

        return {
            "question": question,
            "answer": correct_answer,
            "options": options  # New field for multiple-choice options
        }

    def generate_flashcards(self, pdf_path):
        """Generate flashcards from a PDF file."""
        text = self.extract_text_from_pdf(pdf_path)
        sentences = self.preprocess_text(text)
        important_terms = self.identify_key_concepts(sentences)

        flashcards = []
        for sentence in sentences:
            if len(sentence.split()) < 5:  # Ignore very short sentences
                continue
            flashcard = self.generate_flashcard_content(sentence, important_terms)
            flashcards.append(flashcard)

        return flashcards

    def save_flashcards(self, flashcards, output_path):
        """Save generated flashcards to a JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(flashcards, f, indent=4, ensure_ascii=False)

    def run(self, pdf_path):
        """Main execution function modified to return flashcards instead of saving to JSON."""
        try:
            flashcards = self.generate_flashcards(pdf_path)
            if flashcards:
                return flashcards  # Return the flashcards list instead of writing to a file
            else:
                return []  # Return an empty list if no flashcards are generated
        except FileNotFoundError:
            return [{"error": f"Error: PDF file not found at {pdf_path}. Ensure the file exists."}]
        except Exception as e:
            return [{"error": f"Error: {str(e)}"}]
