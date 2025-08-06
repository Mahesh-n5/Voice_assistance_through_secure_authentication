import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
import spacy
import pickle
import numpy as np
import sys

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

class NLPProcessor:
    def __init__(self):
        # Initialize NLTK tools
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Load SpaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model...")
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Load Sentence-BERT model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load embeddings and answers
        try:
            with open('question_embeddings.pkl', 'rb') as f:
                self.question_embeddings = pickle.load(f)
            
            with open('answers.pkl', 'rb') as f:
                self.answers = pickle.load(f)
        except FileNotFoundError as e:
            print(f"Error loading NLP data files: {e}")
            sys.exit(1)

    def preprocess(self, text):
        """Preprocess text for NLP analysis"""
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalpha() and word not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)

    def process_query(self, query):
        """Process a user query and return the best answer"""
        processed_query = self.preprocess(query)
        user_embedding = self.model.encode([processed_query])[0]
        
        similarities = np.dot(self.question_embeddings, user_embedding) / (
            np.linalg.norm(self.question_embeddings, axis=1) * np.linalg.norm(user_embedding))
        
        best_match_index = similarities.argmax()
        best_match_score = similarities[best_match_index]
        
        return self.answers[best_match_index] if best_match_score >= 0.7 else "Not found"

def main():
    processor = NLPProcessor()
    print("NLP Processor ready...", flush=True)
    
    # Process commands from stdin
    for line in sys.stdin:
        command = line.strip()
        if not command:
            continue
            
        response = processor.process_query(command)
        print(response, flush=True)

if __name__ == "__main__":
    main()