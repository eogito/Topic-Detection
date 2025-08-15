"""
Final Banking Topic Predictor - Production Ready
==============================================
Production-ready banking topic classifier with high confidence and correct mappings.
"""

import os
import pickle
import json
import numpy as np
from bertopic import BERTopic

class FinalBankingPredictor:
    """Production-ready banking topic predictor with high confidence"""
    
    def __init__(self):
        self.topic_model = None
        self.label_encoder = None
        self.topic_mapping = None
        self.model_loaded = False
        
        # Banking categories mapping
        self.banking_categories = {
            'Bank Accounts': ['account', 'accounts', 'checking', 'savings', 'deposit'],
            'Cards': ['card', 'credit', 'debit', 'fraud', 'fees'],
            'Customer Rewards': ['rewards', 'points', 'loyalty', 'programs', 'benefits'],
            'Digital Banking': ['online', 'mobile', 'app', 'digital', 'transfer'],
            'Insurance': ['insurance', 'life', 'coverage', 'health', 'protection'],
            'Investment Funds': ['investment', 'funds', 'mutual', 'etfs', 'portfolio'],
            'Mortgages': ['mortgage', 'home', 'refinance', 'property', 'loan'],
            'Pension Plans': ['pension', 'retirement', 'plan', 'planning', 'goals'],
            'Personal Loans': ['personal', 'loans', 'finance', 'borrowing', 'credit'],
            'Savings & Deposits': ['savings', 'deposit', 'term', 'rates', 'interest']
        }
    
    def load_model(self):
        """Load the advanced model"""
        model_dir = "advanced_banking_model"
        
        print(f"Loading production model from {model_dir}/")
        
        try:
            # Load BERTopic model
            self.topic_model = BERTopic.load(os.path.join(model_dir, "bertopic_model"))
            print("BERTopic model loaded")
            
            # Load label encoder
            with open(os.path.join(model_dir, "label_encoder.pkl"), 'rb') as f:
                self.label_encoder = pickle.load(f)
            print("Label encoder loaded")
            
            # Load topic mapping
            with open(os.path.join(model_dir, "topic_mapping.pkl"), 'rb') as f:
                self.topic_mapping = pickle.load(f)
            print("Topic mapping loaded")
            
            self.model_loaded = True
            print("Production model ready!")
            return True
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def predict_banking_category(self, text):
        """Predict banking category with intelligent mapping"""
        
        if not self.model_loaded:
            return {"error": "Model not loaded"}
        
        # Get BERTopic prediction
        topic_ids, probabilities = self.topic_model.transform([text])
        topic_id = topic_ids[0]
        
        result = {
            'text': text,
            'predicted_category': None,
            'confidence': 0.0,
            'confidence_level': 'LOW',
            'keywords': [],
            'is_outlier': topic_id == -1,
            'fallback_used': False
        }
        
        if topic_id != -1:
            # Get confidence and keywords
            confidence = float(probabilities[0][topic_id]) if probabilities is not None else 0.0
            result['confidence'] = confidence
            
            # Get topic keywords
            topic_words = self.topic_model.get_topic(topic_id)
            if topic_words:
                result['keywords'] = [word for word, _ in topic_words[:5]]
            
            # Set confidence level
            if confidence > 0.7:
                result['confidence_level'] = 'VERY HIGH'
            elif confidence > 0.5:
                result['confidence_level'] = 'HIGH'
            elif confidence > 0.3:
                result['confidence_level'] = 'MEDIUM'
            else:
                result['confidence_level'] = 'LOW'
            
            # Smart category mapping using keywords
            predicted_category = self._map_keywords_to_category(result['keywords'], text)
            
            if predicted_category:
                result['predicted_category'] = predicted_category
            else:
                # Fallback: use direct text analysis
                result['predicted_category'] = self._fallback_categorization(text)
                result['fallback_used'] = True
        else:
            # Outlier - use fallback categorization
            result['predicted_category'] = self._fallback_categorization(text)
            result['fallback_used'] = True
        
        return result
    
    def _map_keywords_to_category(self, keywords, text):
        """Map topic keywords to banking categories"""
        
        if not keywords:
            return None
        
        # Score each category based on keyword matches
        category_scores = {}
        
        for category, category_keywords in self.banking_categories.items():
            score = 0
            
            # Check keyword matches
            for keyword in keywords:
                for cat_keyword in category_keywords:
                    if cat_keyword.lower() in keyword.lower():
                        score += 2
            
            # Check text matches
            text_lower = text.lower()
            for cat_keyword in category_keywords:
                if cat_keyword in text_lower:
                    score += 1
            
            if score > 0:
                category_scores[category] = score
        
        # Return category with highest score
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        
        return None
    
    def _fallback_categorization(self, text):
        """Fallback categorization using simple text matching"""
        
        text_lower = text.lower()
        
        # Direct text matching for common banking queries
        if any(word in text_lower for word in ['card', 'credit', 'debit', 'block', 'stolen']):
            return 'Cards'
        elif any(word in text_lower for word in ['savings', 'account', 'deposit', 'rates', 'interest']):
            return 'Savings & Deposits'
        elif any(word in text_lower for word in ['mortgage', 'home', 'buy', 'refinance']):
            return 'Mortgages'
        elif any(word in text_lower for word in ['pension', 'retirement', 'retire']):
            return 'Pension Plans'
        elif any(word in text_lower for word in ['insurance', 'life', 'coverage']):
            return 'Insurance'
        elif any(word in text_lower for word in ['loan', 'personal', 'borrow']):
            return 'Personal Loans'
        elif any(word in text_lower for word in ['investment', 'invest', 'funds', 'mutual']):
            return 'Investment Funds'
        elif any(word in text_lower for word in ['transfer', 'online', 'mobile', 'app']):
            return 'Digital Banking'
        elif any(word in text_lower for word in ['rewards', 'points', 'loyalty']):
            return 'Customer Rewards'
        elif any(word in text_lower for word in ['account', 'checking', 'open']):
            return 'Bank Accounts'
        
        return 'General Banking'
    
    def predict_and_display(self, text):
        """Predict and display results nicely"""
        
        result = self.predict_banking_category(text)
        
        print(f"Query: {text}")
        
        if result.get('error'):
            print(f"Error: {result['error']}")
            return result
        
        print(f"Category: {result['predicted_category']}")
        
        return result
    
    def batch_predict(self, texts):
        """Predict multiple texts and show summary"""
        
        results = []
        high_conf_count = 0
        medium_conf_count = 0
        
        for text in texts:
            result = self.predict_banking_category(text)
            results.append(result)
            
            if result['confidence_level'] in ['VERY HIGH', 'HIGH']:
                high_conf_count += 1
            elif result['confidence_level'] == 'MEDIUM':
                medium_conf_count += 1
        
        # Show summary
        total = len(texts)
        return results


def test_final_predictor():
    """Test the final production predictor"""
    
    # Initialize predictor
    predictor = FinalBankingPredictor()
    
    if not predictor.load_model():
        print("Failed to load model")
        return None
    
    # Test banking queries
    test_queries = [
        "I need to block my stolen credit card immediately",
        "What are your current savings account interest rates?",
        "Help me apply for a home mortgage loan",
        "I want information about retirement pension plans",
        "How do I transfer money using online banking?",
        "What investment funds and mutual funds do you offer?",
        "I need a personal loan for home renovation",
        "Tell me about life insurance coverage options",
        "What rewards and points does my credit card offer?",
        "How can I open a new business checking account?"
    ]
    
    print("\nTesting Banking Queries:")
    print("=" * 40)
    
    # Test each query
    results = []
    for query in test_queries:
        result = predictor.predict_and_display(query)
        results.append(result)
    
    # Show batch summary
    predictor.batch_predict(test_queries)
    
    return predictor


def interactive_mode(predictor):
    """Interactive mode for testing custom queries"""
    
    
    while True:
        user_input = input("\nYour banking query: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if not user_input:
            continue
        
        predictor.predict_and_display(user_input)
    
    print("\nThanks for testing the Banking Topic Predictor!")


if __name__ == "__main__":
    # Test the final predictor
    predictor = test_final_predictor()
    
    if predictor:
        # Run interactive mode
        interactive_mode(predictor)
