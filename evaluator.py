"""
Banking Topic Classifier Evaluation
==================================
Comprehensive evaluation with Accuracy, Precision, Recall, F1, and Confusion Matrix
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from bertopic import BERTopic
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class BankingTopicEvaluator:
    """Comprehensive evaluator for banking topic classification"""
    
    def __init__(self):
        self.topic_model = None
        self.label_encoder = None
        self.topic_mapping = None
        self.model_loaded = False
        
        # Banking categories for fallback
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
        """Load the production model"""
        model_dir = "advanced_banking_model"
        
        print(f"Loading model from {model_dir}/")
        
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
            print("Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def load_test_data(self, test_size=0.3):
        """Load and split data for evaluation"""
        
        print("Loading banking conversation dataset...")
        
        # Load dataset
        ds = load_dataset("oopere/RetailBanking-Conversations", split="train")
        df = ds.to_pandas()
        
        # Clean data
        df = df.dropna(subset=["rol2", "topic"])
        df = df[df["rol2"].str.len() > 10]  # Remove very short texts
        
        # Use same sample size as training (1200)
        df = df.sample(n=min(1200, len(df)), random_state=42)
        
        print(f"Dataset size: {len(df)} examples")
        print(f"Unique topics: {df['topic'].nunique()}")
        
        # Split data
        texts = df["rol2"].tolist()
        labels = df["topic"].tolist()
        
        # Use the same train/test split as during training
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, 
            random_state=42, stratify=labels
        )
        
        print(f"Test set size: {len(X_test)} examples")
        
        return X_test, y_test
    
    def predict_with_fallback(self, text):
        """Predict with intelligent fallback like production model"""
        
        # Get BERTopic prediction
        topic_ids, probabilities = self.topic_model.transform([text])
        topic_id = topic_ids[0]
        
        if topic_id != -1:
            # Get topic keywords
            topic_words = self.topic_model.get_topic(topic_id)
            keywords = [word for word, _ in topic_words[:5]] if topic_words else []
            
            # Smart category mapping using keywords
            predicted_category = self._map_keywords_to_category(keywords, text)
            
            if predicted_category:
                return predicted_category
            else:
                # Fallback: use direct text analysis
                return self._fallback_categorization(text)
        else:
            # Outlier - use fallback categorization
            return self._fallback_categorization(text)
    
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
    
    def evaluate_model(self, X_test, y_test):
        """Comprehensive model evaluation"""
        
        print("\nEvaluating model performance...")
        print("=" * 50)
        
        # Get predictions
        y_pred = []
        for text in X_test:
            pred = self.predict_with_fallback(text)
            y_pred.append(pred)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        
        # Print overall metrics
        print(f"OVERALL PERFORMANCE METRICS")
        print(f"=" * 30)
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        
        # Detailed classification report
        print(f"\nDETAILED CLASSIFICATION REPORT")
        print(f"=" * 40)
        report = classification_report(y_test, y_pred, zero_division=0)
        print(report)
        
        # Confusion Matrix
        self.plot_confusion_matrix(y_test, y_pred)
        
        # Per-class metrics
        self.print_per_class_metrics(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_true': y_test,
            'y_pred': y_pred
        }
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot and save confusion matrix"""
        
        # Get unique labels
        labels = sorted(list(set(y_true + y_pred)))
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Create plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues',
                    xticklabels=labels,
                    yticklabels=labels,
                    cbar_kws={'label': 'Count'})
        
        plt.title('Banking Topic Classification - Confusion Matrix', fontsize=16, pad=20)
        plt.xlabel('Predicted Category', fontsize=12)
        plt.ylabel('True Category', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix saved as 'confusion_matrix.png'")
        
        # Show plot
        plt.show()
    
    def print_per_class_metrics(self, y_true, y_pred):
        """Print detailed per-class metrics"""
        
        print(f"\nPER-CLASS PERFORMANCE BREAKDOWN")
        print(f"=" * 45)
        
        # Get unique labels
        labels = sorted(list(set(y_true)))
        
        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
        
        # Create DataFrame for better display
        metrics_df = pd.DataFrame({
            'Category': labels,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        })
        
        # Format and display
        print(metrics_df.round(4).to_string(index=False))
        
        # Find best and worst performing categories
        best_f1_idx = np.argmax(f1)
        worst_f1_idx = np.argmin(f1)
        
        print(f"\nBEST PERFORMING CATEGORY:")
        print(f"  {labels[best_f1_idx]}: F1 = {f1[best_f1_idx]:.4f}")
        
        print(f"\nWORST PERFORMING CATEGORY:")
        print(f"  {labels[worst_f1_idx]}: F1 = {f1[worst_f1_idx]:.4f}")


def run_comprehensive_evaluation():
    """Run complete evaluation pipeline"""
    
    print("BANKING TOPIC CLASSIFIER - COMPREHENSIVE EVALUATION")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = BankingTopicEvaluator()
    
    # Load model
    if not evaluator.load_model():
        print("Failed to load model. Please ensure the model is trained.")
        return
    
    # Load test data
    X_test, y_test = evaluator.load_test_data()
    
    # Run evaluation
    results = evaluator.evaluate_model(X_test, y_test)
    
    print(f"\nEVALUATION COMPLETED!")
    print(f"Results saved and confusion matrix plotted.")
    
    return results


if __name__ == "__main__":
    # Run comprehensive evaluation
    results = run_comprehensive_evaluation()
