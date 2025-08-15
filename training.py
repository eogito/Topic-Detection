"""
Advanced Supervised BERTopic Training
====================================
High-performance supervised topic modeling with optimized parameters
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from bertopic import BERTopic
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

class AdvancedBankingTrainer:
    """Advanced trainer for high-confidence banking topic classification"""
    
    def __init__(self):
        self.topic_model = None
        self.label_encoder = None
        self.embedding_model = None
        self.vectorizer_model = None
        self.umap_model = None
        self.hdbscan_model = None
        
    def setup_advanced_models(self):
        """Setup optimized models for better performance"""
        
        print("🔧 Setting up advanced models...")
        
        # Use a better sentence transformer model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Optimized UMAP for better clustering
        self.umap_model = UMAP(
            n_neighbors=15,          # More neighbors for stability
            n_components=5,          # More dimensions for complex data
            min_dist=0.0,           # Tight clusters
            metric='cosine',        # Good for text
            random_state=42
        )
        
        # Optimized HDBSCAN for supervised learning
        self.hdbscan_model = HDBSCAN(
            min_cluster_size=10,     # Larger minimum cluster size
            min_samples=5,           # More samples for core points
            metric='euclidean',      # Good for UMAP output
            cluster_selection_method='eom',  # Excess of mass method
            prediction_data=True     # Enable prediction for new data
        )
        
        # Better vectorizer for keywords
        self.vectorizer_model = CountVectorizer(
            ngram_range=(1, 2),      # Include bigrams
            stop_words="english",    # Remove stop words
            min_df=2,               # Remove very rare words
            max_df=0.95             # Remove very common words
        )
        
        print("✅ Advanced models configured!")
    
    def load_and_prepare_data(self, sample_size=None):
        """Load and prepare the banking dataset"""
        
        print("📊 Loading banking conversation dataset...")
        
        # Load dataset
        ds = load_dataset("oopere/RetailBanking-Conversations", split="train")
        df = ds.to_pandas()
        
        # Clean data
        df = df.dropna(subset=["rol2", "topic"])
        df = df[df["rol2"].str.len() > 10]  # Remove very short texts
        
        # Use more data for better training
        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        print(f"📈 Dataset size: {len(df)} examples")
        print(f"📋 Unique topics: {df['topic'].nunique()}")
        
        # Show topic distribution
        topic_counts = df['topic'].value_counts()
        print("\n📊 Topic distribution:")
        for topic, count in topic_counts.items():
            print(f"   {topic}: {count} examples")
        
        return df
    
    def train_supervised_model(self, df, test_size=0.2):
        """Train supervised BERTopic model with advanced configuration"""
        
        print("\n🚀 Training Advanced Supervised BERTopic Model")
        print("=" * 50)
        
        # Prepare data
        texts = df["rol2"].tolist()
        labels = df["topic"].tolist()
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        numeric_labels = self.label_encoder.fit_transform(labels)
        
        print(f"📊 Training on {len(texts)} examples")
        print(f"🏷️ Number of classes: {len(self.label_encoder.classes_)}")
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(
            texts, numeric_labels, test_size=test_size, 
            random_state=42, stratify=numeric_labels
        )
        
        print(f"📈 Training set: {len(X_train)} examples")
        print(f"📉 Test set: {len(X_test)} examples")
        
        # Create advanced BERTopic model
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            vectorizer_model=self.vectorizer_model,
            language="english",
            calculate_probabilities=True,
            verbose=True
        )
        
        # Train the model
        print("\n🔥 Training model (this may take a few minutes)...")
        topics, probabilities = self.topic_model.fit_transform(X_train, y=y_train)
        
        # Evaluate on test set
        print("\n📊 Evaluating model performance...")
        test_topics, test_probs = self.topic_model.transform(X_test)
        
        # Calculate accuracy (excluding outliers)
        valid_mask = test_topics != -1
        print(f"🔍 Debug - valid_mask sum: {np.sum(valid_mask)}")
        print(f"🔍 Debug - test_topics shape: {np.array(test_topics).shape}")
        print(f"🔍 Debug - y_test shape: {np.array(y_test).shape}")
        
        if np.sum(valid_mask) > 0:
            y_test_valid = np.array(y_test)[valid_mask]
            test_topics_valid = np.array(test_topics)[valid_mask]
            
            print(f"🔍 Debug - y_test_valid shape: {y_test_valid.shape}")
            print(f"🔍 Debug - test_topics_valid shape: {test_topics_valid.shape}")
            print(f"🔍 Debug - y_test_valid sample: {y_test_valid[:5]}")
            print(f"🔍 Debug - test_topics_valid sample: {test_topics_valid[:5]}")
            
            # Simple accuracy calculation
            if len(y_test_valid) > 0 and len(test_topics_valid) > 0:
                accuracy = np.mean(y_test_valid == test_topics_valid)
            else:
                accuracy = 0.0
                
            outlier_rate = np.mean(test_topics == -1)
            
            print(f"✅ Model trained successfully!")
            print(f"🎯 Test Accuracy: {accuracy:.3f}")
            print(f"🏷️ Outlier Rate: {outlier_rate:.3f}")
            
            # Calculate confidence statistics
            if test_probs is not None:
                valid_probs = test_probs[valid_mask]
                if len(valid_probs) > 0:
                    max_probs = np.max(valid_probs, axis=1)
                    mean_confidence = np.mean(max_probs)
                    high_conf_rate = np.mean(max_probs > 0.5)
                    
                    print(f"📈 Mean Confidence: {mean_confidence:.3f}")
                    print(f"🎪 High Confidence Rate (>0.5): {high_conf_rate:.3f}")
        
        return X_test, y_test, test_topics, test_probs
    
    def save_advanced_model(self):
        """Save the advanced model and all components"""
        
        save_dir = "advanced_banking_model"
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n💾 Saving advanced model to {save_dir}/")
        
        try:
            # Save BERTopic model
            model_path = os.path.join(save_dir, "bertopic_model")
            self.topic_model.save(model_path, serialization="pickle")
            print("✅ BERTopic model saved")
            
            # Save label encoder
            with open(os.path.join(save_dir, "label_encoder.pkl"), 'wb') as f:
                pickle.dump(self.label_encoder, f)
            print("✅ Label encoder saved")
            
            # Save topic mapping
            topic_mapping = {
                'topic_names': self.label_encoder.classes_.tolist(),
                'total_topics': len(self.label_encoder.classes_),
                'model_type': 'advanced_supervised_bertopic'
            }
            
            with open(os.path.join(save_dir, "topic_mapping.pkl"), 'wb') as f:
                pickle.dump(topic_mapping, f)
            print("✅ Topic mapping saved")
            
            # Save model configuration
            model_info = {
                'embedding_model': 'all-MiniLM-L6-v2',
                'umap_neighbors': 15,
                'umap_components': 5,
                'hdbscan_min_cluster_size': 10,
                'hdbscan_min_samples': 5,
                'vectorizer_ngrams': [1, 2],
                'model_version': 'advanced_v1'
            }
            
            with open(os.path.join(save_dir, "model_info.json"), 'w') as f:
                json.dump(model_info, f, indent=2)
            print("✅ Model info saved")
            
            print(f"🎉 All components saved successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def show_model_insights(self):
        """Show detailed model insights"""
        
        if self.topic_model is None:
            print("❌ No model trained yet")
            return
        
        print("\n📊 Model Insights")
        print("=" * 30)
        
        # Get topic info
        topic_info = self.topic_model.get_topic_info()
        
        # Show topic distribution
        print(f"📈 Total topics discovered: {len(topic_info)}")
        print(f"📋 Valid topics (non-outliers): {len(topic_info[topic_info['Topic'] != -1])}")
        
        # Show top topics
        print("\n🏆 Top 10 Topics:")
        valid_topics = topic_info[topic_info['Topic'] != -1].head(10)
        
        for _, row in valid_topics.iterrows():
            topic_id = row['Topic']
            count = row['Count']
            
            # Get topic name
            topic_name = "Unknown"
            if self.label_encoder and topic_id < len(self.label_encoder.classes_):
                topic_name = self.label_encoder.classes_[topic_id]
            
            # Get keywords
            keywords = self.topic_model.get_topic(topic_id)
            if keywords:
                top_words = [word for word, _ in keywords[:3]]
                print(f"   Topic {topic_id} ({topic_name}): {count} docs - {', '.join(top_words)}")


def run_advanced_training():
    """Run the complete advanced training pipeline"""
    
    print("🏦 Advanced Banking Topic Model Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = AdvancedBankingTrainer()
    
    # Setup advanced models
    trainer.setup_advanced_models()
    
    # Load data (use more data for better performance)
    df = trainer.load_and_prepare_data(sample_size=1200)  # Use more data
    
    # Train supervised model
    X_test, y_test, test_topics, test_probs = trainer.train_supervised_model(df)
    
    # Show insights
    trainer.show_model_insights()
    
    # Save model
    if trainer.save_advanced_model():
        print("\n🎉 Advanced model training completed successfully!")
        
        # Test with sample queries
        print("\n🧪 Quick Test:")
        test_queries = [
            "I want to block my credit card",
            "What are savings account rates?", 
            "Help with mortgage application"
        ]
        
        for query in test_queries:
            topic_ids, probs = trainer.topic_model.transform([query])
            topic_id = topic_ids[0]
            
            if topic_id != -1:
                confidence = probs[0][topic_id] if probs is not None else 0.0
                topic_name = trainer.label_encoder.classes_[topic_id] if topic_id < len(trainer.label_encoder.classes_) else "Unknown"
                print(f"   '{query}' → {topic_name} (confidence: {confidence:.3f})")
            else:
                print(f"   '{query}' → Outlier")
    
    return trainer


if __name__ == "__main__":
    # Run the advanced training
    trainer = run_advanced_training()
