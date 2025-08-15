from datasets import load_dataset
import pandas as pd
from bertopic import BERTopic

# 1. Load the RetailBanking─Conversations dataset
ds = load_dataset("oopere/RetailBanking-Conversations", split="train")

# 2. Convert to pandas DataFrame
df = ds.to_pandas()

# 3. Prepare text and labels
df = df.dropna(subset=["rol2", "topic"])
texts = df["rol2"].tolist()
labels = df["topic"].tolist()

# Convert string labels to numeric labels for supervised learning
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
numeric_labels = label_encoder.fit_transform(labels)

print(f"Total examples: {len(texts)} with {len(set(labels))} topics.")
print(f"Label mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

# 4. Train supervised BERTopic
topic_model = BERTopic(language="english", calculate_probabilities=True, verbose=True)
topics, probs = topic_model.fit_transform(texts, y=numeric_labels)

# 5. Examine learned topic representations
info = topic_model.get_topic_info()
print(info)

# 6. Inspect the keywords per topic
for topic_id in info["Topic"].unique():
    print(topic_id, topic_model.get_topic(topic_id))

# 7. Save model and label encoder
import pickle
import os

# Save the model using multiple methods for reliability
model_dir = "saved_model"
os.makedirs(model_dir, exist_ok=True)

# Method 1: Save using BERTopic's built-in save method
try:
    model_path = os.path.join(model_dir, "bertopic_model")
    topic_model.save(model_path)
    print(f"✅ BERTopic model saved to: {model_path}")
except Exception as e:
    print(f"❌ Could not save BERTopic model: {e}")

# Method 2: Save the label encoder separately (important for mapping back to original labels)
try:
    label_encoder_path = os.path.join(model_dir, "label_encoder.pkl")
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"✅ Label encoder saved to: {label_encoder_path}")
except Exception as e:
    print(f"❌ Could not save label encoder: {e}")

# Method 3: Save topic mapping for reference
try:
    topic_mapping = {
        'label_to_numeric': dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))),
        'numeric_to_label': dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_)),
        'total_topics': len(label_encoder.classes_),
        'topic_names': list(label_encoder.classes_)
    }
    
    mapping_path = os.path.join(model_dir, "topic_mapping.pkl")
    with open(mapping_path, 'wb') as f:
        pickle.dump(topic_mapping, f)
    print(f"✅ Topic mapping saved to: {mapping_path}")
except Exception as e:
    print(f"❌ Could not save topic mapping: {e}")

# Method 4: Save model info as JSON for easy inspection
try:
    import json
    model_info = {
        'total_documents': len(texts),
        'unique_topics': len(set(labels)),
        'topic_names': list(label_encoder.classes_),
        'model_creation_date': str(pd.Timestamp.now()),
        'bertopic_version': BERTopic.__version__ if hasattr(BERTopic, '__version__') else 'unknown'
    }
    
    info_path = os.path.join(model_dir, "model_info.json")
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2, default=str)
    print(f"✅ Model info saved to: {info_path}")
except Exception as e:
    print(f"❌ Could not save model info: {e}")

print(f"\n📁 All model files saved in directory: {os.path.abspath(model_dir)}")

# 8. Use it to predict for new texts
new = ["I want to block my card immediately"]
pred, prob = topic_model.transform(new)
print(pred, prob)
