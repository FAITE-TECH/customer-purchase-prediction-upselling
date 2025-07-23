import os
import pandas as pd
import numpy as np
from lightfm import LightFM
from scipy.sparse import coo_matrix
import pickle

def load_data(filepath="Assignment-1_Data.csv"):
    print("Loading dataset...")
    df = pd.read_csv(
        filepath,
        sep=';',              # semicolon separator
        decimal=',',          # comma decimal separator for Price
        parse_dates=['Date'], # parse the Date column automatically
        dayfirst=True         # day comes before month in date format
    )
    print(f"Loaded {len(df)} records")
    return df

def preprocess_data(df):
    print("Preprocessing data: dropping rows with missing CustomerID or Itemname")
    df = df.dropna(subset=['CustomerID', 'Itemname'])
    df['user_id'] = df['CustomerID'].astype('category').cat.codes
    df['item_id'] = df['Itemname'].astype('category').cat.codes
    return df

def create_interactions(df):
    print("Creating interaction matrix...")
    interactions = coo_matrix(
        (df['Quantity'], (df['user_id'], df['item_id']))
    )
    print(f"Interaction matrix shape: {interactions.shape}")
    return interactions

def train_model(interactions, epochs=10, num_threads=4):
    print("Training LightFM model...")
    model = LightFM(loss='warp')
    model.fit(interactions, epochs=epochs, num_threads=num_threads)
    print("Training completed.")
    return model

def build_mappings(df):
    print("Building user and item mappings...")
    user_map = dict(zip(df['CustomerID'], df['user_id']))
    item_map = dict(zip(df['Itemname'], df['item_id']))
    return user_map, item_map

def save_artifacts(model, user_map, item_map, model_path='lightfm_model.pkl', user_map_path='user_map.pkl', item_map_path='item_map.pkl'):
    print("Saving model and mappings...")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(user_map_path, 'wb') as f:
        pickle.dump(user_map, f)
    with open(item_map_path, 'wb') as f:
        pickle.dump(item_map, f)
    print(f"Saved model to {model_path}")
    print(f"Saved user_map to {user_map_path}")
    print(f"Saved item_map to {item_map_path}")

def recommend_for_user(model, real_user_id, user_map, item_map, num_recommendations=5):
    if real_user_id not in user_map:
        print(f"User ID {real_user_id} not found in data.")
        return
    
    user_internal_id = user_map[real_user_id]
    n_items = len(item_map)
    
    scores = model.predict(user_internal_id, np.arange(n_items))
    top_items = np.argsort(-scores)[:num_recommendations]
    
    inv_item_map = {v: k for k, v in item_map.items()}
    recommended_items = [inv_item_map[i] for i in top_items]
    
    print(f"Top {num_recommendations} recommendations for user {real_user_id}:")
    for i, item in enumerate(recommended_items, 1):
        print(f"{i}. {item}")

def main():
    df = load_data()
    df = preprocess_data(df)
    interactions = create_interactions(df)
    model = train_model(interactions)
    user_map, item_map = build_mappings(df)
    save_artifacts(model, user_map, item_map)

    # Example: Recommend for a sample user
    sample_user = list(user_map.keys())[0]
    recommend_for_user(model, sample_user, user_map, item_map)

if __name__ == "__main__":
    main()
