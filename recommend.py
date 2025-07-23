import pickle
import numpy as np

def load_artifacts(model_path='lightfm_model.pkl', user_map_path='user_map.pkl', item_map_path='item_map.pkl'):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(user_map_path, 'rb') as f:
        user_map = pickle.load(f)
    with open(item_map_path, 'rb') as f:
        item_map = pickle.load(f)
    return model, user_map, item_map

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
    model, user_map, item_map = load_artifacts()
    
    # Replace with any real user ID to test
    test_user_id = float(input("Enter CustomerID for recommendations: "))
    recommend_for_user(model, test_user_id, user_map, item_map)

if __name__ == "__main__":
    main()
