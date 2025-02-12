import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import time
import matplotlib.pyplot as plt


def filter_users_with_min_reviews(merged_df, min_reviews=50):
    """
    Filter the merged DataFrame to include only users with at least min_reviews.
    """
    user_counts = merged_df.groupby('user_id').size()
    valid_users = user_counts[user_counts >= min_reviews].index
    return merged_df[merged_df['user_id'].isin(valid_users)].copy()


def leave_10_out_split(ratings_df, test_items=10, random_state=42):
    """
    For each user, randomly sample `test_items` ratings to be held out as the test set.
    Returns two DataFrames: one for training and one for testing.
    """
    train_list = []
    test_list = []
    np.random.seed(random_state)
    
    # Process each user's ratings individually.
    for user, group in ratings_df.groupby('user_id'):
        if len(group) >= test_items:
            group_test = group.sample(n=test_items, random_state=random_state)
            group_train = group.drop(group_test.index)
            train_list.append(group_train)
            test_list.append(group_test)
        else:
            # Optionally, skip users with fewer than test_items ratings.
            print(f"User {user} has fewer than {test_items} ratings; skipping in split.")
    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    return train_df, test_df



def build_model_custom(merged_df, lambda_reg=20, n_components=100, num_iter=10):
    """
    Build the recommendation model from the training data:
      - Create the user–item rating matrix R (using np.float32 for memory efficiency).
      - Compute the global mean.
      - Iteratively estimate user biases (b_u) and item biases (b_i) with regularization.
      - Compute the residual matrix and apply Truncated SVD to extract latent factors.
      
    Returns a dictionary containing the model parameters.
    """
    # Create mappings for users and items.
    unique_users = merged_df['user_id'].unique()
    unique_items = merged_df['anime_id'].unique()
    user2index = {user: idx for idx, user in enumerate(unique_users)}
    item2index = {anime: idx for idx, anime in enumerate(unique_items)}
    num_users = len(unique_users)
    num_items = len(unique_items)
    
    # Build the rating matrix R and a binary mask.
    R = np.zeros((num_users, num_items), dtype=np.float32)
    mask = np.zeros((num_users, num_items), dtype=np.float32)
    for row in merged_df.itertuples():
        u_idx = user2index[row.user_id]
        i_idx = item2index[row.anime_id]
        R[u_idx, i_idx] = row.user_rating
        mask[u_idx, i_idx] = 1.0
        
    # Global mean (computed only on observed ratings).
    global_mean = np.sum(R) / np.sum(mask)
    
    # -------------------------------
    # Regularized Bias Estimation
    # -------------------------------
    b_u = np.zeros(num_users, dtype=np.float32)  # User biases.
    b_i = np.zeros(num_items, dtype=np.float32)  # Item biases.
    
    for it in range(num_iter):
        # Update item biases.
        for i in range(num_items):
            idx = np.where(mask[:, i] == 1)[0]
            if len(idx) > 0:
                b_i[i] = np.sum(R[idx, i] - global_mean - b_u[idx]) / (lambda_reg + len(idx))
        # Update user biases.
        for u in range(num_users):
            idx = np.where(mask[u, :] == 1)[0]
            if len(idx) > 0:
                b_u[u] = np.sum(R[u, idx] - global_mean - b_i[idx]) / (lambda_reg + len(idx))
    
    # -------------------------------
    # Compute Residuals and Apply SVD
    # -------------------------------
    print("Performing Truncated SVD on residuals (lambda_reg={} and n_components={})...".format(lambda_reg, n_components))
    residual = mask * (R - global_mean - b_u[:, np.newaxis] - b_i[np.newaxis, :])
    svd_start = time.time()
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    U_svd = svd.fit_transform(residual)
    Sigma = svd.singular_values_
    VT = svd.components_
    residual_approx = np.dot(U_svd, np.dot(np.diag(Sigma), VT))
    print("SVD completed in {:.2f} seconds.".format(time.time() - svd_start))
    
    model = {
        'global_mean': global_mean,
        'b_u': b_u,
        'b_i': b_i,
        'residual_approx': residual_approx,
        'user2index': user2index,
        'item2index': item2index,
        'R': R,
        'mask': mask
    }
    return model

def predict_rating(user_id, anime_id, model):
    """
    Predict the rating for a given user and anime using the model.
    The predicted rating is:
         global_mean + b_u + b_i + latent_component.
    If a user or anime is unknown, return the global mean.
    """
    global_mean = model['global_mean']
    b_u = model['b_u']
    b_i = model['b_i']
    residual_approx = model['residual_approx']
    user2index = model['user2index']
    item2index = model['item2index']
    
    if user_id not in user2index or anime_id not in item2index:
        return global_mean
    u_idx = user2index[user_id]
    i_idx = item2index[anime_id]
    pred = global_mean + b_u[u_idx] + b_i[i_idx] + residual_approx[u_idx, i_idx]
    return np.clip(pred, 1, 10)


def top_n_for_user(user_id, candidate_items, model, top_n=5):
    """
    For a given user, compute predicted ratings for each candidate item
    and return the top_n items ranked by predicted rating.
    """
    preds = {}
    for anime in candidate_items:
        preds[anime] = predict_rating(user_id, anime, model)
    ranked = sorted(preds.items(), key=lambda x: x[1], reverse=True)
    return [anime for anime, score in ranked[:top_n]]


def evaluate_top_n(test_df, candidate_items, model, top_n=5, threshold=7.5):
    """
    Evaluate top-N recommendations on the test set.
    An item is considered relevant if the user_rating >= threshold.
    Computes Precision@N and Recall@N for each user and returns the average.
    """
    user_metrics = []
    for user, group in test_df.groupby('user_id'):
        relevant_items = set(group[group['user_rating'] >= threshold]['anime_id'])
        if len(relevant_items) == 0:
            continue
        recs = set(top_n_for_user(user, candidate_items, model, top_n=top_n))
        precision = len(recs & relevant_items) / float(top_n)
        recall = len(recs & relevant_items) / float(len(relevant_items))
        user_metrics.append((precision, recall))
    if not user_metrics:
        return None, None
    avg_precision = np.mean([m[0] for m in user_metrics])
    avg_recall = np.mean([m[1] for m in user_metrics])
    return avg_precision, avg_recall


def grid_search(train_df, test_df, candidate_lambdas, candidate_n_components, top_n=10, threshold=7.5):
    """
    Perform a grid search over lambda (regularization) and number of latent factors.
    For each combination, build a model on train_df and evaluate on test_df.
    Returns a list of tuples: (lambda, n_components, avg_precision, avg_recall)
    """
    results = []
    candidate_items = set(train_df['anime_id'].unique())
    for lam in candidate_lambdas:
        for n_comp in candidate_n_components:
            print("Evaluating lambda={} and n_components={}".format(lam, n_comp))
            model = build_model_custom(train_df, lambda_reg=lam, n_components=n_comp)
            avg_precision, avg_recall = evaluate_top_n(test_df, candidate_items, model, top_n=top_n, threshold=threshold)
            results.append((lam, n_comp, avg_precision, avg_recall))
            print("lambda: {}, n_components: {} => Precision@{}: {:.4f}, Recall@{}: {:.4f}".format(
                lam, n_comp, top_n, avg_precision, top_n, avg_recall))
    return results

###############################################
# Main Pipeline and Evaluation
###############################################

# Load data.
anime_df = pd.read_csv('improved_anime.csv')
ratings_df = pd.read_csv('rating.csv')

# Remove duplicate entries.
anime_df.drop_duplicates(subset='anime_id', inplace=True)
ratings_df.drop_duplicates(inplace=True)

# Drop rows with missing critical data in anime dataset.
anime_df.dropna(subset=['name', 'genre', 'type', 'episodes', 'rating', 'members', 'Scored By'], inplace=True)
anime_df['episodes'] = pd.to_numeric(anime_df['episodes'], errors='coerce')
anime_df['Scored By'] = pd.to_numeric(anime_df['Scored By'], errors='coerce')
anime_df.dropna(subset=['episodes', 'Scored By'], inplace=True)

# Drop ratings that are placeholder (-1).
ratings_df = ratings_df[ratings_df['rating'] != -1]

# Filter out shows with less than 1000 reviews.
print("Removing shows with less than 1000 reviews...")
print("Previous anime shape:", anime_df.shape)
anime_df = anime_df[anime_df['Scored By'] > 1000]
print("Anime shape:", anime_df.shape)

# Merge datasets.
merged_df = pd.merge(ratings_df, anime_df, on='anime_id', how='inner')
if 'rating_x' in merged_df.columns:
    merged_df.rename(columns={'rating_x': 'user_rating'}, inplace=True)
else:
    merged_df.rename(columns={'rating': 'user_rating'}, inplace=True)

# Filter to keep only users with at least 50 reviews.
merged_df = filter_users_with_min_reviews(merged_df, min_reviews=50)
print("Filtered merged_df shape:", merged_df.shape)

# Remove content columns.
merged_df = merged_df[['user_id', 'anime_id', 'user_rating']].copy()

# Split data: For each user, randomly hold out 10 ratings.
train_df, test_df = leave_10_out_split(merged_df, test_items=10, random_state=42)
print("Train shape:", train_df.shape, "Test shape:", test_df.shape)

###############################################
# Run Grid Search
###############################################

# Define candidate hyperparameter values.
candidate_lambdas = [0.1, 1, 5, 10, 20, 50]     
candidate_n_components = [20, 50, 100, 150]       

grid_results = grid_search(train_df, test_df, candidate_lambdas, candidate_n_components, top_n=10, threshold=7.5)

print("\nGrid Search Results:")
for res in grid_results:
    lam, n_comp, prec, rec = res
    print("lambda: {}, n_components: {} => Precision@10: {:.4f}, Recall@10: {:.4f}".format(lam, n_comp, prec, rec))
