import os
import glob
import pandas as pd
import difflib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import kagglehub

# Display settings for terminal
PD_DISPLAY_ROWS = 8
PD_DISPLAY_COLS = 200
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)
pd.set_option("display.max_colwidth", 40)

# -----------------------------
# 1. Download & Load Datasets (Kaggle)
# -----------------------------
# Store games metadata
store_path = kagglehub.dataset_download("nikdavis/steam-store-games")
store_csv_files = glob.glob(os.path.join(store_path, "*.csv"))
if not store_csv_files:
    raise FileNotFoundError(f"No CSV files found in steam-store-games dataset at {store_path}")
store_csv = store_csv_files[0]

# Reviews / interactions (use tamber/steam-video-games)
reviews_path = kagglehub.dataset_download("tamber/steam-video-games")
reviews_csv_files = glob.glob(os.path.join(reviews_path, "*.csv"))
if not reviews_csv_files:
    reviews_csv_files = glob.glob(os.path.join(reviews_path, "**", "*.csv"), recursive=True)
if not reviews_csv_files:
    reviews_csv_files = glob.glob(os.path.join(reviews_path, "*.dat"))
    if not reviews_csv_files:
        reviews_csv_files = glob.glob(os.path.join(reviews_path, "**", "*.dat"), recursive=True)
if not reviews_csv_files:
    raise FileNotFoundError(f"No CSV or DAT files found in tamber/steam-video-games dataset at {reviews_path}")
reviews_csv = reviews_csv_files[0]

# Load games metadata
games_df = pd.read_csv(store_csv)

# =============================
# DATA DISPLAY: Raw metadata (store-games)
# =============================
print("\n" + "=" * 60)
print("  RAW DATASET 1: Steam Store Games (Metadata)")
print("=" * 60)
print(f"Shape: {games_df.shape[0]} rows x {games_df.shape[1]} columns")
print(f"Columns: {list(games_df.columns)}")
print("\nFirst 5 rows (relevant columns):")
cols_meta = [c for c in ['name', 'genres', 'categories', 'steamspy_tags', 'developer', 'publisher'] if c in games_df.columns]
print(games_df[cols_meta].head(5).to_string())
print(f"\nMissing values per column:\n{games_df[cols_meta].isna().sum()}")
print("=" * 60)

# -----------------------------
# 2. Data cleaning: Metadata
# -----------------------------
df = games_df[['name', 'genres', 'categories', 'steamspy_tags', 'developer', 'publisher']].copy()
# Trim whitespace in string columns
for c in df.select_dtypes(include=['object']).columns:
    df[c] = df[c].astype(str).str.strip()
df.fillna('', inplace=True)
before_dedup = len(df)
df.drop_duplicates(subset='name', inplace=True)
df.reset_index(drop=True, inplace=True)

# =============================
# DATA DISPLAY: Cleaned metadata
# =============================
print("\n" + "=" * 60)
print("  CLEANED DATASET 1: Metadata (after cleaning)")
print("=" * 60)
print(f"Steps: trim whitespace, fillna(''), drop_duplicates(subset='name')")
print(f"Rows removed (duplicates): {before_dedup - len(df)}")
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head(5).to_string())
print("=" * 60)

# -----------------------------
# 3. Combine Text Features (Content-Based)
# -----------------------------
df['combined_features'] = (
    df['genres'] + ' ' +
    df['categories'] + ' ' +
    df['steamspy_tags'] + ' ' +
    df['developer'] + ' ' +
    df['publisher']
)

# -----------------------------
# 4. TF-IDF Vectorization (normalization for content-based)
# -----------------------------
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# =============================
# DATA DISPLAY: Normalized metadata (TF-IDF)
# =============================
print("\n" + "=" * 60)
print("  NORMALIZED DATASET 1: TF-IDF vectors (content-based)")
print("=" * 60)
print("Normalization: TF-IDF + L2 norm (cosine similarity uses normalized vectors)")
print(f"Matrix shape: {tfidf_matrix.shape[0]} games x {tfidf_matrix.shape[1]} features")
dense_sample = tfidf_matrix[:3].toarray()
print("Sample (first 3 games, first 10 features):")
print(pd.DataFrame(dense_sample[:, :10]).to_string())
print("=" * 60)

# -----------------------------
# 5. Cosine Similarity
# -----------------------------
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['name']).drop_duplicates()

# -----------------------------
# 5a. Fuzzy Matching (used by CF cleaning)
# -----------------------------
def find_closest_game(game_name):
    matches = difflib.get_close_matches(
        game_name,
        df['name'].tolist(),
        n=1,
        cutoff=0.4
    )
    return matches[0] if matches else None

# -----------------------------
# 5b. Collaborative Filtering Data (from interactions dataset)
# -----------------------------
try:
    # Handle .dat files (tamber dataset format) - usually tab or space separated
    if reviews_csv.endswith('.dat'):
        try:
            reviews_df = pd.read_csv(reviews_csv, sep='\t', header=None,
                                   names=['user_id', 'name', 'behavior-name', 'value'])
        except Exception:
            reviews_df = pd.read_csv(reviews_csv, sep=' ', header=None,
                                   names=['user_id', 'name', 'behavior-name', 'value'])
    else:
        # steam-200k.csv has no header - it's: user-id, game-title, behavior-name, value, hours
        reviews_df = pd.read_csv(reviews_csv, header=None,
                                names=['user-id', 'game-title', 'behavior-name', 'value', 'hours'])

    # ========== RAW INTERACTIONS DISPLAY ==========
    print("\n" + "=" * 60)
    print("  RAW DATASET 2: Steam Interactions (user-game)")
    print("=" * 60)
    print(f"Shape: {reviews_df.shape[0]} rows x {reviews_df.shape[1]} columns")
    print(f"Columns: {list(reviews_df.columns)}")
    print("\nFirst 8 rows:")
    print(reviews_df.head(8).to_string())
    print(f"\nMissing values:\n{reviews_df.isna().sum()}")
    print("=" * 60)

    # Standardise user and game columns if present
    # Common schemas:
    # - andrewmvd/steam-reviews: author_steamid, app_name, recommended
    # - tamber/steam-video-games (steam-200k): user-id, game-title, behavior-name, value
    if 'author_steamid' in reviews_df.columns and 'user_id' not in reviews_df.columns:
        reviews_df = reviews_df.rename(columns={'author_steamid': 'user_id'})
    if 'user-id' in reviews_df.columns and 'user_id' not in reviews_df.columns:
        reviews_df = reviews_df.rename(columns={'user-id': 'user_id'})

    if 'app_name' in reviews_df.columns and 'name' not in reviews_df.columns:
        reviews_df = reviews_df.rename(columns={'app_name': 'name'})
    if 'game-title' in reviews_df.columns and 'name' not in reviews_df.columns:
        reviews_df = reviews_df.rename(columns={'game-title': 'name'})

    # Ensure required columns exist
    if 'user_id' in reviews_df.columns and 'name' in reviews_df.columns:
        interactions_df = reviews_df.copy()

        # Cleaning: keep only games that exist in metadata
        interactions_df_filtered = interactions_df[interactions_df['name'].isin(df['name'])]
        if len(interactions_df_filtered) == 0:
            name_mapping = {}
            for game_name in interactions_df['name'].unique():
                match = find_closest_game(game_name)
                if match:
                    name_mapping[game_name] = match
            if name_mapping:
                interactions_df['name'] = interactions_df['name'].map(name_mapping).fillna(interactions_df['name'])
                interactions_df_filtered = interactions_df[interactions_df['name'].isin(df['name'])]
        interactions_df = interactions_df_filtered

        if len(interactions_df) == 0:
            interactions_df = None
            user_item_matrix = None
            item_user_matrix = None
            item_cf_sim = None
            cf_indices = None
        else:
            if 'recommended' in interactions_df.columns:
                interactions_df = interactions_df[
                    interactions_df['recommended'].astype(str).str.lower().isin(
                        ['true', '1', 'recommended', 'yes']
                    )
                ]
            if 'behavior-name' in interactions_df.columns:
                interactions_df = interactions_df[
                    interactions_df['behavior-name'].isin(['purchase', 'play'])
                ]

            # ========== CLEANED INTERACTIONS DISPLAY ==========
            print("\n" + "=" * 60)
            print("  CLEANED DATASET 2: Interactions (after cleaning)")
            print("=" * 60)
            print("Steps: column rename, keep games in metadata, filter behavior (purchase/play)")
            print(f"Shape: {len(interactions_df)} rows")
            print(f"Unique users: {interactions_df['user_id'].nunique()}, unique games: {interactions_df['name'].nunique()}")
            print("\nFirst 8 rows:")
            print(interactions_df.head(8).to_string())
            print("=" * 60)

            # Normalization: rating 0-1 (use hours if present, else binary 1.0)
            if 'hours' in interactions_df.columns:
                interactions_df = interactions_df.copy()
                interactions_df['hours'] = pd.to_numeric(interactions_df['hours'], errors='coerce').fillna(0)
                h = interactions_df['hours']
                min_h, max_h = h.min(), h.max()
                if max_h > min_h:
                    interactions_df['rating'] = (h - min_h) / (max_h - min_h)
                else:
                    interactions_df['rating'] = 1.0
            else:
                interactions_df = interactions_df.copy()
                interactions_df['rating'] = 1.0
            interactions_df = interactions_df[['user_id', 'name', 'rating']].copy()

            # ========== NORMALIZED INTERACTIONS DISPLAY ==========
            print("\n" + "=" * 60)
            print("  NORMALIZED DATASET 2: Interactions (rating 0-1)")
            print("=" * 60)
            print("Normalization: rating = min-max scaled (0-1) from hours if present, else 1.0")
            print("\nFirst 8 rows (user_id, name, rating):")
            print(interactions_df.head(8).to_string())
            print(f"\nRating stats: min={interactions_df['rating'].min():.4f}, max={interactions_df['rating'].max():.4f}, mean={interactions_df['rating'].mean():.4f}")
            print("=" * 60)

            # Train / Validation / Test split (70% / 15% / 15%)
            interactions_shuffled = interactions_df.sample(frac=1, random_state=42).reset_index(drop=True)
            train_size = int(0.70 * len(interactions_shuffled))
            val_size = int(0.15 * len(interactions_shuffled))
            interactions_train = interactions_shuffled.iloc[:train_size]
            interactions_val = interactions_shuffled.iloc[train_size:train_size + val_size]
            interactions_test = interactions_shuffled.iloc[train_size + val_size:]


            # ========== TRAIN/VAL/TEST SPLIT DISPLAY ==========
            print("\n" + "=" * 60)
            print("  TRAIN / VALIDATION / TEST SPLIT")
            print("=" * 60)
            print(f"Total interactions: {len(interactions_df)}")
            print(f"  Training   (70%): {len(interactions_train)} rows")
            print(f"  Validation (15%): {len(interactions_val)} rows")
            print(f"  Test       (15%): {len(interactions_test)} rows")
            print("\nTraining set - first 5 rows:")
            print(interactions_train.head(5).to_string())
            print("\nValidation set - first 3 rows:")
            print(interactions_val.head(3).to_string())
            print("\nTest set - first 3 rows:")
            print(interactions_test.head(3).to_string())
            print("=" * 60)

            # Build user–item matrix from TRAINING set only
            user_item_matrix = interactions_train.pivot_table(
                index='user_id',
                columns='name',
                values='rating',
                aggfunc='mean'
            ).fillna(0.0)

            item_user_matrix = user_item_matrix.T
            item_cf_sim = cosine_similarity(item_user_matrix, item_user_matrix)
            cf_indices = pd.Series(
                index=item_user_matrix.index,
                data=np.arange(len(item_user_matrix.index))
            )
    else:
        interactions_df = None
        user_item_matrix = None
        item_user_matrix = None
        item_cf_sim = None
        cf_indices = None
except FileNotFoundError as e:
    interactions_df = None
    user_item_matrix = None
    item_user_matrix = None
    item_cf_sim = None
    cf_indices = None
except Exception as e:
    print(f"Error loading collaborative filtering data: {type(e).__name__}: {e}")
    interactions_df = None
    user_item_matrix = None
    item_user_matrix = None
    item_cf_sim = None
    cf_indices = None

# -----------------------------
# 6. Build User Preference Vector
# -----------------------------
def build_user_profile(game_list):
    vectors = []
    matched_games = []

    for game in game_list:
        match = find_closest_game(game)
        if match:
            matched_games.append(match)
            # Convert sparse matrix to dense array
            vectors.append(tfidf_matrix[indices[match]].toarray()[0])

    if not vectors:
        return None, None

    # Average all user vectors and reshape for cosine similarity
    user_profile = np.mean(vectors, axis=0).reshape(1, -1)
    return user_profile, matched_games

# -----------------------------
# 8. Generate Recommendations
# -----------------------------
def recommend_games_from_profile(user_profile, excluded_games, top_n=5):
    similarities = cosine_similarity(user_profile, tfidf_matrix)[0]

    recommendations = sorted(
        enumerate(similarities),
        key=lambda x: x[1],
        reverse=True
    )

    results = []
    for idx, score in recommendations:
        game_name = df.iloc[idx]['name']
        if game_name not in excluded_games:
            results.append((idx, score))
        if len(results) == top_n:
            break

    return results

# -----------------------------
# 8b. Collaborative Filtering Recommendations (item–item)
# -----------------------------
def recommend_games_collaborative(game_list, top_n=5):

    scores = np.zeros(len(cf_indices))
    liked_cf_games = []

    for game in game_list:

        # ✅ Direct match first (important fix)
        if game in cf_indices.index:
            liked_cf_games.append(game)
            idx = cf_indices[game]
            scores += item_cf_sim[idx]
            continue

        # Fallback to fuzzy matching
        match = find_closest_game(game)
        if match and match in cf_indices.index:
            liked_cf_games.append(match)
            idx = cf_indices[match]
            scores += item_cf_sim[idx]

    if not liked_cf_games:
        print("\nNo valid games found for collaborative filtering.")
        return []

    # Remove already liked games
    for game in liked_cf_games:
        scores[cf_indices[game]] = -1

    # Get top recommendations
    top_indices = scores.argsort()[-top_n:][::-1]

    recommendations = []
    for idx in top_indices:
        if scores[idx] < 0:
            continue
        recommendations.append((idx, scores[idx]))

    return recommendations

# -----------------------------
# 8c. Build Game Type Description Vector
# -----------------------------
def build_game_type_profile(game_type_description):
    """
    Convert a user's game type description into a TF-IDF vector
    for similarity matching with games in the dataset.
    """
    # Transform the description using the same TF-IDF vectorizer
    description_vector = tfidf.transform([game_type_description])
    return description_vector

# -----------------------------
# 8d. Recommend Games from Description
# -----------------------------
def recommend_games_from_description(game_type_description, top_n=5):
    """
    Recommend games based on a user's description of the type of game they want to play.
    """
    description_vector = build_game_type_profile(game_type_description)
    similarities = cosine_similarity(description_vector, tfidf_matrix)[0]

    recommendations = sorted(
        enumerate(similarities),
        key=lambda x: x[1],
        reverse=True
    )

    results = []
    for idx, score in recommendations:
        results.append((idx, score))
        if len(results) == top_n:
            break

    return results

# -----------------------------
# 9. Create Short Game Intro
# -----------------------------
def generate_intro(game_idx):
    row = df.iloc[game_idx]
    return (
        f"{row['name']} is a "
        f"{row['genres'].split(';')[0] if row['genres'] else 'unique'} game "
        f"developed by {row['developer'] or 'an independent studio'}. "
        f"It features {row['steamspy_tags'].split(';')[0] if row['steamspy_tags'] else 'engaging gameplay'} "
        f"and supports {row['categories'].split(';')[0] if row['categories'] else 'multiple play styles'}."
    )

# -----------------------------
# 12. Evaluation on test set (Precision, Recall, F1)
# -----------------------------


# ------------------------------------
# 12. Proper Evaluation on Test Set
# ------------------------------------
import numpy as np

def evaluate_on_test(interactions_train, interactions_test, top_n=5):

    print("\n======================================")
    print(f"📊 EVALUATION FOR K = {top_n}")
    print("======================================")

    test_users = interactions_test['user_id'].unique()

    precisions = []
    recalls = []

    for user in test_users:

        # Games user interacted with in TRAIN (used as input)
        train_games = interactions_train[
            interactions_train['user_id'] == user
        ]['name'].tolist()

        # Games user actually interacted with in TEST (ground truth)
        test_games = interactions_test[
            interactions_test['user_id'] == user
        ]['name'].tolist()

        # Skip users with no train/test data
        if len(train_games) == 0 or len(test_games) == 0:
            continue

        # Get CF recommendations
        recs = recommend_games_collaborative(train_games, top_n=top_n)

        if not recs:
            continue

        rec_games = [df.iloc[idx]['name'] for idx, _ in recs]

        rec_set = set(rec_games)
        test_set = set(test_games)

        # Number of correct recommendations
        hits = len(rec_set.intersection(test_set))

        precision = hits / top_n
        recall = hits / len(test_set)

        precisions.append(precision)
        recalls.append(recall)

    if len(precisions) == 0:
        print("No users could be evaluated.")
        return

    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)

    if avg_precision + avg_recall > 0:
        f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
    else:
        f1 = 0

    print(f"Precision@{top_n}: {avg_precision:.4f}")
    print(f"Recall@{top_n}:    {avg_recall:.4f}")
    print(f"F1 Score:          {f1:.4f}")

# -----------------------------
# 10. CLI Interface
# -----------------------------
def main():
    print("\n🎮 Advanced Steam Game Recommendation System")
    print("=" * 55)
    print("\nChoose recommendation method:")
    print("1. Based on games you've played (CONTENT-BASED)")
    print("2. Based on game type description (CONTENT-BASED)")
    print("3. Based on games you've played (COLLABORATIVE)")
    print()

    choice = input("Enter your choice (1, 2 or 3): ").strip()

    if choice == "1":
        # Original functionality: recommend based on played games
        print("\n" + "=" * 55)
        print("Enter games you have played (comma-separated)")
        print("Example: Counter Strike, Dota 2, Portal\n")

        user_input = input("Your games: ")
        game_list = [g.strip() for g in user_input.split(",")]

        user_profile, matched_games = build_user_profile(game_list)

        if user_profile is None:
            print("\nNo valid games found. Try again.")
            return

        print("\nBased on your interest in:")
        for g in matched_games:
            print(f"- {g}")

        recommendations = recommend_games_from_profile(
            user_profile,
            excluded_games=matched_games,
            top_n=5
        )

        print("\n🎯 Personalized Game Recommendations")
        print("=" * 55)
        for i, (idx, score) in enumerate(recommendations, 1):
            game = df.iloc[idx]
            print(f"\n🎮 Recommendation #{i}: {game['name']}")
            print("-" * 55)
            print(f"Genre      : {game['genres'] or 'N/A'}")
            print(f"Developer  : {game['developer'] or 'N/A'}")
            print(f"Tags       : {game['steamspy_tags'] or 'N/A'}")
            print(f"Similarity : {score:.4f}")
            print("\nWhy you may like it:")
            print(f"• Similar gameplay elements to your previous games")
            print(f"• Matches your preferred genres and play style")
            print()

    elif choice == "2":
        # New functionality: recommend based on game type description
        print("\n" + "=" * 55)
        print("Describe the type of game you want to play")
        print("Examples:")
        print("  - 'multiplayer action game with RPG elements'")
        print("  - 'puzzle game with strategy elements'")
        print("  - 'horror survival game with crafting'")
        print("  - 'racing game with multiplayer'")
        print()

        game_type_description = input("Your game type description: ").strip()

        if not game_type_description:
            print("\nPlease provide a description. Try again.")
            return

        recommendations = recommend_games_from_description(
            game_type_description,
            top_n=5
        )

        print("\n🎯 Game Recommendations Based on Your Description")
        print("=" * 55)
        print(f"Searching for: '{game_type_description}'\n")

        for i, (idx, score) in enumerate(recommendations, 1):
            game = df.iloc[idx]
            print(f"\n🎮 Recommendation #{i}: {game['name']}")
            print("-" * 55)
            print(f"Genre      : {game['genres'] or 'N/A'}")
            print(f"Developer  : {game['developer'] or 'N/A'}")
            print(f"Tags       : {game['steamspy_tags'] or 'N/A'}")
            print(f"Categories : {game['categories'] or 'N/A'}")
            print("\nWhy this matches:")
            print(f"• Matches your description: '{game_type_description}'")
            print(f"• Similarity score: {score:.4f}")
            print()

    elif choice == "3":
        # Collaborative filtering: recommend based on played games and other users' behaviour
        if item_cf_sim is None or cf_indices is None:
            print("\nCollaborative filtering data is not available.")
            print("Make sure the 'tamber/steam-video-games' dataset was downloaded correctly.")
            return

        print("\n" + "=" * 55)
        print("Enter games you have played (comma-separated)")
        print("Example: Counter Strike, Dota 2, Portal\n")

        user_input = input("Your games: ")
        game_list = [g.strip() for g in user_input.split(",") if g.strip()]

        if not game_list:
            print("\nNo games entered. Try again.")
            return

        recommendations = recommend_games_collaborative(
            game_list,
            top_n=5
        )

        if not recommendations:
            print("\nCould not generate collaborative recommendations.")
            return

        print("\n🎯 Collaborative Game Recommendations")
        print("=" * 55)
        for i, (idx, score) in enumerate(recommendations, 1):
            game = df.iloc[idx]
            print(f"\n🎮 Recommendation #{i}: {game['name']}")
            print("-" * 55)
            print(f"Genre      : {game['genres'] or 'N/A'}")
            print(f"Developer  : {game['developer'] or 'N/A'}")
            print(f"Tags       : {game['steamspy_tags'] or 'N/A'}")
            print("\nWhy you may like it:")
            print("• Players with similar tastes also enjoyed this game")
            print(f"• Collaborative similarity score: {score:.4f}")
            print()
        print("\nEvaluating collaborative filtering model...")
        for k in [3, 5, 10]:
            evaluate_on_test(interactions_train, interactions_test, top_n=k)

    else:
        print("\nInvalid choice. Please run the program again and select 1, 2 or 3.")

# -----------------------------
# 11. Run Program
# -----------------------------


# Call evaluation after running main()
if __name__ == "__main__":
    main()

