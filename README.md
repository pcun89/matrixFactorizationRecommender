# Matrix Factorization Recommender (SGD)

**Project:** Collaborative filtering using matrix factorization (user/item embeddings) trained with SGD for explicit ratings.

**Language:** Python 3.8+  
**Libraries:** numpy, pandas, joblib, scikit-learn (for RMSE) — minimal dependencies.

## Files
mf-recommender/
├── mf_recommender.py
├── requirements.txt
└── README.md

markdown
Copy code

## What it does
- Generates a small synthetic user-item ratings dataset (or loads your CSV)
- Trains a matrix-factorization model with L2 regularized SGD
- Evaluates with RMSE
- Exposes functions to predict a single rating and to recommend top-N items for a user
- Saves trained factors to `mf_model.joblib`

## Quick start
1. Create virtualenv (recommended) and install deps:
```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows
pip install -r requirements.txt
Run:

bash
Copy code
python mf_recommender.py
This will:

Train on synthetic data (1000 users × 500 items sparsity)

Print RMSE on held-out test set

Show top-5 recommendations for a sample user

Save mf_model.joblib (contains user_factors, item_factors, user_map, item_map)

To use your own dataset
Provide a CSV with columns: user_id,item_id,rating and run:

bash
Copy code
python mf_recommender.py --csv_path path/to/ratings.csv --user_col user_id --item_col item_id --rating_col rating
Next steps / extensions
Use ALS instead of SGD for faster convergence on large sparse matrices

Add implicit feedback (weighted-ALS)

Add bias terms (user/item biases) and learning-rate schedule

Use LightFM or implicit libraries for productionxFactorizationRecommender