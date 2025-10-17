
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix
import re, warnings
warnings.filterwarnings("ignore")

# SMAPE METRIC
def smape(y_true, y_pred):
    epsilon = 1e-6
    y_pred = np.maximum(y_pred, epsilon)
    y_true = np.maximum(y_true, epsilon)
    return np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2)) * 100

def lgb_smape_metric(y_true, y_pred):
    true_price = np.expm1(y_true)
    predicted_price = np.expm1(y_pred)
    score = smape(true_price, predicted_price)
    return 'SMAPE', score, False

# LOAD DATA
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

# print("🔹 Loading data...")
df_train = pd.read_csv(TRAIN_FILE)
df_test = pd.read_csv(TEST_FILE)

len_train = len(df_train)
len_test = len(df_test)

df_all = pd.concat([df_train.drop(columns=["price"]), df_test], ignore_index=True)
y_train_log = np.log1p(df_train["price"])

# FEATURE ENGINEERING
# print("🔹 Generating TF-IDF features...")

tfidf_vec = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    max_features=67000,
    norm='l2',
    min_df=3,
    max_df=0.9
)

X_all_tfidf = tfidf_vec.fit_transform(df_all["catalog_content"].fillna(""))

# print("🔹 Extracting IPQ features...")

def extract_ipq(text):
    if isinstance(text, str):
        match = re.search(r'(\d+)\s*(?:pack|qty|count|pc|x|ipq)', text.lower())
        if match:
            return float(match.group(1))
    return 1.0

df_all["extracted_ipq"] = df_all["catalog_content"].apply(extract_ipq)

ipq = df_all["extracted_ipq"].values.reshape(-1, 1)
X_all = hstack([X_all_tfidf, csr_matrix(ipq)])

X_train = X_all[:len_train]
X_test = X_all[len_train:]

# TRAIN/VALIDATION SPLIT
X_train_fit, X_val, y_train_fit, y_val = train_test_split(
    X_train, y_train_log, test_size=0.25, random_state=46
)

# MODEL TRAINING (OPTIMIZED)
# print(f"🔹 Training LightGBM on {X_train.shape[1]} features...")

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.2,         
    'num_leaves': 63,
    'max_depth': -1,
    'min_child_samples': 50,
    'subsample': 0.9,
    'subsample_freq': 1,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.3,
    'random_state': 42,
    'n_jobs': -1,
    'force_col_wise': True,
    'device': 'gpu',               
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'max_bin': 255                 
}

model = lgb.LGBMRegressor(**lgb_params)

model.fit(
    X_train_fit, y_train_fit,
    eval_set=[(X_val, y_val)],
    eval_metric=lgb_smape_metric,
    callbacks=[
        lgb.early_stopping(stopping_rounds=5, verbose=50)
    ]
)


# print(f"\n✅ Best iteration: {model.best_iteration_}")

# PREDICTION
# print("🔹 Generating predictions...")
log_preds_test = model.predict(X_test, num_iteration=model.best_iteration_)
preds_test = np.maximum(np.expm1(log_preds_test), 1e-4)

# OUTPUT
submission = pd.DataFrame({
    'sample_id': df_test['sample_id'],
    'price': preds_test.astype(float)
})
submission.to_csv("test_out.csv", index=False)

# VALIDATION SMAPE
val_preds_log = model.predict(X_val, num_iteration=model.best_iteration_)
val_preds = np.expm1(val_preds_log)
val_true = np.expm1(y_val)
val_smape = smape(val_true, val_preds)

print(f"\nSMAPE: {val_smape:.4f}%")
# print("✅ Predictions saved to test_out.csv")
