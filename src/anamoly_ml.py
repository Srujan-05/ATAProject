import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
from sklearn.model_selection import train_test_split
import random

try:
    import lightgbm as lgb
    lgb.register_logger(None)
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False

EPS = 1e-8


def make_silver_labels(df):
    df = df.copy()

    z = np.abs(df['z_resid'].fillna(0.0))

    in_pi = (df['y_true'] >= df['lo']) & (df['y_true'] <= df['hi'])

    cond_pos = (z >= 3.5) | ((~in_pi) & (z >= 2.5))
    cond_neg = (z < 1.0) & (in_pi)

    df['silver_label'] = np.where(cond_pos, 1,
                           np.where(cond_neg, 0, np.nan))
    return df


def sample_for_human_verification(df, n_samples=100, approx_pos=50, seed=42):
    random.seed(seed)

    pos = df[df['silver_label'] == 1]
    neg = df[df['silver_label'] == 0]

    n_pos = min(len(pos), approx_pos)
    n_neg = n_samples - n_pos
    n_neg = min(len(neg), n_neg)

    sampled_pos = pos.sample(n=n_pos, random_state=seed) if n_pos > 0 else pos.iloc[0:0]
    sampled_neg = neg.sample(n=n_neg, random_state=seed) if n_neg > 0 else neg.iloc[0:0]

    sampled = pd.concat([sampled_pos, sampled_neg]) \
                .sample(frac=1, random_state=seed) \
                .reset_index(drop=True)

    return sampled


def build_features(df, max_lag=48):
    df = df.copy().reset_index(drop=True)
    N = len(df)

    feat = pd.DataFrame(index=df.index)

    # lags
    for lag in range(1, max_lag + 1):
        feat[f'resid_lag_{lag}'] = df['resid'].shift(lag)
        feat[f'yhat_lag_{lag}'] = df['yhat'].shift(lag)

    # rolling windows
    for w in [6, 24, 48]:
        feat[f'resid_roll_mean_{w}'] = df['resid'].rolling(window=w, min_periods=1).mean().shift(1)
        feat[f'resid_roll_std_{w}'] = df['resid'].rolling(window=w, min_periods=1).std(ddof=0).shift(1)

    ts = pd.to_datetime(df['timestamp'], errors='coerce')
    feat['hour'] = ts.dt.hour
    feat['dow'] = ts.dt.dayofweek

    feat['lo'] = df['lo']
    feat['hi'] = df['hi']
    feat['horizon'] = df['horizon'] if 'horizon' in df else 24

    feat['resid'] = df['resid']
    feat['yhat'] = df['yhat']

    return feat


def train_anomaly_classifier(features_df, labels, model_type='lgb', random_state=42):

    mask = ~np.isnan(labels)
    X = features_df.loc[mask].copy()
    y = labels[mask].astype(int)

    X = X.dropna(axis=1, how='all')
    X = X.fillna(0.0)

    if _HAS_LGB and model_type == 'lgb':
        model = lgb.LGBMClassifier(n_estimators=200, random_state=random_state)
        model.fit(X, y)

        def predict_proba(Xp):
            Xp = Xp.reindex(columns=X.columns).fillna(0.0)
            return model.predict_proba(Xp)[:, 1]

        return model, predict_proba, X, y

    else:
        clf = LogisticRegression(max_iter=1000, random_state=random_state)
        clf.fit(X, y)

        def predict_proba(Xp):
            Xp = Xp.reindex(columns=X.columns).fillna(0.0)
            return clf.predict_proba(Xp)[:, 1]

        return clf, predict_proba, X, y


def evaluate_classifier(proba, y_true, fixed_precision=0.80):
    ap = average_precision_score(y_true, proba)

    precision, recall, thresholds = precision_recall_curve(y_true, proba)

    thr = None
    achieved_precision = None
    achieved_recall = None
    achieved_f1 = None

    for p, r, t in zip(precision[:-1], recall[:-1], np.append(thresholds, 1.0)):
        if p >= fixed_precision:
            thr = t
            achieved_precision = p
            achieved_recall = r
            achieved_f1 = 2 * p * r / (p + r + EPS)
            break

    if thr is None:
        best_idx = np.argmax(precision[:-1])
        thr = thresholds[best_idx] if len(thresholds) > 0 else 0.5
        achieved_precision = precision[best_idx]
        achieved_recall = recall[best_idx]
        achieved_f1 = 2 * achieved_precision * achieved_recall / (achieved_precision + achieved_recall + EPS)

    f1_default = f1_score(y_true, (proba >= 0.5).astype(int))

    return {
        'pr_auc': float(ap),
        'threshold_for_precision_{}'.format(fixed_precision): float(thr),
        'precision_at_thr': float(achieved_precision),
        'recall_at_thr': float(achieved_recall),
        'f1_at_thr': float(achieved_f1),
        'f1_at_0_5': float(f1_default)
    }


def run_anomaly_ml_pipeline(anomalies_csv_path, country_code, output_dir="../outputs",
                            sample_n=100, approx_pos=50):

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(anomalies_csv_path)

    for col in ['y_true', 'yhat', 'lo', 'hi']:
        if col in df:
            df[col] = df[col].astype(float)

    if 'resid' not in df:
        df['resid'] = df['y_true'] - df['yhat']

    # --------------------------
    # SILVER LABELS
    # --------------------------
    df_silver = make_silver_labels(df)
    silver_path = os.path.join(output_dir, f"{country_code}_anomaly_silver_labels.csv")
    df_silver.to_csv(silver_path, index=False)
    print("Saved silver labels:", silver_path)

    # --------------------------
    # HUMAN REVIEW SAMPLE
    # --------------------------
    sampled = sample_for_human_verification(df_silver, n_samples=sample_n, approx_pos=approx_pos)
    sample_path = os.path.join(output_dir, f"{country_code}_anomaly_labels_for_human_review.csv")
    sampled.to_csv(sample_path, index=False)
    print("Saved sample for human review:", sample_path)

    # --------------------------
    # HUMAN VERIFIED LABELS
    # --------------------------
    verified_path = os.path.join(output_dir, f"{country_code}_anomaly_labels_verified.csv")
    if not os.path.exists(verified_path):
        print("Please verify samples and save:", verified_path)
        return

    verified = pd.read_csv(verified_path)

    df = df.merge(verified[['timestamp', 'verified_label']], on='timestamp', how='left')

    # --------------------------
    # FEATURE BUILDING
    # --------------------------
    feats = build_features(df, max_lag=48)
    mask = ~df['verified_label'].isna()

    X = feats.loc[mask].fillna(0.0)
    y = df.loc[mask, 'verified_label'].astype(int).values

    # --------------------------
    # TRAIN/VAL SPLIT
    # --------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    model_type = "lgb" if _HAS_LGB else "logistic"

    model, proba_func, X_used, y_used = train_anomaly_classifier(
        pd.concat([X_train, X_val]),
        np.concatenate([y_train, y_val]),
        model_type=model_type
    )

    # --------------------------
    # EVALUATE
    # --------------------------
    proba_val = proba_func(X_val)
    eval_results = evaluate_classifier(proba_val, y_val, fixed_precision=0.80)

    eval_path = os.path.join(output_dir, f"{country_code}_anomaly_ml_eval.json")
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print("Saved ML eval:", eval_path)

    # --------------------------
    # GENERATE PREDICTIONS FOR ALL
    # --------------------------
    proba_all = proba_func(feats.fillna(0.0))

    df_out = df.copy()
    df_out['anomaly_proba'] = proba_all
    df_out['anomaly_pred_at_0_5'] = (df_out['anomaly_proba'] >= 0.5).astype(int)

    out_path = os.path.join(output_dir, f"{country_code}_anomaly_ml_predictions_all.csv")
    df_out.to_csv(out_path, index=False)
    print("Saved all predictions:", out_path)

    return eval_results


if __name__ == "__main__":
    cc = ["CH", "FR", "AT"]
    for c in cc:
        run_anomaly_ml_pipeline(f"../outputs/{c}_anomalies.csv", country_code=c)
