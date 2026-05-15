"""
Main experiment runner for 2M-AHA Software Defect Prediction.
Run this script or import run_experiment() in a Jupyter notebook.
"""
import os, sys, time, warnings, json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import friedmanchisquare
warnings.filterwarnings('ignore')

# Try importing SMOTE
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    print("WARNING: imbalanced-learn not installed. Run: pip install imbalanced-learn")
    HAS_SMOTE = False

from data_loader import get_all_datasets
from aha_optimizer import AHA, TwoPhaseAHA
from baseline_optimizers import GWO, HHO, WOA, SSO, PSO


def get_classifiers():
    return {
        'RF': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'GB': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'AB': AdaBoostClassifier(n_estimators=50, random_state=42, algorithm='SAMME'),
        'KNN': KNeighborsClassifier(n_neighbors=5),
    }


def get_optimizers(n_pop=5, max_iter=50):
    return {
        '2M-AHA': TwoPhaseAHA(n_pop=n_pop, max_iter=max_iter),
        'AHA': AHA(n_pop=n_pop, max_iter=max_iter),
        'GWO': GWO(n_pop=n_pop, max_iter=max_iter),
        'HHO': HHO(n_pop=n_pop, max_iter=max_iter),
        'WOA': WOA(n_pop=n_pop, max_iter=max_iter),
        'SSO': SSO(n_pop=n_pop, max_iter=max_iter),
        'PSO': PSO(n_pop=n_pop, max_iter=max_iter),
    }


def apply_smote(X_train, y_train):
    if not HAS_SMOTE:
        return X_train, y_train
    if len(np.unique(y_train)) < 2:
        return X_train, y_train
    min_count = min(np.bincount(y_train))
    if min_count < 2:
        return X_train, y_train
    k = min(5, min_count - 1)
    try:
        sm = SMOTE(k_neighbors=k, random_state=42)
        return sm.fit_resample(X_train, y_train)
    except:
        return X_train, y_train


def evaluate_wfs(X, y, clf_name, clf, n_folds=10):
    """Evaluate Without Feature Selection baseline."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    aucs = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_train, y_train = apply_smote(X_train, y_train)
        from sklearn.base import clone
        c = clone(clf)
        c.fit(X_train, y_train)
        try:
            if hasattr(c, 'predict_proba'):
                proba = c.predict_proba(X_test)[:, 1]
            else:
                proba = c.decision_function(X_test)
            auc = roc_auc_score(y_test, proba)
        except:
            auc = 0.5
        aucs.append(auc)
    return np.mean(aucs)


def run_single_experiment(dataset_name, X, y, opt_name, optimizer, clf_name, clf, n_folds=10, seed=42):
    """Run a single experiment: optimizer + classifier on one dataset."""
    np.random.seed(seed)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    aucs = []
    n_selected_list = []
    from sklearn.base import clone

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_train_sm, y_train_sm = apply_smote(X_train, y_train)
        c = clone(clf)
        best_features, best_fitness = optimizer.optimize(
            X_train_sm, y_train_sm, X_test, y_test, c
        )
        selected = np.where(best_features == 1)[0]
        n_selected_list.append(len(selected))
        c2 = clone(clf)
        c2.fit(X_train_sm[:, selected], y_train_sm)
        try:
            if hasattr(c2, 'predict_proba'):
                proba = c2.predict_proba(X_test[:, selected])[:, 1]
            else:
                proba = c2.decision_function(X_test[:, selected])
            auc = roc_auc_score(y_test, proba)
        except:
            auc = 0.5
        aucs.append(auc)

    return {
        'dataset': dataset_name,
        'optimizer': opt_name,
        'classifier': clf_name,
        'mean_auc': np.mean(aucs),
        'std_auc': np.std(aucs),
        'mean_features': np.mean(n_selected_list),
        'total_features': X.shape[1]
    }


def run_full_experiment(n_runs=3, n_pop=5, max_iter=30, classifier_name='RF',
                        save_dir=None, max_datasets=None):
    """Run full experiment across all datasets, optimizers, and runs."""
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(__file__))

    datasets = get_all_datasets()
    if max_datasets:
        dataset_items = list(datasets.items())[:max_datasets]
    else:
        dataset_items = list(datasets.items())

    classifiers = get_classifiers()
    clf = classifiers[classifier_name]
    optimizers = get_optimizers(n_pop=n_pop, max_iter=max_iter)

    all_results = []
    total = len(dataset_items) * len(optimizers) * n_runs
    count = 0

    print(f"{'='*70}")
    print(f"Running experiments: {len(dataset_items)} datasets × {len(optimizers)} optimizers × {n_runs} runs")
    print(f"Classifier: {classifier_name} | Pop: {n_pop} | Iter: {max_iter}")
    print(f"{'='*70}")

    for ds_name, (X, y) in dataset_items:
        print(f"\n--- {ds_name} ({X.shape[0]} instances, {X.shape[1]} features, {y.mean()*100:.1f}% defective) ---")

        # WFS baseline
        wfs_auc = evaluate_wfs(X, y, classifier_name, clf)
        all_results.append({
            'dataset': ds_name, 'optimizer': 'WFS', 'classifier': classifier_name,
            'run': 0, 'mean_auc': wfs_auc, 'std_auc': 0,
            'mean_features': X.shape[1], 'total_features': X.shape[1]
        })
        print(f"  WFS: ROC_AUC = {wfs_auc:.4f}")

        for opt_name, optimizer in optimizers.items():
            run_aucs = []
            for run in range(n_runs):
                count += 1
                seed = 42 + run
                result = run_single_experiment(ds_name, X, y, opt_name, optimizer,
                                               classifier_name, clf, seed=seed)
                result['run'] = run
                all_results.append(result)
                run_aucs.append(result['mean_auc'])
                print(f"  {opt_name:8s} Run {run+1}/{n_runs}: AUC={result['mean_auc']:.4f} "
                      f"Features={result['mean_features']:.1f}/{result['total_features']} "
                      f"[{count}/{total}]")
            print(f"  {opt_name:8s} Mean: {np.mean(run_aucs):.4f} ± {np.std(run_aucs):.4f}")

    # Save results
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(save_dir, f'results_{classifier_name}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    return df


def analyze_results(results_df):
    """Analyze and print summary statistics."""
    # Aggregate by dataset and optimizer
    summary = results_df.groupby(['dataset', 'optimizer']).agg(
        mean_auc=('mean_auc', 'mean'),
        std_auc=('mean_auc', 'std'),
        mean_features=('mean_features', 'mean')
    ).reset_index()

    # Pivot table
    pivot = summary.pivot(index='dataset', columns='optimizer', values='mean_auc')
    print("\n" + "="*70)
    print("ROC_AUC Results Summary")
    print("="*70)
    print(pivot.round(4).to_string())

    # Mean across datasets
    print("\n--- Mean ROC_AUC across all datasets ---")
    means = pivot.mean().sort_values(ascending=False)
    for opt, val in means.items():
        print(f"  {opt:10s}: {val:.4f}")

    # Friedman test
    opt_names = [c for c in pivot.columns if c != 'WFS']
    if len(opt_names) >= 3 and len(pivot) >= 3:
        data_for_friedman = [pivot[opt].dropna().values for opt in opt_names]
        min_len = min(len(d) for d in data_for_friedman)
        data_for_friedman = [d[:min_len] for d in data_for_friedman]
        try:
            stat, p_value = friedmanchisquare(*data_for_friedman)
            print(f"\n--- Friedman Test ---")
            print(f"  Statistic: {stat:.4f}, p-value: {p_value:.6f}")
            if p_value < 0.05:
                print("  Result: Significant difference (p < 0.05)")
            else:
                print("  Result: No significant difference (p >= 0.05)")
        except Exception as e:
            print(f"  Friedman test error: {e}")

    # Rank analysis
    print("\n--- Mean Rank (lower is better) ---")
    ranks = pivot.rank(axis=1, ascending=False)
    mean_ranks = ranks.mean().sort_values()
    for opt, rank in mean_ranks.items():
        print(f"  {opt:10s}: {rank:.2f}")

    return summary, pivot


if __name__ == "__main__":
    # Quick test run
    df = run_full_experiment(n_runs=2, n_pop=3, max_iter=20,
                             classifier_name='RF', max_datasets=3)
    analyze_results(df)
