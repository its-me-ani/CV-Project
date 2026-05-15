"""
Data loading utilities for NASA MDP, PROMISE, AEEEM, and JIRA datasets.
"""
import os, glob, warnings
import numpy as np
import pandas as pd
from scipy.io import arff

warnings.filterwarnings('ignore')
BASE = os.path.dirname(os.path.abspath(__file__))


def load_arff(filepath):
    """Load ARFF file and return X, y as numpy arrays."""
    try:
        data, meta = arff.loadarff(filepath)
        df = pd.DataFrame(data)
    except:
        # Fallback: manual parsing
        df = _parse_arff_manual(filepath)
    # Decode bytes if needed
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: x.decode() if isinstance(x, bytes) else x)
    # Last column is label
    label_col = df.columns[-1]
    y_raw = df[label_col].values
    X = df.iloc[:, :-1].values.astype(float)
    # Convert labels to binary (0/1)
    unique_labels = np.unique(y_raw)
    # Check if labels are numeric (e.g., bug count)
    try:
        y_numeric = np.array([float(v) for v in y_raw])
        is_numeric = True
    except (ValueError, TypeError):
        is_numeric = False

    if is_numeric and len(unique_labels) > 2:
        # Numeric bug count: > 0 means defective
        y = (y_numeric > 0).astype(int)
    elif len(unique_labels) == 2:
        # Map defective/buggy to 1
        pos_labels = {'Y', 'y', 'yes', 'Yes', 'YES', 'true', 'True', 'TRUE',
                      'buggy', 'Buggy', 'BUGGY', 'defective', 'Defective', '1', 1}
        y = np.array([1 if str(v).strip() in pos_labels else 0 for v in y_raw])
    else:
        y = np.array([1 if str(v).strip() not in {'N', 'n', 'no', 'No', 'clean', 'Clean', 'false', 'False', '0', 0} else 0 for v in y_raw])
    # Remove NaN rows
    mask = ~np.isnan(X).any(axis=1)
    return X[mask], y[mask]


def _parse_arff_manual(filepath):
    """Manual ARFF parser for files scipy can't handle."""
    attrs = []
    data_started = False
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            if line.lower().startswith('@attribute'):
                parts = line.split()
                attrs.append(parts[1])
            elif line.lower().startswith('@data'):
                data_started = True
            elif data_started:
                vals = line.split(',')
                rows.append(vals)
    df = pd.DataFrame(rows, columns=attrs)
    # Convert numeric columns
    for col in df.columns[:-1]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def load_csv_promise(filepath):
    """Load PROMISE CSV dataset."""
    df = pd.read_csv(filepath)
    # Drop non-numeric identifier columns
    drop_cols = [c for c in df.columns if c.lower() in ('name', 'version', 'name.1')]
    df = df.drop(columns=drop_cols, errors='ignore')
    label_col = df.columns[-1]  # 'bug' column
    y_raw = df[label_col].values
    X = df.iloc[:, :-1].values.astype(float)
    # bug > 0 means defective
    y = np.array([1 if float(v) > 0 else 0 for v in y_raw])
    mask = ~np.isnan(X).any(axis=1)
    return X[mask], y[mask]


def get_all_datasets(base_dir=None):
    """Return dict of {name: (X, y)} for all datasets."""
    if base_dir is None:
        base_dir = BASE
    datasets = {}

    # 1. NASA MDP (D' version)
    nasa_dir = os.path.join(base_dir, "MDP", "D'")
    if os.path.isdir(nasa_dir):
        for f in sorted(glob.glob(os.path.join(nasa_dir, "*.arff"))):
            name = os.path.splitext(os.path.basename(f))[0]
            if os.path.getsize(f) < 100:
                continue  # Skip empty files like KC4
            try:
                X, y = load_arff(f)
                if len(X) > 10:
                    datasets[f"NASA_{name}"] = (X, y)
            except Exception as e:
                print(f"  Skipping {name}: {e}")

    # 2. AEEEM
    aeeem_dir = os.path.join(base_dir, "AEEEM")
    if os.path.isdir(aeeem_dir):
        for f in sorted(glob.glob(os.path.join(aeeem_dir, "*.arff"))):
            name = os.path.splitext(os.path.basename(f))[0]
            try:
                X, y = load_arff(f)
                if len(X) > 10:
                    datasets[f"AEEEM_{name}"] = (X, y)
            except Exception as e:
                print(f"  Skipping {name}: {e}")

    # 3. PROMISE (OO metrics CSV)
    oo_dir = os.path.join(base_dir, "OO metrics Data")
    if os.path.isdir(oo_dir):
        target_projects = {
            'ant': 'ant-1.7.csv', 'camel': 'camel-1.6.csv',
            'jedit': 'jedit-4.3.csv', 'lucene': 'lucene-2.4.csv',
            'poi': 'poi-3.0.csv', 'xalan': 'xalan-2.6.csv'
        }
        for proj, fname in target_projects.items():
            fpath = os.path.join(oo_dir, proj, fname)
            if os.path.exists(fpath):
                try:
                    X, y = load_csv_promise(fpath)
                    if len(X) > 10:
                        datasets[f"PROMISE_{fname.replace('.csv', '')}"] = (X, y)
                except Exception as e:
                    print(f"  Skipping {fname}: {e}")

    # 4. JIRA
    jira_dir = os.path.join(base_dir, "JIRA")
    if os.path.isdir(jira_dir):
        for f in sorted(glob.glob(os.path.join(jira_dir, "*.arff"))):
            name = os.path.splitext(os.path.basename(f))[0]
            try:
                X, y = load_arff(f)
                if len(X) > 10:
                    datasets[f"JIRA_{name}"] = (X, y)
            except Exception as e:
                print(f"  Skipping {name}: {e}")

    return datasets


if __name__ == "__main__":
    datasets = get_all_datasets()
    print(f"\nLoaded {len(datasets)} datasets:")
    for name, (X, y) in sorted(datasets.items()):
        print(f"  {name:35s} | Instances: {X.shape[0]:6d} | Features: {X.shape[1]:3d} | Defective: {y.mean()*100:.1f}%")
