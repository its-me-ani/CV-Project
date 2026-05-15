# Research Plan: 2M-AHA — Two-Phase Mutation Artificial Hummingbird Algorithm for Software Defect Prediction

## 1. Title & Novelty Claim

**Proposed Title:**  
*"2M-AHA: A Two-Phase Mutation Artificial Hummingbird Algorithm for Wrapper-Based Feature Selection in Software Defect Prediction"*

**Core Novelty:**  
A wrapper-based feature selection method using the Artificial Hummingbird Algorithm (AHA) enhanced with a two-phase mutation mechanism (2M-AHA) for software defect prediction, demonstrating superior performance over GWO, HHO, WOA, SSO, and PSO across 27 publicly available datasets from four repositories.

> [!IMPORTANT]
> The Artificial Hummingbird Algorithm (AHA) was chosen over Social Spider Optimization (SSO) because:
> 1. AHA is more recent (2022) and has proven superior convergence in continuous/discrete optimization
> 2. AHA's three foraging strategies (guided, territorial, migration) naturally map to exploration-exploitation balance
> 3. AHA has **not** been applied to SDP feature selection before — stronger novelty claim
> 4. SSO (Salp Swarm) was already tested in the 2M-GWO paper as a baseline and was outperformed

---

## 2. Research Questions

| RQ# | Question |
|-----|----------|
| RQ1 | Does 2M-AHA significantly improve SDP model performance compared to standard AHA and without-feature-selection (WFS) models? |
| RQ2 | How does 2M-AHA compare against GWO, 2M-GWO, HHO, WOA, SSO, and PSO for feature selection in SDP? |
| RQ3 | Which classifier performs best in conjunction with 2M-AHA for SDP? |
| RQ4 | Does the two-phase mutation mechanism meaningfully enhance AHA's search capability? |

---

## 3. Datasets (27 Projects, 4 Repositories)

### 3.1 NASA MDP (13 projects — D' cleaned version)
| Dataset | #Instances | #Features | %Defective |
|---------|-----------|-----------|------------|
| CM1 | 327 | 37 | ~12.8% |
| JM1 | 7,720 | 21 | ~19.3% |
| KC1 | 1,162 | 21 | ~15.5% |
| KC3 | 194 | 39 | ~18.6% |
| MC1 | 1,952 | 38 | ~2.3% |
| MC2 | 125 | 39 | ~32.0% |
| MW1 | 250 | 37 | ~7.6% |
| PC1 | 679 | 37 | ~6.9% |
| PC2 | 722 | 36 | ~0.4% |
| PC3 | 1,053 | 37 | ~10.4% |
| PC4 | 1,270 | 37 | ~12.2% |
| PC5 | 1,694 | 38 | ~27.3% |
| KC4 | (excluded if empty) | — | — |

### 3.2 PROMISE Repository (via OO Metrics — select projects)
Projects: ant-1.7, camel-1.6, jedit-4.2, jedit-4.3, lucene-2.4, poi-3.0, xalan-2.6, log4j-1.2, velocity-1.6, synapse-1.2, xerces-1.4  
**Features:** CK/OO metrics (wmc, dit, noc, cbo, rfc, lcom, ca, ce, npm, loc, etc.)

### 3.3 AEEEM (5 projects)
| Dataset | #Instances | #Features |
|---------|-----------|-----------|
| EQ | 324 | 61 |
| JDT | 997 | 61 |
| Lucene | 691 | 61 |
| Mylyn | 1,862 | 61 |
| PDE | 1,492 | 61 |

### 3.4 JIRA (select projects)
Projects: activemq-5.0.0, derby-10.5.1.1, groovy-1.6, hbase-0.94.0, hive-0.9.0, jruby-1.1, wicket-1.3.0

---

## 4. Proposed Methodology

### 4.1 Pipeline Overview

```
Raw Dataset → 10-Fold CV Split → SMOTE (training fold) → 2M-AHA Feature Selection → Classifier Training → ROC_AUC Evaluation
```

### 4.2 Artificial Hummingbird Algorithm (AHA)

**Three foraging strategies:**

1. **Guided Foraging:** Move toward a known good food source
   - `V_i(t+1) = X_target(t) + a × D × (X_i(t) − X_target(t))`
   
2. **Territorial Foraging:** Local search within current territory
   - `V_i(t+1) = X_i(t) + b × D × X_i(t)`
   
3. **Migration Foraging:** Random restart for worst-performing hummingbird
   - `X_worst(t+1) = LB + r × (UB − LB)`

**Binary conversion for feature selection:**
- `BX_ij = 1 if sigmoid(X_ij) > rand(), else 0`

### 4.3 Two-Phase Mutation (Novel Enhancement)

Adapted from the 2M-GWO paper's mutation mechanism:

**Phase 1 — Guided Mutation (Exploitation):**
- After standard AHA position update
- Apply polynomial mutation to top-K solutions
- Enhances local search around promising regions

**Phase 2 — Random Mutation (Exploration):**
- Applied to solutions that haven't improved for τ iterations
- Bit-flip mutation on binary feature vector
- Prevents premature convergence and local optima trapping

### 4.4 Fitness Function

```
Fitness = μ₁ × (1 − ROC_AUC) + μ₂ × (|S| / |F|)
```
Where:
- μ₁ = 0.99 (classification weight)
- μ₂ = 0.01 (feature reduction weight)
- |S| = number of selected features
- |F| = total features

### 4.5 SMOTE for Class Imbalance
- Applied **only on training folds** (not test folds)
- Generates synthetic minority class samples

### 4.6 Classifiers
| Classifier | Key Hyperparameters |
|------------|-------------------|
| Random Forest (RF) | n_estimators=100, default params |
| Support Vector Machine (SVM) | RBF kernel, default C |
| Gradient Boosting (GB) | n_estimators=100, lr=0.1 |
| AdaBoost (AB) | n_estimators=50 |
| K-Nearest Neighbor (KNN) | k=5, Euclidean distance |

---

## 5. Baseline Algorithms for Comparison

| Algorithm | Reference |
|-----------|-----------|
| GWO (Grey Wolf Optimizer) | Mirjalili et al. (2014) |
| 2M-GWO (Two-Phase Mutation GWO) | Reference paper |
| HHO (Harris Hawks Optimization) | Heidari et al. (2019) |
| WOA (Whale Optimization Algorithm) | Mirjalili & Lewis (2016) |
| SSO (Salp Swarm Optimization) | Mirjalili et al. (2017) |
| PSO (Particle Swarm Optimization) | Kennedy & Eberhart (1995) |
| WFS (Without Feature Selection) | Full feature set baseline |

---

## 6. Evaluation Metrics & Statistical Tests

### 6.1 Primary Metric
- **ROC_AUC** (Area Under the ROC Curve) — same as reference paper

### 6.2 Statistical Analysis
1. **Friedman Test** — non-parametric test to compare ranked performance across datasets
2. **Mean Rank** — average rank across all 27 datasets
3. **Mean ROC_AUC** — average performance per algorithm

### 6.3 Experimental Setup
- **20 independent runs** per algorithm per dataset (random seeds)
- **100 iterations** maximum per optimization run
- **Population size:** 5 search agents
- **10-fold cross-validation** for each iteration
- **μ₁ = 0.99, μ₂ = 0.01** for fitness function

---

## 7. Expected Contributions

1. **Novel Algorithm:** First application of AHA with two-phase mutation for SDP
2. **Comprehensive Evaluation:** 27 datasets from 4 repositories (NASA, PROMISE, AEEEM, JIRA)
3. **Benchmark Comparison:** Against 6 metaheuristic baselines + WFS
4. **Statistical Validation:** Friedman test + mean ranking analysis
5. **Feature Reduction Analysis:** Demonstrating effective dimensionality reduction

---

## 8. Implementation Stack

| Component | Tool |
|-----------|------|
| Language | Python 3.10+ |
| ML Framework | scikit-learn |
| SMOTE | imbalanced-learn |
| ARFF Parsing | scipy.io.arff / liac-arff |
| Statistical Tests | scipy.stats |
| Visualization | matplotlib, seaborn |
| Report | LaTeX (IEEE format) |

---

## 9. File Deliverables

1. **LaTeX Report** — Full research paper in IEEE two-column format
2. **Jupyter Notebook** — Complete implementation with all experiments
3. **Results Tables** — ROC_AUC comparison tables per classifier per algorithm
4. **Statistical Test Results** — Friedman test p-values and rankings
