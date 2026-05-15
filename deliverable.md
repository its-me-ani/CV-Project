# 2M-AHA: Software Defect Prediction — Deliverables Summary

## ✅ All Files Created

| File | Description |
|------|------------|
| [2M_AHA_Report.tex](file:///Users/anirudhsharma/Downloads/fwddatasets/2M_AHA_Report.tex) | IEEE two-column LaTeX paper with full methodology, equations, algorithm pseudocode, and placeholders for result tables |
| [2M_AHA_Experiment.ipynb](file:///Users/anirudhsharma/Downloads/fwddatasets/2M_AHA_Experiment.ipynb) | Jupyter notebook with complete experiment pipeline, visualizations, statistical analysis, and LaTeX table generation |
| [aha_optimizer.py](file:///Users/anirudhsharma/Downloads/fwddatasets/aha_optimizer.py) | Core AHA + **2M-AHA** (Two-Phase Mutation) optimizer classes |
| [baseline_optimizers.py](file:///Users/anirudhsharma/Downloads/fwddatasets/baseline_optimizers.py) | All 5 baseline optimizers: GWO, HHO, WOA, SSO, PSO |
| [data_loader.py](file:///Users/anirudhsharma/Downloads/fwddatasets/data_loader.py) | Unified loader for NASA MDP, PROMISE, AEEEM, JIRA (30 datasets) |
| [experiment_runner.py](file:///Users/anirudhsharma/Downloads/fwddatasets/experiment_runner.py) | Main experiment orchestrator with SMOTE, 10-fold CV, Friedman test |
| [research_plan.md](file:///Users/anirudhsharma/.gemini/antigravity/brain/14d8abca-267e-44d8-812f-e579fc297b24/artifacts/research_plan.md) | Detailed research plan document |

## 🔬 Methodology (2M-AHA)

```
Dataset → 10-Fold CV → SMOTE (train only) → 2M-AHA Feature Selection → Classifier → ROC_AUC
```

### Artificial Hummingbird Algorithm (AHA) — Three Foraging Strategies:
1. **Guided Foraging:** Move toward known food source (exploitation)
2. **Territorial Foraging:** Local search within territory (local exploration)
3. **Migration Foraging:** Random restart for worst agent (global exploration)

### Two-Phase Mutation (Novel Enhancement):
- **Phase 1 — Guided Mutation:** Polynomial mutation on top-K solutions (exploitation boost)
- **Phase 2 — Random Mutation:** Bit-flip on stagnant solutions after τ=10 iterations (exploration boost)

## 📊 Datasets — 30 Loaded Successfully

| Repository | Projects | Features | Notes |
|-----------|---------|---------|-------|
| NASA MDP (D') | 12 | 21-39 | Static code metrics |
| PROMISE (OO) | 6 | 20 | CK/OO class-level metrics |
| AEEEM | 5 | 61 | Change metrics + entropy |
| JIRA | 7 | 65 | Bug count based labels |

## 🏃 How to Run

### Quick Validation (~20 min)
```python
from experiment_runner import run_full_experiment, analyze_results
df = run_full_experiment(n_runs=2, n_pop=3, max_iter=20, classifier_name='RF', max_datasets=5)
analyze_results(df)
```

### Full Experiment (several hours)
```python
df = run_full_experiment(n_runs=20, n_pop=5, max_iter=100, classifier_name='RF')
analyze_results(df)
```

### Publication-Quality Settings (matching reference paper)
- `n_runs=20`, `n_pop=5`, `max_iter=100` → expect 6-12 hours for all 30 datasets

## ✅ Validation Run Results

Quick test (2 datasets, 10 iterations, 3 agents) completed successfully:
- All 7 optimizers + WFS baseline ran without errors
- Feature selection properly reduces dimensions (~50%)
- ROC_AUC computed correctly with SMOTE + 10-fold CV
- Results saved to CSV automatically

> [!TIP]
> With only 10 iterations, the optimizers don't fully converge. Use **max_iter=50-100** for meaningful results. The reference paper uses 100 iterations × 20 runs.

## 📈 Benchmark Comparison Structure

| Algorithm | Type | Role |
|-----------|------|------|
| **2M-AHA** | Proposed | Novel contribution |
| AHA | Ablation | Shows mutation impact |
| GWO | Baseline | Reference paper's method |
| HHO | Baseline | Harris Hawks |
| WOA | Baseline | Whale Optimization |
| SSO | Baseline | Salp Swarm |
| PSO | Baseline | Particle Swarm |
| WFS | Baseline | No feature selection |

## 📝 LaTeX Report Sections
1. Introduction with contributions
2. Related work (cites all 4 reference papers)
3. Methodology (AHA equations, 2M mutation, fitness function, SMOTE, classifiers)
4. Algorithm pseudocode
5. Experimental results (tables placeholder — fill from notebook)
6. Statistical analysis (Friedman test)
7. Conclusion and future work
8. 19 references
