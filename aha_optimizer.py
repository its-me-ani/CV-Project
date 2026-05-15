"""
2M-AHA: Two-Phase Mutation Artificial Hummingbird Algorithm for Feature Selection in SDP
Core optimizer module
"""
import numpy as np
from scipy.special import expit  # sigmoid

class AHA:
    """Standard Artificial Hummingbird Algorithm for binary feature selection."""
    def __init__(self, n_pop=5, max_iter=100, mu1=0.99, mu2=0.01):
        self.n_pop = n_pop
        self.max_iter = max_iter
        self.mu1 = mu1
        self.mu2 = mu2

    def _init_population(self, n_features):
        return np.random.uniform(-1, 1, (self.n_pop, n_features))

    def _to_binary(self, X):
        return (expit(X) > np.random.random(X.shape)).astype(int)

    def _ensure_at_least_one(self, bx):
        if bx.sum() == 0:
            bx[np.random.randint(len(bx))] = 1
        return bx

    def _fitness(self, binary_vec, X_train, y_train, X_test, y_test, clf):
        selected = np.where(binary_vec == 1)[0]
        if len(selected) == 0:
            return 1.0
        clf.fit(X_train[:, selected], y_train)
        from sklearn.metrics import roc_auc_score
        try:
            if hasattr(clf, 'predict_proba'):
                proba = clf.predict_proba(X_test[:, selected])[:, 1]
            else:
                proba = clf.decision_function(X_test[:, selected])
            auc = roc_auc_score(y_test, proba)
        except:
            auc = 0.5
        n_selected = len(selected)
        n_total = len(binary_vec)
        return self.mu1 * (1 - auc) + self.mu2 * (n_selected / n_total)

    def _guided_foraging(self, Xi, X_target, n_features):
        a = np.random.randn()
        skill = np.random.randint(3)
        D = np.zeros(n_features)
        if skill == 0:  # axial
            D[np.random.randint(n_features)] = 1
        elif skill == 1:  # diagonal
            k = np.random.randint(1, n_features)
            idx = np.random.choice(n_features, k, replace=False)
            D[idx] = 1
        else:  # omnidirectional
            D[:] = 1
        return X_target + a * D * (Xi - X_target)

    def _territorial_foraging(self, Xi, n_features):
        b = np.random.randn()
        skill = np.random.randint(3)
        D = np.zeros(n_features)
        if skill == 0:
            D[np.random.randint(n_features)] = 1
        elif skill == 1:
            k = np.random.randint(1, n_features)
            idx = np.random.choice(n_features, k, replace=False)
            D[idx] = 1
        else:
            D[:] = 1
        return Xi + b * D * Xi

    def optimize(self, X_train, y_train, X_test, y_test, clf):
        n_features = X_train.shape[1]
        pop = self._init_population(n_features)
        nectar_rates = np.zeros(self.n_pop)
        fitness_vals = np.full(self.n_pop, np.inf)

        # Evaluate initial population
        for i in range(self.n_pop):
            bx = self._ensure_at_least_one(self._to_binary(pop[i]))
            fitness_vals[i] = self._fitness(bx, X_train, y_train, X_test, y_test, clf)
            nectar_rates[i] = fitness_vals[i]

        best_idx = np.argmin(fitness_vals)
        best_pos = pop[best_idx].copy()
        best_fit = fitness_vals[best_idx]

        for t in range(self.max_iter):
            for i in range(self.n_pop):
                if np.random.random() < 0.5:
                    # Guided foraging
                    target = np.random.randint(self.n_pop)
                    while target == i:
                        target = np.random.randint(self.n_pop)
                    V = self._guided_foraging(pop[i], pop[target], n_features)
                else:
                    # Territorial foraging
                    V = self._territorial_foraging(pop[i], n_features)

                V = np.clip(V, -10, 10)
                bv = self._ensure_at_least_one(self._to_binary(V))
                fit_v = self._fitness(bv, X_train, y_train, X_test, y_test, clf)

                if fit_v < fitness_vals[i]:
                    pop[i] = V
                    fitness_vals[i] = fit_v
                    if fit_v < best_fit:
                        best_fit = fit_v
                        best_pos = V.copy()

            # Migration foraging for worst
            worst_idx = np.argmax(fitness_vals)
            pop[worst_idx] = np.random.uniform(-1, 1, n_features)
            bw = self._ensure_at_least_one(self._to_binary(pop[worst_idx]))
            fitness_vals[worst_idx] = self._fitness(bw, X_train, y_train, X_test, y_test, clf)
            if fitness_vals[worst_idx] < best_fit:
                best_fit = fitness_vals[worst_idx]
                best_pos = pop[worst_idx].copy()

        return self._ensure_at_least_one(self._to_binary(best_pos)), best_fit


class TwoPhaseAHA(AHA):
    """2M-AHA: AHA with Two-Phase Mutation for enhanced feature selection."""
    def __init__(self, n_pop=5, max_iter=100, mu1=0.99, mu2=0.01,
                 stagnation_threshold=10, eta=20):
        super().__init__(n_pop, max_iter, mu1, mu2)
        self.stagnation_threshold = stagnation_threshold
        self.eta = eta

    def _guided_mutation(self, X, n_features):
        """Phase 1: Polynomial mutation for exploitation."""
        X_new = X.copy()
        for j in range(n_features):
            if np.random.random() < 1.0 / n_features:
                r = np.random.random()
                if r < 0.5:
                    delta = (2 * r) ** (1.0 / (self.eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - r)) ** (1.0 / (self.eta + 1))
                X_new[j] = X[j] + delta * 2  # scale factor
        return np.clip(X_new, -10, 10)

    def _random_mutation(self, bx, n_features):
        """Phase 2: Bit-flip mutation for exploration."""
        bx_new = bx.copy()
        pm = 1.0 / n_features
        for j in range(n_features):
            if np.random.random() < pm:
                bx_new[j] = 1 - bx_new[j]
        return self._ensure_at_least_one(bx_new)

    def optimize(self, X_train, y_train, X_test, y_test, clf):
        n_features = X_train.shape[1]
        pop = self._init_population(n_features)
        fitness_vals = np.full(self.n_pop, np.inf)
        stagnation = np.zeros(self.n_pop, dtype=int)

        for i in range(self.n_pop):
            bx = self._ensure_at_least_one(self._to_binary(pop[i]))
            fitness_vals[i] = self._fitness(bx, X_train, y_train, X_test, y_test, clf)

        best_idx = np.argmin(fitness_vals)
        best_pos = pop[best_idx].copy()
        best_fit = fitness_vals[best_idx]

        for t in range(self.max_iter):
            for i in range(self.n_pop):
                if np.random.random() < 0.5:
                    target = np.random.randint(self.n_pop)
                    while target == i:
                        target = np.random.randint(self.n_pop)
                    V = self._guided_foraging(pop[i], pop[target], n_features)
                else:
                    V = self._territorial_foraging(pop[i], n_features)

                V = np.clip(V, -10, 10)
                bv = self._ensure_at_least_one(self._to_binary(V))
                fit_v = self._fitness(bv, X_train, y_train, X_test, y_test, clf)

                if fit_v < fitness_vals[i]:
                    pop[i] = V
                    fitness_vals[i] = fit_v
                    stagnation[i] = 0
                    if fit_v < best_fit:
                        best_fit = fit_v
                        best_pos = V.copy()
                else:
                    stagnation[i] += 1

            # Phase 1: Guided mutation on top-K solutions
            sorted_idx = np.argsort(fitness_vals)
            top_k = max(1, self.n_pop // 2)
            for k in range(top_k):
                idx = sorted_idx[k]
                V_mut = self._guided_mutation(pop[idx], n_features)
                bv_mut = self._ensure_at_least_one(self._to_binary(V_mut))
                fit_mut = self._fitness(bv_mut, X_train, y_train, X_test, y_test, clf)
                if fit_mut < fitness_vals[idx]:
                    pop[idx] = V_mut
                    fitness_vals[idx] = fit_mut
                    stagnation[idx] = 0
                    if fit_mut < best_fit:
                        best_fit = fit_mut
                        best_pos = V_mut.copy()

            # Phase 2: Random mutation on stagnant solutions
            for i in range(self.n_pop):
                if stagnation[i] >= self.stagnation_threshold:
                    bx = self._to_binary(pop[i])
                    bx_mut = self._random_mutation(bx, n_features)
                    fit_mut = self._fitness(bx_mut, X_train, y_train, X_test, y_test, clf)
                    if fit_mut < fitness_vals[i]:
                        pop[i] = np.where(bx_mut == 1, np.abs(pop[i]), -np.abs(pop[i]))
                        fitness_vals[i] = fit_mut
                        stagnation[i] = 0
                        if fit_mut < best_fit:
                            best_fit = fit_mut
                            best_pos = pop[i].copy()
                    stagnation[i] = 0

            # Migration foraging
            worst_idx = np.argmax(fitness_vals)
            pop[worst_idx] = np.random.uniform(-1, 1, n_features)
            bw = self._ensure_at_least_one(self._to_binary(pop[worst_idx]))
            fitness_vals[worst_idx] = self._fitness(bw, X_train, y_train, X_test, y_test, clf)
            if fitness_vals[worst_idx] < best_fit:
                best_fit = fitness_vals[worst_idx]
                best_pos = pop[worst_idx].copy()

        return self._ensure_at_least_one(self._to_binary(best_pos)), best_fit
