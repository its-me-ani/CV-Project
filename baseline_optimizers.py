"""
Baseline metaheuristic optimizers for comparison: GWO, HHO, WOA, SSO, PSO
All adapted for binary feature selection with the same interface as AHA.
"""
import numpy as np
from scipy.special import expit
from aha_optimizer import AHA


class GWO(AHA):
    """Grey Wolf Optimizer for feature selection."""
    def optimize(self, X_train, y_train, X_test, y_test, clf):
        n_features = X_train.shape[1]
        pop = self._init_population(n_features)
        fitness_vals = np.full(self.n_pop, np.inf)
        for i in range(self.n_pop):
            bx = self._ensure_at_least_one(self._to_binary(pop[i]))
            fitness_vals[i] = self._fitness(bx, X_train, y_train, X_test, y_test, clf)

        for t in range(self.max_iter):
            sorted_idx = np.argsort(fitness_vals)
            alpha, beta, delta = pop[sorted_idx[0]], pop[sorted_idx[1]], pop[sorted_idx[min(2, self.n_pop-1)]]
            a = 2 - 2 * t / self.max_iter
            for i in range(self.n_pop):
                for j in range(n_features):
                    r1, r2 = np.random.random(2)
                    A1 = 2 * a * r1 - a; C1 = 2 * r2
                    D_alpha = abs(C1 * alpha[j] - pop[i][j])
                    X1 = alpha[j] - A1 * D_alpha
                    r1, r2 = np.random.random(2)
                    A2 = 2 * a * r1 - a; C2 = 2 * r2
                    D_beta = abs(C2 * beta[j] - pop[i][j])
                    X2 = beta[j] - A2 * D_beta
                    r1, r2 = np.random.random(2)
                    A3 = 2 * a * r1 - a; C3 = 2 * r2
                    D_delta = abs(C3 * delta[j] - pop[i][j])
                    X3 = delta[j] - A3 * D_delta
                    pop[i][j] = (X1 + X2 + X3) / 3
                pop[i] = np.clip(pop[i], -10, 10)
                bx = self._ensure_at_least_one(self._to_binary(pop[i]))
                fit = self._fitness(bx, X_train, y_train, X_test, y_test, clf)
                fitness_vals[i] = fit
        best_idx = np.argmin(fitness_vals)
        return self._ensure_at_least_one(self._to_binary(pop[best_idx])), fitness_vals[best_idx]


class HHO(AHA):
    """Harris Hawks Optimization for feature selection."""
    def optimize(self, X_train, y_train, X_test, y_test, clf):
        n_features = X_train.shape[1]
        pop = self._init_population(n_features)
        fitness_vals = np.full(self.n_pop, np.inf)
        for i in range(self.n_pop):
            bx = self._ensure_at_least_one(self._to_binary(pop[i]))
            fitness_vals[i] = self._fitness(bx, X_train, y_train, X_test, y_test, clf)

        for t in range(self.max_iter):
            best_idx = np.argmin(fitness_vals)
            rabbit = pop[best_idx].copy()
            E0 = 2 * np.random.random() - 1
            E = 2 * E0 * (1 - t / self.max_iter)
            for i in range(self.n_pop):
                if abs(E) >= 1:  # exploration
                    rand_idx = np.random.randint(self.n_pop)
                    X_rand = pop[rand_idx]
                    pop[i] = X_rand - np.random.random() * np.abs(X_rand - 2 * np.random.random() * pop[i])
                else:  # exploitation
                    r = np.random.random()
                    if r >= 0.5 and abs(E) >= 0.5:  # soft besiege
                        pop[i] = rabbit - E * np.abs(rabbit - pop[i])
                    elif r >= 0.5 and abs(E) < 0.5:  # hard besiege
                        pop[i] = rabbit - E * np.abs(rabbit - pop[i])
                    else:  # rapid dives
                        J = 2 * (1 - np.random.random())
                        pop[i] = rabbit - E * np.abs(J * rabbit - pop[i])
                pop[i] = np.clip(pop[i], -10, 10)
                bx = self._ensure_at_least_one(self._to_binary(pop[i]))
                fitness_vals[i] = self._fitness(bx, X_train, y_train, X_test, y_test, clf)
        best_idx = np.argmin(fitness_vals)
        return self._ensure_at_least_one(self._to_binary(pop[best_idx])), fitness_vals[best_idx]


class WOA(AHA):
    """Whale Optimization Algorithm for feature selection."""
    def optimize(self, X_train, y_train, X_test, y_test, clf):
        n_features = X_train.shape[1]
        pop = self._init_population(n_features)
        fitness_vals = np.full(self.n_pop, np.inf)
        for i in range(self.n_pop):
            bx = self._ensure_at_least_one(self._to_binary(pop[i]))
            fitness_vals[i] = self._fitness(bx, X_train, y_train, X_test, y_test, clf)

        for t in range(self.max_iter):
            best_idx = np.argmin(fitness_vals)
            leader = pop[best_idx].copy()
            a = 2 - 2 * t / self.max_iter
            for i in range(self.n_pop):
                r = np.random.random()
                A = 2 * a * np.random.random() - a
                C = 2 * np.random.random()
                p = np.random.random()
                b_const = 1
                l = np.random.uniform(-1, 1)
                if p < 0.5:
                    if abs(A) < 1:
                        D = np.abs(C * leader - pop[i])
                        pop[i] = leader - A * D
                    else:
                        rand_idx = np.random.randint(self.n_pop)
                        X_rand = pop[rand_idx]
                        D = np.abs(C * X_rand - pop[i])
                        pop[i] = X_rand - A * D
                else:
                    D = np.abs(leader - pop[i])
                    pop[i] = D * np.exp(b_const * l) * np.cos(2 * np.pi * l) + leader
                pop[i] = np.clip(pop[i], -10, 10)
                bx = self._ensure_at_least_one(self._to_binary(pop[i]))
                fitness_vals[i] = self._fitness(bx, X_train, y_train, X_test, y_test, clf)
        best_idx = np.argmin(fitness_vals)
        return self._ensure_at_least_one(self._to_binary(pop[best_idx])), fitness_vals[best_idx]


class SSO(AHA):
    """Salp Swarm Optimization for feature selection."""
    def optimize(self, X_train, y_train, X_test, y_test, clf):
        n_features = X_train.shape[1]
        pop = self._init_population(n_features)
        fitness_vals = np.full(self.n_pop, np.inf)
        for i in range(self.n_pop):
            bx = self._ensure_at_least_one(self._to_binary(pop[i]))
            fitness_vals[i] = self._fitness(bx, X_train, y_train, X_test, y_test, clf)

        ub, lb = 10, -10
        for t in range(self.max_iter):
            best_idx = np.argmin(fitness_vals)
            food = pop[best_idx].copy()
            c1 = 2 * np.exp(-(4 * t / self.max_iter) ** 2)
            for i in range(self.n_pop):
                if i == 0:  # leader
                    for j in range(n_features):
                        c2 = np.random.random()
                        c3 = np.random.random()
                        if c3 < 0.5:
                            pop[i][j] = food[j] + c1 * ((ub - lb) * c2 + lb)
                        else:
                            pop[i][j] = food[j] - c1 * ((ub - lb) * c2 + lb)
                else:  # follower
                    pop[i] = (pop[i] + pop[i - 1]) / 2
                pop[i] = np.clip(pop[i], lb, ub)
                bx = self._ensure_at_least_one(self._to_binary(pop[i]))
                fitness_vals[i] = self._fitness(bx, X_train, y_train, X_test, y_test, clf)
        best_idx = np.argmin(fitness_vals)
        return self._ensure_at_least_one(self._to_binary(pop[best_idx])), fitness_vals[best_idx]


class PSO(AHA):
    """Particle Swarm Optimization for feature selection."""
    def __init__(self, n_pop=5, max_iter=100, mu1=0.99, mu2=0.01, w=0.7, c1=1.5, c2=1.5):
        super().__init__(n_pop, max_iter, mu1, mu2)
        self.w = w; self.c1 = c1; self.c2 = c2

    def optimize(self, X_train, y_train, X_test, y_test, clf):
        n_features = X_train.shape[1]
        pop = self._init_population(n_features)
        vel = np.random.uniform(-1, 1, (self.n_pop, n_features))
        fitness_vals = np.full(self.n_pop, np.inf)
        pbest = pop.copy()
        pbest_fit = fitness_vals.copy()
        for i in range(self.n_pop):
            bx = self._ensure_at_least_one(self._to_binary(pop[i]))
            fitness_vals[i] = self._fitness(bx, X_train, y_train, X_test, y_test, clf)
            pbest_fit[i] = fitness_vals[i]
        gbest_idx = np.argmin(pbest_fit)
        gbest = pbest[gbest_idx].copy()
        gbest_fit = pbest_fit[gbest_idx]

        for t in range(self.max_iter):
            w = self.w - 0.4 * t / self.max_iter
            for i in range(self.n_pop):
                r1, r2 = np.random.random(n_features), np.random.random(n_features)
                vel[i] = w * vel[i] + self.c1 * r1 * (pbest[i] - pop[i]) + self.c2 * r2 * (gbest - pop[i])
                vel[i] = np.clip(vel[i], -4, 4)
                pop[i] = pop[i] + vel[i]
                pop[i] = np.clip(pop[i], -10, 10)
                bx = self._ensure_at_least_one(self._to_binary(pop[i]))
                fitness_vals[i] = self._fitness(bx, X_train, y_train, X_test, y_test, clf)
                if fitness_vals[i] < pbest_fit[i]:
                    pbest[i] = pop[i].copy()
                    pbest_fit[i] = fitness_vals[i]
                    if fitness_vals[i] < gbest_fit:
                        gbest = pop[i].copy()
                        gbest_fit = fitness_vals[i]
        return self._ensure_at_least_one(self._to_binary(gbest)), gbest_fit
