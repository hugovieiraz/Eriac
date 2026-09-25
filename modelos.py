"""Regressores candidatos para a severidade α. Cada um é (pipeline, grade de hiperparâmetros).

A grade é explorada por validação interna dentro do treino, nunca no teste.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import config


class PLS1D(BaseEstimator, RegressorMixin):
    """PLSRegression devolve (n, 1); aqui devolve (n,) como os demais regressores."""

    def __init__(self, n_components: int = 4):
        self.n_components = n_components

    def fit(self, X, y):
        n = max(1, min(self.n_components, X.shape[1], X.shape[0] - 1))
        self.pls_ = PLSRegression(n_components=n, scale=False).fit(X, y)
        return self

    def predict(self, X):
        return self.pls_.predict(X).ravel()


class PCASeguro(PCA):
    """PCA que reduz n_components quando o conjunto tem menos amostras ou colunas."""

    def fit(self, X, y=None):
        self._limitar(X)
        return super().fit(X, y)

    def fit_transform(self, X, y=None):
        self._limitar(X)
        return super().fit_transform(X, y)

    def _limitar(self, X):
        if isinstance(self.n_components, int):
            self.n_components = min(self.n_components, X.shape[0], X.shape[1])


def _pipe(*passos) -> Pipeline:
    return Pipeline([("escala", StandardScaler()), *passos])


REGRESSORES: dict[str, tuple[Pipeline, dict]] = {
    "ridge": (_pipe(("modelo", Ridge())), {"modelo__alpha": config.GRADE_RIDGE}),
    "pls": (_pipe(("modelo", PLS1D())), {"modelo__n_components": [2, 4, 8, 16]}),
    "pca_ridge": (_pipe(("pca", PCASeguro(random_state=config.SEMENTE)), ("modelo", Ridge())),
                  {"pca__n_components": [10, 30], "modelo__alpha": [1.0, 10.0]}),
    "kernel_ridge": (_pipe(("modelo", KernelRidge(kernel="rbf"))), {"modelo__alpha": [0.01, 0.1, 1.0]}),
    "svr": (_pipe(("modelo", SVR(kernel="rbf", gamma="scale", epsilon=0.01))), {"modelo__C": [0.3, 3.0, 30.0]}),
    "gp": (_pipe(("pca", PCASeguro(n_components=20, random_state=config.SEMENTE)),
                 ("modelo", GaussianProcessRegressor(
                     kernel=ConstantKernel(1.0) * RBF(length_scale=5.0) + WhiteKernel(1e-3),
                     normalize_y=True, random_state=config.SEMENTE))), {}),
}


class MediaDeModelos(BaseEstimator, RegressorMixin):
    """Média das previsões de regressores ajustados em blocos de colunas diferentes.

    `blocos` é uma lista de (fatia de colunas, regressor). Os membros são fixados antes de
    olhar qualquer resultado, então a média não envolve seleção pelo teste."""

    def __init__(self, blocos):
        self.blocos = blocos

    def fit(self, X, y):
        self.ajustados_ = [(fatia, clone(reg).fit(X[:, fatia], y)) for fatia, reg in self.blocos]
        return self

    def predict(self, X):
        return np.mean([reg.predict(X[:, fatia]) for fatia, reg in self.ajustados_], axis=0)
