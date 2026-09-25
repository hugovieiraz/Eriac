"""Protocolos de avaliação.

Esquemas de divisão:
- aleatoria: StratifiedKFold por imagem. Quadros vizinhos quase idênticos caem dos dois
  lados; serve só para medir o quanto esse vazamento infla os números.
- blocos: cada dobra testa o bloco temporal k de TODAS as classes; o treino perde os
  quadros vizinhos à fronteira do bloco de teste (purga). É o esquema principal.
- interpolacao: uma severidade intermediária sai inteira do treino (só regressão).
- campanha: uma campanha de aquisição inteira sai do treino, junto com o enquadramento
  dela. É o teste mais duro que este dataset permite.

Hiperparâmetros são escolhidos por validação interna, dentro do treino de cada dobra.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    precision_score, r2_score, recall_score, roc_auc_score, root_mean_squared_error,
)
from sklearn.model_selection import (
    GridSearchCV, LeaveOneGroupOut, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from dados import vizinhos_para_purgar

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ALFAS = np.array(config.ALFA_NOMINAL)
INDICADORES_COERENCIA = ["fracao_area_quente", "indice_media", "gradiente_medio", "hotspot_distancia_centro"]


# ------------------------------------------------------------- utilitários ----
def classe_mais_proxima(alfa: np.ndarray) -> np.ndarray:
    """Nível nominal cujo alfa está mais perto. As classes NÃO são igualmente espaçadas
    (SC560 -> SC600 são 40 espiras, as demais 80), então round(alfa * 7.5) erra."""
    alfa = np.atleast_1d(alfa)
    return np.abs(alfa[:, None] - ALFAS[None, :]).argmin(axis=1)


def _clf(C: float = 1.0) -> Pipeline:
    return Pipeline([("escala", StandardScaler()),
                     ("modelo", LogisticRegression(C=C, max_iter=5000, class_weight="balanced",
                                                   random_state=config.SEMENTE))])


def _reg(alpha: float = 10.0) -> Pipeline:
    return Pipeline([("escala", StandardScaler()), ("modelo", Ridge(alpha=alpha))])


def ajustar(tarefa: str, X: np.ndarray, y: np.ndarray, divisoes_internas, fixo: float | None = None):
    """Ajusta o modelo da tarefa. Com `fixo`, usa o hiperparâmetro dado (protocolo original);
    senão escolhe por GridSearch nas divisões internas. Devolve (modelo, hiperparâmetro)."""
    if tarefa == "regressao":
        base, grade, nome, pontuacao = _reg(), config.GRADE_RIDGE, "modelo__alpha", "neg_mean_absolute_error"
    else:
        base, grade, nome, pontuacao = _clf(), config.GRADE_C, "modelo__C", (
            "balanced_accuracy" if tarefa == "binaria" else "f1_macro")
    if fixo is not None:
        return clone(base).set_params(**{nome: fixo}).fit(X, y), fixo
    busca = GridSearchCV(base, {nome: grade}, cv=divisoes_internas, scoring=pontuacao, n_jobs=-1, refit=True)
    busca.fit(X, y)
    return busca.best_estimator_, busca.best_params_[nome]


def divisoes_internas_por_grupo(grupos: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    return list(LeaveOneGroupOut().split(np.zeros(len(grupos)), groups=grupos))


def divisoes_internas_estratificadas(y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    return list(StratifiedKFold(4, shuffle=True, random_state=config.SEMENTE).split(np.zeros(len(y)), y))


# ----------------------------------------------------------------- métricas ----
def metricas_binarias(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "acuracia": accuracy_score(y, pred),
        "precisao": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "especificidade": tn / (tn + fp) if tn + fp else np.nan,
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, prob) if len(np.unique(y)) == 2 else np.nan,
        "saudaveis_no_teste": int((y == 0).sum()),
    }


def metricas_multiclasse(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    dist = np.abs(y - pred)
    return {
        "acuracia_balanceada": balanced_accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "nivel_correto": float(np.mean(dist == 0)),
        "erro_um_nivel": float(np.mean(dist == 1)),
        "erro_dois_ou_mais": float(np.mean(dist >= 2)),
    }


def metricas_regressao(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y, pred)
    return {
        "mae_alfa": mae,
        "rmse_alfa": float(root_mean_squared_error(y, pred)),
        # R² não é definido quando o alvo do teste é constante (uma classe só).
        "r2": r2_score(y, pred) if np.ptp(y) > 0 else np.nan,
        "mae_espiras": mae * config.TOTAL_ESPIRAS,
        "vies_alfa": float(np.mean(pred - y)),
    }


# ------------------------------------------------ incerteza, domínio, coerência ----
def quantil_conformal(modelo, X: np.ndarray, y: np.ndarray, divisoes) -> float:
    """Meia-largura do intervalo conformal (split por validação cruzada interna): quantil
    ajustado dos resíduos absolutos fora da dobra no treino."""
    oof = np.clip(cross_val_predict(clone(modelo), X, y, cv=divisoes), 0, 1)
    res = np.abs(oof - y)
    n = len(res)
    nivel = min(1.0, np.ceil((n + 1) * config.NIVEL_INTERVALO) / n)
    return float(np.quantile(res, nivel, method="higher"))


@dataclass
class DetectorDominio:
    """Distância média aos k vizinhos de treino no espaço padronizado. Limiar = percentil 99
    da mesma distância calculada no próprio treino (excluindo o próprio ponto)."""
    k: int = 5
    escala: StandardScaler = field(default_factory=StandardScaler)

    def ajustar(self, X: np.ndarray) -> "DetectorDominio":
        Z = self.escala.fit_transform(X)
        self.vizinhos = NearestNeighbors(n_neighbors=self.k + 1).fit(Z)
        d, _ = self.vizinhos.kneighbors(Z)
        self.limiar = float(np.percentile(d[:, 1:].mean(axis=1), 99))
        return self

    def pontuar(self, X: np.ndarray) -> np.ndarray:
        d, _ = self.vizinhos.kneighbors(self.escala.transform(X), n_neighbors=self.k)
        return d.mean(axis=1) / self.limiar  # > 1 significa fora do domínio


def coerencia_visual(ind_treino: pd.DataFrame, alfa_treino: np.ndarray, ind_teste: pd.DataFrame,
                     alfa_previsto: np.ndarray) -> np.ndarray:
    """Compara os indicadores da imagem com a mediana das 10 amostras DE TREINO de alfa mais
    próximo do previsto. 1 = idêntico ao perfil esperado. Não é validação eletrotérmica."""
    escala = ind_treino[INDICADORES_COERENCIA].std().replace(0, 1.0).to_numpy()
    valores = ind_treino[INDICADORES_COERENCIA].to_numpy()
    obs = ind_teste[INDICADORES_COERENCIA].to_numpy()
    saida = np.empty(len(alfa_previsto))
    for i, a in enumerate(alfa_previsto):
        perto = np.argsort(np.abs(alfa_treino - a), kind="stable")[:10]
        esperado = np.median(valores[perto], axis=0)
        saida[i] = np.exp(-np.mean(np.abs(obs[i] - esperado) / escala))
    return saida


# --------------------------------------------------------- divisões externas ----
def divisoes_externas(esquema: str, catalogo: pd.DataFrame):
    """Gera (nome_dobra, treino, teste, divisoes_internas_fn)."""
    niveis = catalogo.nivel.to_numpy()
    n = len(catalogo)
    if esquema == "aleatoria":
        for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=config.SEMENTE).split(np.zeros(n), niveis)):
            yield f"a{k}", tr, te, lambda tr_: divisoes_internas_estratificadas(niveis[tr_])
    elif esquema == "original":
        rep = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=config.SEMENTE)
        for k, (tr, te) in enumerate(rep.split(np.zeros(n), niveis)):
            yield f"o{k}", tr, te, None
    elif esquema == "blocos":
        blocos = catalogo.bloco.to_numpy()
        for b in range(config.N_BLOCOS):
            te = np.flatnonzero(blocos == b)
            purgar = vizinhos_para_purgar(catalogo, te)
            tr = np.setdiff1d(np.flatnonzero(blocos != b), purgar)
            yield f"b{b}", tr, te, lambda tr_: divisoes_internas_por_grupo(blocos[tr_])
    else:
        raise ValueError(esquema)


# ------------------------------------------------------------- validação ----
def validar(X: np.ndarray, catalogo: pd.DataFrame, ind: pd.DataFrame, esquema: str, nome: str,
            testes_extras: dict[str, np.ndarray] | None = None) -> dict:
    """Validação cruzada das três tarefas. `testes_extras` são versões alternativas de X
    (imagens perturbadas) avaliadas com os mesmos modelos treinados em dados limpos."""
    niveis = catalogo.nivel.to_numpy()
    alfa = catalogo.alfa.to_numpy()
    binario = (niveis > 0).astype(int)
    fixo = esquema == "original"
    linhas, oof, extras = [], [], []
    confusao = np.zeros((9, 9), dtype=int)
    for dobra, tr, te, internas_fn in divisoes_externas(esquema, catalogo):
        internas = None if fixo else internas_fn(tr)
        m_bin, c_bin = ajustar("binaria", X[tr], binario[tr], internas, config.C_ORIGINAL if fixo else None)
        m_mul, c_mul = ajustar("multiclasse", X[tr], niveis[tr], internas, config.C_ORIGINAL if fixo else None)
        m_reg, a_reg = ajustar("regressao", X[tr], alfa[tr], internas, config.RIDGE_ORIGINAL if fixo else None)
        p_bin = m_bin.predict_proba(X[te])[:, 1]
        p_mul = m_mul.predict(X[te])
        conf = m_mul.predict_proba(X[te]).max(axis=1)
        p_reg = np.clip(m_reg.predict(X[te]), 0, 1)
        linha = {
            "conjunto": nome, "esquema": esquema, "dobra": dobra, "n_treino": len(tr), "n_teste": len(te),
            "C_binaria": c_bin, "C_multiclasse": c_mul, "ridge_alpha": a_reg,
            **{f"bin_{k}": v for k, v in metricas_binarias(binario[te], p_bin).items()},
            **{f"mul_{k}": v for k, v in metricas_multiclasse(niveis[te], p_mul).items()},
            **{f"reg_{k}": v for k, v in metricas_regressao(alfa[te], p_reg).items()},
            "reg_nivel_mais_proximo_correto": float(np.mean(classe_mais_proxima(p_reg) == niveis[te])),
        }
        confusao += confusion_matrix(niveis[te], p_mul, labels=np.arange(9))
        if esquema == "blocos":
            q = quantil_conformal(m_reg, X[tr], alfa[tr], internas)
            dominio = DetectorDominio().ajustar(X[tr])
            score_dom = dominio.pontuar(X[te])
            coer = coerencia_visual(ind.iloc[tr], alfa[tr], ind.iloc[te], p_reg)
            cobre = (alfa[te] >= np.clip(p_reg - q, 0, 1)) & (alfa[te] <= np.clip(p_reg + q, 0, 1))
            linha.update({"conformal_meia_largura": q, "conformal_cobertura": float(cobre.mean()),
                          "fora_dominio_fracao": float(np.mean(score_dom > 1))})
            for j, i in enumerate(te):
                oof.append({
                    "indice": int(i), "dobra": dobra, "prob_defeito": float(p_bin[j]),
                    "nivel_previsto": int(p_mul[j]), "confianca_classificador": float(conf[j]),
                    "alfa_previsto": float(p_reg[j]), "alfa_inferior": float(max(0.0, p_reg[j] - q)),
                    "alfa_superior": float(min(1.0, p_reg[j] + q)), "score_dominio": float(score_dom[j]),
                    "coerencia_visual": float(coer[j]),
                })
            for tipo, X_alt in (testes_extras or {}).items():
                pb = m_bin.predict_proba(X_alt[te])[:, 1]
                pm = m_mul.predict(X_alt[te])
                pr = np.clip(m_reg.predict(X_alt[te]), 0, 1)
                extras.append({"conjunto": nome, "perturbacao": tipo, "dobra": dobra,
                               "bin_acuracia": accuracy_score(binario[te], (pb >= 0.5).astype(int)),
                               "mul_f1_macro": f1_score(niveis[te], pm, average="macro", zero_division=0),
                               "reg_mae_espiras": mean_absolute_error(alfa[te], pr) * config.TOTAL_ESPIRAS,
                               "fora_dominio_fracao": float(np.mean(dominio.pontuar(X_alt[te]) > 1))})
        linhas.append(linha)
    return {"dobras": pd.DataFrame(linhas), "confusao": confusao,
            "oof": pd.DataFrame(oof), "robustez": pd.DataFrame(extras)}


def resumir(dobras: pd.DataFrame) -> pd.DataFrame:
    metricas = [c for c in dobras.columns if c.startswith(("bin_", "mul_", "reg_", "conformal_", "fora_"))]
    g = dobras.groupby(["conjunto", "esquema"], sort=False)[metricas]
    media, desvio = g.mean(), g.std(ddof=1)
    saida = media.copy()
    for c in metricas:
        saida[c + "_dp"] = desvio[c]
    return saida.reset_index()


# ------------------------------------------------------- testes de generalização ----
def _regressao_com_grupos(X_tr, y_tr, grupos_tr):
    return ajustar("regressao", X_tr, y_tr, divisoes_internas_por_grupo(grupos_tr))


def interpolacao(X: np.ndarray, catalogo: pd.DataFrame, nome: str) -> pd.DataFrame:
    """Retira uma severidade intermediária inteira; validação interna agrupada por nível,
    para que a escolha do hiperparâmetro também simule um nível não visto."""
    niveis, alfa = catalogo.nivel.to_numpy(), catalogo.alfa.to_numpy()
    linhas = []
    for held in range(1, 8):
        tr, te = np.flatnonzero(niveis != held), np.flatnonzero(niveis == held)
        modelo, a = _regressao_com_grupos(X[tr], alfa[tr], niveis[tr])
        pred = np.clip(modelo.predict(X[te]), 0, 1)
        dominio = DetectorDominio().ajustar(X[tr])
        linhas.append({
            "conjunto": nome, "condicao_retirada": config.ORDEM_CLASSES[held], "alfa_real": alfa[te][0],
            "n_teste": len(te), "ridge_alpha": a, "alfa_previsto_medio": pred.mean(),
            "alfa_previsto_dp": pred.std(ddof=1),
            **{k: v for k, v in metricas_regressao(alfa[te], pred).items() if k != "r2"},
            "nivel_mais_proximo_correto": float(np.mean(classe_mais_proxima(pred) == held)),
            "fora_dominio_fracao": float(np.mean(dominio.pontuar(X[te]) > 1)),
        })
    return pd.DataFrame(linhas)


def por_campanha(X: np.ndarray, catalogo: pd.DataFrame, nome: str) -> pd.DataFrame:
    """Retira uma campanha de aquisição inteira (com seu enquadramento de câmera)."""
    niveis, alfa, camp = catalogo.nivel.to_numpy(), catalogo.alfa.to_numpy(), catalogo.campanha.to_numpy()
    linhas = []
    for c in sorted(np.unique(camp)):
        tr, te = np.flatnonzero(camp != c), np.flatnonzero(camp == c)
        faixa_tr = (alfa[tr].min(), alfa[tr].max())
        tipo = "interpolacao" if faixa_tr[0] < alfa[te].min() and alfa[te].max() < faixa_tr[1] else "extrapolacao"
        modelo, a = _regressao_com_grupos(X[tr], alfa[tr], niveis[tr])
        pred = np.clip(modelo.predict(X[te]), 0, 1)
        dominio = DetectorDominio().ajustar(X[tr])
        fora = dominio.pontuar(X[te]) > 1
        for nv in np.unique(niveis[te]):
            s = niveis[te] == nv
            linhas.append({
                "conjunto": nome, "campanha_retirada": c, "tipo": tipo,
                "condicao": config.ORDEM_CLASSES[nv], "alfa_real": alfa[te][s][0], "n_teste": int(s.sum()),
                "alfa_previsto_medio": pred[s].mean(),
                "mae_espiras": mean_absolute_error(alfa[te][s], pred[s]) * config.TOTAL_ESPIRAS,
                "vies_espiras": float(np.mean(pred[s] - alfa[te][s]) * config.TOTAL_ESPIRAS),
                "fora_dominio_fracao": float(fora[s].mean()),
            })
    return pd.DataFrame(linhas)


def comparacao_pareada(dobras: pd.DataFrame, a: str, b: str, metrica: str, maior_melhor: bool) -> dict:
    """Diferença dobra a dobra (mesmas dobras para os dois conjuntos)."""
    da = dobras[dobras.conjunto == a].set_index("dobra")[metrica]
    db = dobras[dobras.conjunto == b].set_index("dobra")[metrica]
    dif = (da - db).loc[da.index]
    ganha = (dif > 0) if maior_melhor else (dif < 0)
    return {"a": a, "b": b, "metrica": metrica, "diferenca_media": float(dif.mean()),
            "dobras_a_vence": int(ganha.sum()), "dobras_empate": int((dif == 0).sum()), "n_dobras": int(len(dif))}
