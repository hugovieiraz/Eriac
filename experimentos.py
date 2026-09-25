"""Experimentos da Fase A (ver PLANO.md). Resultados em saida/experimentos/.

    .venv/Scripts/python.exe experimentos.py            (tudo)
    .venv/Scripts/python.exe experimentos.py zoo aninhada   (só algumas partes)

Partes: zoo, aninhada, robustez, aumento, estatistica, temporal, sensibilidade.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import RidgeCV
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import avaliacao as av
import config
import dados
import features as ft
from modelos import REGRESSORES, MediaDeModelos
from paleta import ConversorPaleta, recuperar_paleta

DIR = config.SAIDA / "experimentos"
REGS = ["ridge", "pls", "pca_ridge", "kernel_ridge", "svr", "gp"]
VERSAO_CACHE = 2
T0 = time.perf_counter()


def log(msg: str) -> None:
    print(f"[{time.perf_counter() - T0:7.1f}s] {msg}", flush=True)


# ------------------------------------------------------------------ dados ----
def preparar() -> dict:
    catalogo = dados.construir_catalogo()
    imagens = dados.carregar_imagens(catalogo)
    conversor = ConversorPaleta(recuperar_paleta(imagens)[0])
    chave = hashlib.sha256(("".join(catalogo.sha256) + f"v{VERSAO_CACHE}").encode()).hexdigest()[:16]
    arq = config.CACHE / f"experimentos_{chave}.npz"
    config.CACHE.mkdir(exist_ok=True)
    if arq.exists():
        z = np.load(arq, allow_pickle=True)
        ind = pd.DataFrame(z["ind"], columns=list(z["ind_cols"]))
        ind_f = pd.DataFrame(z["ind_f"], columns=list(z["ind_cols"]))
        mob = z["mob"]
        log(f"features do cache {arq.name}")
    else:
        ind = ft.indicadores(imagens, conversor, filtro_mediana=0)
        ind_f = ft.indicadores(imagens, conversor, filtro_mediana=5)
        mob = ft.mobilenet(imagens)
        np.savez(arq, ind=ind.to_numpy(), ind_f=ind_f.to_numpy(), ind_cols=np.array(ind.columns), mob=mob)
        log("features calculadas e salvas no cache")
    dino, _ = ft.dinov2(catalogo)
    # "indicadores" é a versão padrão do pipeline (com filtro de mediana); a versão sem filtro
    # fica só para a comparação de robustez. Os nomes *_filtrados/*_f são apelidos usados
    # pelas partes de aumento de dados.
    inv = ind_f.drop(columns=ft.INDICADORES_DE_CENA)
    X = {
        "indicadores": ind_f.to_numpy(),
        "indicadores_sem_filtro": ind.to_numpy(),
        "indicadores_filtrados": ind_f.to_numpy(),
        "ind_invariantes": inv.to_numpy(),
        "mobilenet": mob,
        "dinov2": dino,
        "hibrido": np.column_stack([dino, ind_f.to_numpy()]),
        "hibrido_inv": np.column_stack([dino, inv.to_numpy()]),
        "hibrido_mb": np.column_stack([mob, ind_f.to_numpy()]),
        "hibrido_mb_f": np.column_stack([mob, ind_f.to_numpy()]),
        "hibrido_mb_sem_filtro": np.column_stack([mob, ind.to_numpy()]),
        # blocos para as médias de modelos: [dinov2 | mobilenet | indicadores]
        "empilhado": np.column_stack([dino, mob, ind_f.to_numpy()]),
    }
    return {"catalogo": catalogo, "imagens": imagens, "conversor": conversor, "ind": ind_f, "X": X}


def registrar_medias(d: int, m: int, k: int) -> None:
    """Médias de modelos fixadas a priori (sem seleção pelo teste). Cada membro é Ridge com
    α escolhido por validação leave-one-out interna (RidgeCV)."""
    membro = Pipeline([("escala", StandardScaler()), ("modelo", RidgeCV(alphas=config.GRADE_RIDGE))])
    dino, mob, ind = slice(0, d), slice(d, d + m), slice(d + m, d + m + k)
    REGRESSORES["media_3"] = (MediaDeModelos([(dino, membro), (mob, membro), (ind, membro)]), {})
    REGRESSORES["media_dino_mb"] = (MediaDeModelos([(dino, membro), (mob, membro)]), {})


def candidatos(X: dict) -> dict[str, tuple[np.ndarray, str]]:
    conjuntos = ["indicadores", "ind_invariantes", "mobilenet", "dinov2", "hibrido", "hibrido_inv", "hibrido_mb"]
    c = {f"{s}|{r}": (X[s], r) for s in conjuntos for r in REGS}
    c["empilhado|media_3"] = (X["empilhado"], "media_3")
    c["empilhado|media_dino_mb"] = (X["empilhado"], "media_dino_mb")
    return c


# ---------------------------------------------------------------- partes ----
def parte_zoo(P: dict) -> None:
    cat = P["catalogo"]
    interp, camp = [], []
    for nome, (Xc, reg) in candidatos(P["X"]).items():
        t = time.perf_counter()
        interp.append(av.interpolacao(Xc, cat, nome, regressor=reg))
        camp.append(av.por_campanha(Xc, cat, nome, regressor=reg))
        log(f"zoo {nome}: {interp[-1].mae_espiras.mean():.1f} espiras ({time.perf_counter() - t:.0f}s)")
    interp, camp = pd.concat(interp, ignore_index=True), pd.concat(camp, ignore_index=True)
    interp.to_csv(DIR / "zoo_interpolacao.csv", index=False)
    camp.to_csv(DIR / "zoo_campanha.csv", index=False)
    ci = camp[camp.tipo == "interpolacao"]
    resumo = interp.groupby("conjunto", sort=False).agg(
        mae_interp=("mae_espiras", "mean"), pior_interp=("mae_espiras", "max"),
        nivel_correto=("nivel_mais_proximo_correto", "mean"), cobertura_90=("conformal_cobertura", "mean"),
        meia_largura_espiras=("conformal_meia_largura", lambda s: s.mean() * config.TOTAL_ESPIRAS))
    resumo["mae_campanha_BC"] = ci.groupby("conjunto").apply(
        lambda g: (g.mae_espiras * g.n_teste).sum() / g.n_teste.sum())
    resumo.sort_values("mae_interp").round(3).to_csv(DIR / "zoo_resumo.csv")
    log("zoo concluído")


def parte_aninhada(P: dict) -> None:
    """Para cada severidade retirada, escolhe o candidato pelo erro de interpolação DENTRO do
    treino (retirando, um de cada vez, os níveis interiores restantes) e só então prevê a
    severidade retirada. O erro resultante é o do procedimento "escolher e aplicar"."""
    cat = P["catalogo"]
    niveis, alfa = cat.nivel.to_numpy(), cat.alfa.to_numpy()
    cands = candidatos(P["X"])
    linhas = []
    for held in av.niveis_interiores(cat):
        tr, te = np.flatnonzero(niveis != held), np.flatnonzero(niveis == held)
        cat_tr = cat.iloc[tr].reset_index(drop=True)
        notas = {n: av.interpolacao(Xc[tr], cat_tr, n, regressor=r, completo=False).mae_espiras.mean()
                 for n, (Xc, r) in cands.items()}
        escolhido = min(notas, key=notas.get)
        Xc, r = cands[escolhido]
        modelo, _ = av._regressao_com_grupos(Xc[tr], alfa[tr], niveis[tr], r)
        pred = np.clip(modelo.predict(Xc[te]), 0, 1)
        top3 = sorted(notas, key=notas.get)[:3]
        linhas.append({"condicao_retirada": config.ORDEM_CLASSES[held], "escolhido": escolhido,
                       "mae_interno_escolhido": notas[escolhido], "top3_interno": " ; ".join(top3),
                       "alfa_real": alfa[te][0], "alfa_previsto_medio": pred.mean(),
                       "mae_espiras": mean_absolute_error(alfa[te], pred) * config.TOTAL_ESPIRAS})
        log(f"aninhada {config.ORDEM_CLASSES[held]}: escolheu {escolhido} -> {linhas[-1]['mae_espiras']:.1f} espiras")
    df = pd.DataFrame(linhas)
    df.to_csv(DIR / "selecao_aninhada.csv", index=False)
    log(f"seleção aninhada: {df.mae_espiras.mean():.1f} espiras em média")


def parte_robustez(P: dict) -> None:
    cat, ind, imagens, conv = P["catalogo"], P["ind"], P["imagens"], P["conversor"]
    X = P["X"]
    pert = {}
    for tipo in ft.PERTURBACOES:
        imgs = ft.perturbar(imagens, tipo)
        i0 = ft.indicadores(imgs, conv, filtro_mediana=0).to_numpy()
        i5 = ft.indicadores(imgs, conv, filtro_mediana=5).to_numpy()
        mb = ft.mobilenet(imgs)
        pert[tipo] = {"indicadores_sem_filtro": i0, "indicadores": i5, "indicadores_filtrados": i5, "mobilenet": mb,
                      "hibrido_mb_sem_filtro": np.column_stack([mb, i0]), "hibrido_mb": np.column_stack([mb, i5]),
                      "hibrido_mb_f": np.column_stack([mb, i5])}
    log("features perturbadas prontas")
    linhas = []
    for c in ["indicadores_sem_filtro", "indicadores", "mobilenet", "hibrido_mb_sem_filtro", "hibrido_mb"]:
        r = av.validar(X[c], cat, ind, "blocos", c, {t: pert[t][c] for t in pert})
        limpo = r["dobras"][["conjunto", "dobra", "bin_acuracia", "mul_f1_macro", "reg_mae_espiras", "fora_dominio_fracao"]]
        linhas.append(limpo.assign(perturbacao="nenhuma"))
        linhas.append(r["robustez"])
        log(f"robustez {c}")
    df = pd.concat(linhas, ignore_index=True)
    df.groupby(["conjunto", "perturbacao"], sort=False).mean(numeric_only=True).round(4).to_csv(DIR / "robustez_filtro.csv")
    np.savez(config.CACHE / "experimentos_perturbados.npz", **{f"{t}__{c}": v for t, d in pert.items() for c, v in d.items()})


def parte_aumento(P: dict) -> None:
    """Treino com cópias perturbadas em intensidade 0,6 (ruído σ≈5, desfoque σ≈0,9,
    translação ≈7 px, rotação ≈2,4°); teste nas perturbações de intensidade 1."""
    cat, imagens, conv = P["catalogo"], P["imagens"], P["conversor"]
    arq = config.CACHE / "experimentos_perturbados.npz"
    if not arq.exists():
        parte_robustez(P)
    z = np.load(arq)
    teste = {t: {c: z[f"{t}__{c}"] for c in ["indicadores_filtrados", "mobilenet", "hibrido_mb_f"]} for t in ft.PERTURBACOES}
    aum = {}
    for tipo in ft.PERTURBACOES:
        imgs = ft.perturbar(imagens, tipo, semente=config.SEMENTE + 1, intensidade=0.6)
        i5 = ft.indicadores(imgs, conv, filtro_mediana=5).to_numpy()
        mb = ft.mobilenet(imgs)
        aum[tipo] = {"indicadores_filtrados": i5, "mobilenet": mb, "hibrido_mb_f": np.column_stack([mb, i5])}
    log("cópias de treino aumentadas prontas")
    niveis, alfa, blocos = cat.nivel.to_numpy(), cat.alfa.to_numpy(), cat.bloco.to_numpy()
    linhas = []
    for c in ["indicadores_filtrados", "mobilenet", "hibrido_mb_f"]:
        for modo in ("limpo", "aumentado"):
            for dobra, tr, te, _ in av.divisoes_externas("blocos", cat):
                if modo == "limpo":
                    Xtr, ytr_n, ytr_a, g = P["X"][c][tr], niveis[tr], alfa[tr], blocos[tr]
                else:
                    Xtr = np.vstack([P["X"][c][tr]] + [aum[t][c][tr] for t in ft.PERTURBACOES])
                    rep = 1 + len(ft.PERTURBACOES)
                    ytr_n, ytr_a, g = np.tile(niveis[tr], rep), np.tile(alfa[tr], rep), np.tile(blocos[tr], rep)
                internas = av.divisoes_internas_por_grupo(g)
                m_mul, _ = av.ajustar("multiclasse", Xtr, ytr_n, internas)
                m_reg, _ = av.ajustar("regressao", Xtr, ytr_a, internas)
                for tipo, Xte in [("nenhuma", P["X"][c])] + [(t, teste[t][c]) for t in ft.PERTURBACOES]:
                    linhas.append({"conjunto": c, "treino": modo, "perturbacao": tipo, "dobra": dobra,
                                   "mul_f1_macro": f1_score(niveis[te], m_mul.predict(Xte[te]), average="macro"),
                                   "reg_mae_espiras": mean_absolute_error(alfa[te], np.clip(m_reg.predict(Xte[te]), 0, 1)) * config.TOTAL_ESPIRAS})
            log(f"aumento {c} {modo}")
    df = pd.DataFrame(linhas)
    df.to_csv(DIR / "aumento_por_dobra.csv", index=False)
    df.groupby(["conjunto", "treino", "perturbacao"], sort=False)[["mul_f1_macro", "reg_mae_espiras"]].mean().round(4).to_csv(DIR / "aumento_resumo.csv")


def _aumentadas(P: dict, conjunto: str) -> dict[str, np.ndarray]:
    """Features das cópias de treino (intensidade 0,6, semente diferente da do teste)."""
    arq = config.CACHE / f"experimentos_aumento_{conjunto}.npz"
    if arq.exists():
        return dict(np.load(arq))
    saida = {}
    for tipo in ft.PERTURBACOES:
        imgs = ft.perturbar(P["imagens"], tipo, semente=config.SEMENTE + 1, intensidade=0.6)
        i5 = ft.indicadores(imgs, P["conversor"], filtro_mediana=5).to_numpy()
        mb = ft.mobilenet(imgs)
        saida[tipo] = {"indicadores_filtrados": i5, "mobilenet": mb, "hibrido_mb_f": np.column_stack([mb, i5])}[conjunto]
    np.savez(arq, **saida)
    return saida


def parte_aumento_generalizacao(P: dict) -> None:
    """O aumento de dados também ajuda com severidade e campanha não vistas? O treino de cada
    teste recebe as cópias perturbadas das SUAS imagens; o teste usa só imagens originais."""
    cat, X = P["catalogo"], P["X"]
    niveis, alfa, camp = cat.nivel.to_numpy(), cat.alfa.to_numpy(), cat.campanha.to_numpy()
    linhas = []
    for c in ["mobilenet", "hibrido_mb_f"]:
        aum = _aumentadas(P, c)
        rep = 1 + len(aum)
        Xa = np.vstack([X[c]] + [aum[t] for t in ft.PERTURBACOES])
        na, aa, ca = np.tile(niveis, rep), np.tile(alfa, rep), np.tile(camp, rep)
        for modo in ("limpo", "aumentado"):
            Xt, nt, at, ct = (X[c], niveis, alfa, camp) if modo == "limpo" else (Xa, na, aa, ca)
            for held in av.niveis_interiores(cat):
                tr, te = np.flatnonzero(nt != held), np.flatnonzero(niveis == held)
                m, _ = av._regressao_com_grupos(Xt[tr], at[tr], nt[tr])
                pred = np.clip(m.predict(X[c][te]), 0, 1)
                linhas.append({"conjunto": c, "treino": modo, "teste": "severidade", "retirado": config.ORDEM_CLASSES[held],
                               "n": len(te), "mae_espiras": mean_absolute_error(alfa[te], pred) * config.TOTAL_ESPIRAS})
            for k in ("B", "C"):
                tr, te = np.flatnonzero(ct != k), np.flatnonzero(camp == k)
                m, _ = av._regressao_com_grupos(Xt[tr], at[tr], nt[tr])
                pred = np.clip(m.predict(X[c][te]), 0, 1)
                linhas.append({"conjunto": c, "treino": modo, "teste": "campanha", "retirado": k,
                               "n": len(te), "mae_espiras": mean_absolute_error(alfa[te], pred) * config.TOTAL_ESPIRAS})
            log(f"aumento/generalização {c} {modo}")
    df = pd.DataFrame(linhas)
    df.to_csv(DIR / "aumento_generalizacao.csv", index=False)
    resumir_aumento_generalizacao(df).to_csv(DIR / "aumento_generalizacao_resumo.csv", index=False)


def resumir_aumento_generalizacao(df: pd.DataFrame) -> pd.DataFrame:
    """Severidade: média simples entre os níveis retirados. Campanha: média ponderada pelo
    número de imagens, como em por_campanha."""
    linhas = []
    for (c, modo, teste), g in df.groupby(["conjunto", "treino", "teste"], sort=False):
        media = (g.mae_espiras * g.n).sum() / g.n.sum() if teste == "campanha" else g.mae_espiras.mean()
        linhas.append({"conjunto": c, "treino": modo, "teste": teste, "mae_espiras": round(media, 2),
                       "pior": round(g.mae_espiras.max(), 2)})
    return pd.DataFrame(linhas)


def parte_estatistica(P: dict) -> None:
    """Comparações pareadas por severidade retirada (7 pares) e IC bootstrap da média entre
    severidades. Com 7 pares, o menor p bilateral possível do Wilcoxon é 0,016."""
    z = pd.read_csv(DIR / "zoo_interpolacao.csv")
    tab = z.pivot_table(index="condicao_retirada", columns="conjunto", values="mae_espiras")
    rng = np.random.default_rng(config.SEMENTE)
    ic = []
    for c in tab.columns:
        v = tab[c].to_numpy()
        boot = [rng.choice(v, len(v), replace=True).mean() for _ in range(10000)]
        ic.append({"conjunto": c, "mae_medio": v.mean(), "ic95_inf": np.percentile(boot, 2.5), "ic95_sup": np.percentile(boot, 97.5)})
    pd.DataFrame(ic).sort_values("mae_medio").round(2).to_csv(DIR / "ic_bootstrap_interpolacao.csv", index=False)
    pares = [("hibrido|ridge", "indicadores|ridge"), ("hibrido|ridge", "dinov2|ridge"), ("hibrido|ridge", "mobilenet|ridge"),
             ("hibrido_mb|ridge", "hibrido|ridge"), ("empilhado|media_3", "hibrido|ridge"),
             ("hibrido_inv|ridge", "hibrido|ridge"), ("dinov2|ridge", "indicadores|ridge")]
    linhas = []
    for a, b in pares:
        if a in tab and b in tab:
            d = tab[a] - tab[b]
            p = wilcoxon(tab[a], tab[b]).pvalue if (d != 0).any() else 1.0
            linhas.append({"a": a, "b": b, "diferenca_media_espiras": d.mean(), "niveis_a_melhor": int((d < 0).sum()),
                           "n_niveis": len(d), "p_wilcoxon": p})
    pd.DataFrame(linhas).round(4).to_csv(DIR / "testes_pareados_interpolacao.csv", index=False)
    log("estatística concluída")


def parte_temporal(P: dict) -> None:
    """Dentro de cada gravação: o índice de paleta e o α previsto mudam com a ordem do quadro?"""
    cat, ind = P["catalogo"], P["ind"]
    oof = pd.read_csv(config.SAIDA / "previsoes_oof_principal.csv")
    df = cat[["condicao", "ordem", "alfa"]].assign(indice_media=ind.indice_media)
    df = df.merge(oof[["indice", "alfa_previsto"]], left_index=True, right_on="indice")
    linhas = []
    for cond, g in df.groupby("condicao", sort=False):
        g = g.sort_values("ordem")
        r_ind = spearmanr(g.ordem, g.indice_media)
        r_alf = spearmanr(g.ordem, g.alfa_previsto)
        inc = np.polyfit(g.ordem, g.alfa_previsto * config.TOTAL_ESPIRAS, 1)[0]
        linhas.append({"condicao": cond, "n": len(g), "spearman_indice": r_ind.statistic, "p_indice": r_ind.pvalue,
                       "spearman_alfa_prev": r_alf.statistic, "p_alfa_prev": r_alf.pvalue,
                       "inclinacao_espiras_por_quadro": inc,
                       "variacao_espiras_na_gravacao": inc * (len(g) - 1)})
    pd.DataFrame(linhas).round(4).to_csv(DIR / "deriva_temporal.csv", index=False)
    log("análise temporal concluída")


def parte_sensibilidade(P: dict) -> None:
    cat, imagens, conv, X = P["catalogo"], P["imagens"], P["conversor"], P["X"]
    dino = X["dinov2"]
    linhas = []
    for limiar in (0.55, 0.65, 0.75):
        ind = ft.indicadores(imagens, conv, limiar_quente=limiar)
        Xh = np.column_stack([dino, ind.to_numpy()])
        r = av.validar(Xh, cat, ind, "blocos", "hibrido")["dobras"]
        it = av.interpolacao(Xh, cat, "hibrido", completo=False)
        linhas.append({"variavel": "limiar_quente", "valor": limiar, "f1_blocos": r.mul_f1_macro.mean(),
                       "mae_blocos": r.reg_mae_espiras.mean(), "mae_interpolacao": it.mae_espiras.mean()})
        log(f"sensibilidade limiar {limiar}")
    for nb in (3, 5, 8):
        cat_b = cat.copy()
        cat_b["bloco"] = np.concatenate([dados.atribuir_blocos(len(g), nb) for _, g in cat.groupby("pasta", sort=False)])
        for purga in (0, 1, 3):
            r = av.validar(X["hibrido"], cat_b, P["ind"], "blocos", "hibrido", purga=purga)["dobras"]
            linhas.append({"variavel": "blocos_purga", "valor": f"{nb} blocos, purga {purga}",
                           "f1_blocos": r.mul_f1_macro.mean(), "mae_blocos": r.reg_mae_espiras.mean(),
                           "mae_interpolacao": np.nan})
        log(f"sensibilidade {nb} blocos")
    pd.DataFrame(linhas).round(4).to_csv(DIR / "sensibilidade.csv", index=False)


PARTES = {"zoo": parte_zoo, "aninhada": parte_aninhada, "robustez": parte_robustez, "aumento": parte_aumento,
          "aumento_generalizacao": parte_aumento_generalizacao,
          "estatistica": parte_estatistica, "temporal": parte_temporal, "sensibilidade": parte_sensibilidade}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("partes", nargs="*", default=list(PARTES))
    args = ap.parse_args()
    DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(config.SEMENTE)
    P = preparar()
    registrar_medias(P["X"]["dinov2"].shape[1], P["X"]["mobilenet"].shape[1], P["X"]["indicadores"].shape[1])
    for parte in args.partes:
        PARTES[parte](P)
    (DIR / "LEIA.json").write_text(json.dumps({"partes": args.partes, "regressores": REGS}, indent=1), encoding="utf-8")
    log("fim")


if __name__ == "__main__":
    main()
