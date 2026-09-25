"""O mesmo pipeline aplicado ao motor de indução do mesmo dataset (369 imagens, 11 condições).

Serve para testar o "framework generalista" do resumo: nada aqui é ajustado para o motor
além do catálogo (nomes das condições e a definição de severidade).

Severidade do motor. As condições de curto no estator combinam a porcentagem de espiras em
curto por fase (10, 30 ou 50%) com o número de fases afetadas (1, 2 ou 3). Definimos
α = porcentagem × fases / 3, a fração do total de espiras do estator em curto, supondo fases
com o mesmo número de espiras. É uma hipótese nossa, não um dado do dataset. Rotor travado
(Rotor-0) e falha do ventilador (Fan) não têm severidade e entram só na classificação.

    .venv/Scripts/python.exe motor.py
"""
from __future__ import annotations

import hashlib
import json
import re
import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

import avaliacao as av
import config
import dados
import features as ft
from paleta import ConversorPaleta, recuperar_paleta

DATASET = config.RAIZ / "dados" / "IR_Motor_bmp"
SAIDA = config.SAIDA / "motor"

# pasta -> (porcentagem por fase, fases). Ordem = severidade crescente dentro do estator.
ESTATOR = {
    "Noload": (0, 0), "A10": (10, 1), "A&C10": (10, 2), "A&C&B10": (10, 3), "A30": (30, 1),
    "A50": (50, 1), "A&C30": (30, 2), "A&C&B30": (30, 3), "A&B50": (50, 2),
}
OUTRAS = ["Fan", "Rotor-0"]
T0 = time.perf_counter()


def log(msg: str) -> None:
    print(f"[{time.perf_counter() - T0:6.1f}s] {msg}", flush=True)


def alfa_motor(pasta: str) -> float:
    pct, fases = ESTATOR[pasta]
    return pct / 100 * fases / 3


def catalogo_motor() -> pd.DataFrame:
    ordem_pastas = list(ESTATOR) + OUTRAS
    linhas = []
    for nivel, pasta in enumerate(ordem_pastas):
        arquivos = sorted((DATASET / pasta).glob("*.bmp"), key=lambda p: int(re.search(r"(\d+)", p.name).group(1)))
        for ordem, (caminho, bloco) in enumerate(zip(arquivos, dados.atribuir_blocos(len(arquivos)))):
            linhas.append({"arquivo": caminho.name, "pasta": pasta, "caminho": str(caminho), "condicao": pasta,
                           "nivel": nivel, "alfa": alfa_motor(pasta) if pasta in ESTATOR else np.nan,
                           "ordem": ordem, "bloco": int(bloco), "campanha": "unica",
                           "sha256": hashlib.sha256(caminho.read_bytes()).hexdigest()})
    cat = pd.DataFrame(linhas)
    if len(cat) != 369:
        raise RuntimeError(f"Esperadas 369 imagens do motor; encontradas {len(cat)}.")
    return cat


def classificar_11(X: np.ndarray, cat: pd.DataFrame, nome: str, esquema: str) -> tuple[dict, np.ndarray]:
    niveis = cat.nivel.to_numpy()
    f1s, conf = [], np.zeros((11, 11), dtype=int)
    for _, tr, te, internas_fn in av.divisoes_externas(esquema, cat):
        modelo, _ = av.ajustar("multiclasse", X[tr], niveis[tr], internas_fn(tr))
        p = modelo.predict(X[te])
        f1s.append(f1_score(niveis[te], p, average="macro"))
        conf += confusion_matrix(niveis[te], p, labels=np.arange(11))
    return {"conjunto": nome, "esquema": esquema, "f1_macro": np.mean(f1s), "f1_macro_dp": np.std(f1s, ddof=1)}, conf


def main() -> None:
    SAIDA.mkdir(parents=True, exist_ok=True)
    cat = catalogo_motor()
    imagens = dados.carregar_imagens(cat)
    paleta, aud = recuperar_paleta(imagens)
    conv = ConversorPaleta(paleta)
    log(f"{len(cat)} imagens; paleta: {aud['cores_distintas']} cores, {aud['cores_no_caminho']} no caminho, "
        f"distância máxima fora do caminho {aud['distancia_maxima_fora_do_caminho']:.1f}")
    ind = ft.indicadores(imagens, conv)
    mob = ft.mobilenet(imagens)
    X = {"trivial": ft.trivial(imagens), "posicao": ft.posicao(imagens), "indicadores": ind.to_numpy(),
         "mobilenet": mob, "hibrido_mb": np.column_stack([mob, ind.to_numpy()])}
    log("features prontas")

    # 1. Classificação das 11 condições
    linhas, confusoes = [], {}
    for c, Xc in X.items():
        for esq in ("aleatoria", "blocos"):
            r, m = classificar_11(Xc, cat, c, esq)
            linhas.append(r)
            confusoes[(c, esq)] = m
    clf = pd.DataFrame(linhas)
    clf.round(4).to_csv(SAIDA / "classificacao_11_condicoes.csv", index=False)
    log("classificação concluída")

    # 2. Severidade do estator: blocos e severidade não vista
    est = cat[cat.pasta.isin(list(ESTATOR))].reset_index(drop=True)
    idx = cat.index[cat.pasta.isin(list(ESTATOR))].to_numpy()
    prob = av.Problema(tuple(ESTATOR), tuple(alfa_motor(p) for p in ESTATOR), 100.0)
    reg, interp = [], []
    for c, Xc in X.items():
        r = av.validar(Xc[idx], est, ind.iloc[idx].reset_index(drop=True), "blocos", c, problema=prob)["dobras"]
        reg.append({"conjunto": c, "mae_pontos_percentuais": r.reg_mae_espiras.mean(),
                    "mae_dp": r.reg_mae_espiras.std(ddof=1), "f1_estator": r.mul_f1_macro.mean(),
                    "cobertura_90": r.conformal_cobertura.mean(),
                    "meia_largura_pp": r.conformal_meia_largura.mean() * 100})
        it = av.interpolacao(Xc[idx], est, c, problema=prob)
        interp.append(it)
    pd.DataFrame(reg).round(4).to_csv(SAIDA / "severidade_blocos.csv", index=False)
    interp = pd.concat(interp, ignore_index=True).rename(columns={"mae_espiras": "mae_pontos_percentuais"})
    interp.round(4).to_csv(SAIDA / "severidade_nao_vista.csv", index=False)
    resumo = interp.groupby("conjunto", sort=False).agg(
        mae_medio_pp=("mae_pontos_percentuais", "mean"), pior_pp=("mae_pontos_percentuais", "max"),
        cobertura_90=("conformal_cobertura", "mean"), fora_dominio=("fora_dominio_fracao", "mean"))
    resumo.round(3).to_csv(SAIDA / "severidade_nao_vista_resumo.csv")
    log("severidade concluída")

    (SAIDA / "auditoria.json").write_text(json.dumps({
        "n_imagens": len(cat), "paleta": aud, "alfa_por_condicao": {p: alfa_motor(p) for p in ESTATOR},
        "definicao_alfa": "porcentagem por fase x fases / 3 (hipótese: fases com o mesmo número de espiras)",
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    np.save(SAIDA / "confusao_hibrido_mb_blocos.npy", confusoes[("hibrido_mb", "blocos")])
    log("fim")


if __name__ == "__main__":
    main()
