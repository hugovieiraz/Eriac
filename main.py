"""Pipeline completo. Execute a partir da pasta do projeto:

    .venv/Scripts/python.exe main.py
    .venv/Scripts/python.exe main.py --llm claude --n-llm 20   (requer credencial da Anthropic)

Tudo o que o artigo cita sai em saida/ (tabelas CSV/JSON, figuras, relatórios).
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time

import numpy as np
import pandas as pd

import avaliacao as av
import config
import dados
import features as ft
import figuras as fg
import relatorio_llm as rl
from paleta import ConversorPaleta, recuperar_paleta

CONJUNTOS = ["trivial", "posicao", "indicadores_v1", "indicadores", "mobilenet", "dinov2", "hibrido_v1", "hibrido",
             "hibrido_mb"]
PRINCIPAL = "hibrido"
ROBUSTEZ = ["trivial", "indicadores", "mobilenet", "hibrido_mb"]


def log(msg: str, t0=[time.perf_counter()]) -> None:
    print(f"[{time.perf_counter() - t0[0]:7.1f}s] {msg}", flush=True)


def salvar_json(obj, nome: str) -> None:
    (config.SAIDA / nome).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=float), encoding="utf-8")


# ------------------------------------------------------------- auditorias ----
def auditar_dataset(catalogo, imagens, conversor, dinov2) -> dict:
    indices = np.stack([conversor.indice(img) for img in imagens])
    por_imagem = pd.DataFrame({"condicao": catalogo.condicao, "min": indices.reshape(len(imagens), -1).min(1),
                               "max": indices.reshape(len(imagens), -1).max(1),
                               "saturado": (indices >= ft.LIMIAR_SATURADO).reshape(len(imagens), -1).mean(1)})
    escala = por_imagem.groupby("condicao", sort=False).agg(["mean", "min", "max"]).round(4)
    escala.columns = ["_".join(c) for c in escala.columns]

    # Quase-duplicatas: similaridade de cosseno entre embeddings DINOv2.
    z = dinov2 / np.linalg.norm(dinov2, axis=1, keepdims=True)
    sim = z @ z.T
    np.fill_diagonal(sim, -np.inf)
    vizinho = sim.argmax(axis=1)
    mesma_classe = catalogo.nivel.to_numpy()[vizinho] == catalogo.nivel.to_numpy()
    distancia_ordem = np.abs(catalogo.ordem.to_numpy()[vizinho] - catalogo.ordem.to_numpy())
    outra = sim.copy()
    outra[catalogo.nivel.to_numpy()[:, None] == catalogo.nivel.to_numpy()[None, :]] = -np.inf
    adjacentes = [sim[i, j] for i in range(len(catalogo)) for j in range(len(catalogo))
                  if catalogo.pasta[i] == catalogo.pasta[j] and catalogo.ordem[j] == catalogo.ordem[i] + 1]
    return {
        "n_imagens": int(len(catalogo)),
        "hashes_duplicados": int(catalogo.sha256.duplicated().sum()),
        "campanhas": catalogo.groupby("campanha").pasta.unique().apply(list).to_dict(),
        "imagens_por_condicao": catalogo.condicao.value_counts(sort=False).to_dict(),
        "escala_por_condicao": escala.reset_index().to_dict(orient="records"),
        "quase_duplicatas": {
            "vizinho_mais_proximo_mesma_classe": float(mesma_classe.mean()),
            "vizinho_mais_proximo_quadro_adjacente": float(np.mean(mesma_classe & (distancia_ordem == 1))),
            "vizinho_mais_proximo_ate_3_quadros": float(np.mean(mesma_classe & (distancia_ordem <= 3))),
            "cosseno_quadro_adjacente_mediana": float(np.median(adjacentes)),
            "cosseno_outra_classe_mais_proxima_mediana": float(np.median(outra.max(axis=1))),
        },
    }


# ---------------------------------------------------------------- oclusão ----
def mapa_oclusao(img, modelo, extrair, tamanho=40, passo=20) -> np.ndarray:
    """Queda do alfa previsto quando um quadrado é pintado com a cor mediana do fundo.
    Positivo = a região empurrava a previsão para cima."""
    h, w = img.shape[:2]
    k = max(8, min(h, w) // 12)
    cantos = np.concatenate((img[:k, :k], img[:k, -k:], img[-k:, :k], img[-k:, -k:]), axis=0).reshape(-1, 3)
    fundo = np.median(cantos, axis=0).astype(np.uint8)
    base = float(modelo.predict(extrair(img[None]))[0])
    posicoes = [(y, x) for y in range(0, h - tamanho + 1, passo) for x in range(0, w - tamanho + 1, passo)]
    ocultas = []
    for y, x in posicoes:
        o = img.copy()
        o[y:y + tamanho, x:x + tamanho] = fundo
        ocultas.append(o)
    queda = base - modelo.predict(extrair(np.stack(ocultas)))
    soma, conta = np.zeros((h, w)), np.zeros((h, w))
    for (y, x), q in zip(posicoes, queda):
        soma[y:y + tamanho, x:x + tamanho] += q
        conta[y:y + tamanho, x:x + tamanho] += 1
    return soma / np.maximum(conta, 1)


# ------------------------------------------------------------------ main ----
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", choices=["deterministico", "claude"], default="deterministico")
    parser.add_argument("--n-llm", type=int, default=12, help="imagens enviadas ao Claude (--llm claude)")
    parser.add_argument("--sem-robustez", action="store_true")
    args = parser.parse_args()

    random.seed(config.SEMENTE)
    np.random.seed(config.SEMENTE)
    config.FIGURAS.mkdir(parents=True, exist_ok=True)
    (config.SAIDA / "relatorios").mkdir(exist_ok=True)

    # 1. Dados, paleta e auditoria --------------------------------------------
    catalogo = dados.construir_catalogo()
    if catalogo.drop_duplicates("pasta").set_index("pasta").campanha.to_dict() != config.CAMPANHAS_ESPERADAS:
        raise RuntimeError("Campanhas inferidas diferentes das esperadas.")
    catalogo.drop(columns="caminho").to_csv(config.SAIDA / "catalogo.csv", index=False)
    imagens = dados.carregar_imagens(catalogo)
    paleta, auditoria_paleta = recuperar_paleta(imagens)
    conversor = ConversorPaleta(paleta)
    np.save(config.SAIDA / "paleta_recuperada.npy", paleta)
    log(f"{len(catalogo)} imagens; paleta com {len(paleta)} cores")

    # 2. Features --------------------------------------------------------------
    ind = ft.indicadores(imagens, conversor)
    ind.assign(arquivo=catalogo.arquivo, condicao=catalogo.condicao).to_csv(config.SAIDA / "indicadores_paleta.csv", index=False)
    ind_v1 = ft.indicadores_v1_original(catalogo)
    dinov2, fonte_dinov2 = ft.dinov2(catalogo)
    log(f"DINOv2: {fonte_dinov2}")
    mobilenet = ft.mobilenet(imagens)
    X = {
        "trivial": ft.trivial(imagens),
        "posicao": ft.posicao(imagens),
        "indicadores_v1": ind_v1.to_numpy(),
        "indicadores": ind.to_numpy(),
        "mobilenet": mobilenet,
        "dinov2": dinov2,
        "hibrido_v1": np.column_stack([dinov2, ind_v1.to_numpy()]),
        "hibrido": np.column_stack([dinov2, ind.to_numpy()]),
        "hibrido_mb": np.column_stack([mobilenet, ind.to_numpy()]),
    }
    salvar_json({k: int(v.shape[1]) for k, v in X.items()}, "dimensoes_features.json")
    log("features: " + ", ".join(f"{k}={v.shape[1]}" for k, v in X.items()))

    auditoria = auditar_dataset(catalogo, imagens, conversor, dinov2)
    auditoria["paleta"] = auditoria_paleta
    salvar_json(auditoria, "auditoria_dataset.json")

    perturbados = {}
    robustez_conjuntos = list(ROBUSTEZ)
    if not args.sem_robustez:
        for tipo in ft.PERTURBACOES:
            imgs_p = ft.perturbar(imagens, tipo)
            ind_p = ft.indicadores(imgs_p, conversor).to_numpy()
            mob_p = ft.mobilenet(imgs_p)
            perturbados[tipo] = {"trivial": ft.trivial(imgs_p), "indicadores": ind_p, "mobilenet": mob_p,
                                 "hibrido_mb": np.column_stack([mob_p, ind_p])}
            d_p = ft.dinov2_perturbado(catalogo, tipo)
            if d_p is not None:
                perturbados[tipo].update({"dinov2": d_p, "hibrido": np.column_stack([d_p, ind_p])})
        if all("dinov2" in p for p in perturbados.values()):
            robustez_conjuntos += ["dinov2", "hibrido"]
        log(f"features das imagens perturbadas prontas ({', '.join(robustez_conjuntos)})")

    # 3. Reprodução do protocolo original --------------------------------------
    reproducao = []
    for c in ["dinov2", "hibrido_v1"]:
        r = av.validar(X[c], catalogo, ind, "original", c)
        reproducao.append(r["dobras"])
    reproducao = av.resumir(pd.concat(reproducao))
    original = pd.read_csv(config.REFERENCIA / "resumo_metricas_validacao_cruzada.csv")
    alvo = original[original.metric.isin(["multiclass_f1_macro", "regression_mae_alpha", "binary_accuracy"])]
    reproducao.to_csv(config.SAIDA / "reproducao_protocolo_original.csv", index=False)
    alvo.to_csv(config.SAIDA / "referencia_rascunho_original.csv", index=False)
    log("reprodução do protocolo original concluída")

    # 4. Validação cruzada: aleatória x blocos --------------------------------
    dobras, oof_principal, robustez, confusoes = [], None, [], {}
    for c in CONJUNTOS:
        for esquema in ("aleatoria", "blocos"):
            extras = {t: perturbados[t][c] for t in perturbados} if (esquema == "blocos" and c in robustez_conjuntos) else None
            r = av.validar(X[c], catalogo, ind, esquema, c, extras)
            dobras.append(r["dobras"])
            confusoes[(c, esquema)] = r["confusao"]
            if esquema == "blocos":
                robustez.append(r["robustez"])
                if c == PRINCIPAL:
                    oof_principal = r["oof"]
        log(f"validação {c} concluída")
    dobras = pd.concat(dobras, ignore_index=True)
    dobras.to_csv(config.SAIDA / "metricas_por_dobra.csv", index=False)
    resumo = av.resumir(dobras)
    resumo.to_csv(config.SAIDA / "resumo_validacao.csv", index=False)
    blocos = dobras[dobras.esquema == "blocos"]
    pares = [("hibrido", "dinov2"), ("hibrido", "indicadores"), ("dinov2", "trivial"), ("indicadores", "trivial"),
             ("hibrido", "hibrido_v1"), ("dinov2", "mobilenet"), ("indicadores", "indicadores_v1"),
             ("hibrido_mb", "hibrido")]
    comparacoes = [av.comparacao_pareada(blocos, a, b, "mul_f1_macro", True) for a, b in pares] + \
                  [av.comparacao_pareada(blocos, a, b, "reg_mae_espiras", False) for a, b in pares]
    pd.DataFrame(comparacoes).to_csv(config.SAIDA / "comparacao_pareada_blocos.csv", index=False)
    if robustez and not all(r.empty for r in robustez):
        rob = pd.concat(robustez, ignore_index=True)
        limpo = blocos[blocos.conjunto.isin(robustez_conjuntos)].assign(perturbacao="nenhuma")[
            ["conjunto", "perturbacao", "dobra", "bin_acuracia", "mul_f1_macro", "reg_mae_espiras", "fora_dominio_fracao"]]
        rob = pd.concat([limpo, rob], ignore_index=True)
        rob.groupby(["conjunto", "perturbacao"], sort=False).mean(numeric_only=True).round(4).to_csv(
            config.SAIDA / "robustez_perturbacoes.csv")

    # 5. Generalização: severidade e campanha não vistas ----------------------
    interp = pd.concat([av.interpolacao(X[c], catalogo, c) for c in CONJUNTOS], ignore_index=True)
    interp.to_csv(config.SAIDA / "interpolacao_severidade.csv", index=False)
    resumo_interp = interp.groupby("conjunto", sort=False).agg(
        mae_espiras_medio=("mae_espiras", "mean"), mae_espiras_pior=("mae_espiras", "max"),
        pior_condicao=("mae_espiras", lambda s: interp.loc[s.idxmax(), "condicao_retirada"]),
        nivel_correto_medio=("nivel_mais_proximo_correto", "mean"),
        fora_dominio_medio=("fora_dominio_fracao", "mean"))
    resumo_interp.to_csv(config.SAIDA / "interpolacao_resumo.csv")
    camp = pd.concat([av.por_campanha(X[c], catalogo, c) for c in CONJUNTOS], ignore_index=True)
    camp.to_csv(config.SAIDA / "generalizacao_por_campanha.csv", index=False)
    log("interpolação e campanhas concluídas")

    # 6. Registros estruturados, relatórios e verificador --------------------
    oof_principal = oof_principal.sort_values("indice").reset_index(drop=True)
    oof_principal.merge(catalogo[["arquivo", "condicao", "alfa", "campanha"]], left_on="indice", right_index=True).to_csv(
        config.SAIDA / "previsoes_oof_principal.csv", index=False)
    registros, verificacoes, sensibilidade = [], [], []
    for o in oof_principal.itertuples(index=False):
        r = rl.registro_diagnostico(catalogo.loc[o.indice], o, ind.loc[o.indice])
        texto = rl.gerar_deterministico(r)
        v = rl.verificar(texto, r)
        registros.append(r)
        verificacoes.append({"arquivo": r["identificacao"]["arquivo"], "backend": "deterministico",
                             "aprovado": v["aprovado"], "falhas": len(v["falhas"]), "alertas": len(v["alertas"]),
                             "recomendacao": r["recomendacao"]["codigo"]})
        for tipo, mutado in rl.mutacoes(texto, r).items():
            vm = rl.verificar(mutado, r)
            sensibilidade.append({"mutacao": tipo, "reprovado": not vm["aprovado"], "alerta": bool(vm["alertas"])})
    salvar_json(registros, "registros_estruturados.json")
    pd.DataFrame(verificacoes).to_csv(config.SAIDA / "verificacao_relatorios.csv", index=False)
    sens = pd.DataFrame(sensibilidade).groupby("mutacao").agg(n=("reprovado", "size"), reprovados=("reprovado", "mean"),
                                                             com_alerta=("alerta", "mean"))
    sens.to_csv(config.SAIDA / "sensibilidade_verificador.csv")

    exemplo = next(r for r in registros if r["identificacao"]["arquivo"] == "p5042.bmp")
    (config.SAIDA / "relatorios" / "exemplo_p5042_registro.json").write_text(json.dumps(exemplo, ensure_ascii=False, indent=2), encoding="utf-8")
    (config.SAIDA / "relatorios" / "exemplo_p5042_deterministico.md").write_text(rl.gerar_deterministico(exemplo), encoding="utf-8")
    (config.SAIDA / "relatorios" / "prompt_sistema.md").write_text(
        f"<!-- versão {rl.VERSAO_PROMPT}; modelo {rl.MODELO_CLAUDE} -->\n\n{rl.PROMPT_SISTEMA}\n", encoding="utf-8")

    if args.llm == "claude":
        rng = np.random.default_rng(config.SEMENTE)
        amostra = rng.choice(len(registros), size=min(args.n_llm, len(registros)), replace=False)
        resultados = []
        for k, i in enumerate(amostra, start=1):
            r = registros[i]
            texto, meta = rl.gerar_claude_seguro(r)
            v = rl.verificar(texto, r)
            log(f"Claude {k}/{len(amostra)} {r['identificacao']['arquivo']}: "
                f"{'aprovado' if v['aprovado'] else 'reprovado'}{' (' + meta['erro'] + ')' if 'erro' in meta else ''}")
            nome = r["identificacao"]["arquivo"].replace(".bmp", "")
            (config.SAIDA / "relatorios" / f"claude_{nome}.md").write_text(texto, encoding="utf-8")
            resultados.append({**meta, "arquivo": r["identificacao"]["arquivo"], "aprovado": v["aprovado"],
                               "falhas": " | ".join(v["falhas"]), "alertas": " | ".join(v["alertas"])})
        pd.DataFrame(resultados).to_csv(config.SAIDA / "verificacao_relatorios_claude.csv", index=False)
    log("relatórios e verificação concluídos")

    # 7. Oclusão ---------------------------------------------------------------
    alfa = catalogo.alfa.to_numpy()
    modelos_oclusao = {
        "mobilenet": (av.ajustar("regressao", X["mobilenet"], alfa, av.divisoes_internas_por_grupo(catalogo.bloco.to_numpy()))[0],
                      ft.mobilenet),
        "indicadores": (av.ajustar("regressao", X["indicadores"], alfa, av.divisoes_internas_por_grupo(catalogo.bloco.to_numpy()))[0],
                        lambda imgs: ft.indicadores(imgs, conversor).to_numpy()),
    }
    escolhidas = [int(catalogo.index[catalogo.nivel == n][len(catalogo.index[catalogo.nivel == n]) // 2]) for n in (0, 4, 8)]
    mapas = {i: {m: mapa_oclusao(imagens[i], mod, ext) for m, (mod, ext) in modelos_oclusao.items()} for i in escolhidas}
    mascaras = {i: ft.mascara_transformador(imagens[i]) for i in escolhidas}
    salvar_json({catalogo.arquivo[i]: {m: {"fracao_da_queda_dentro_da_roi": float(np.clip(mapas[i][m], 0, None)[mascaras[i]].sum()
                                                                                   / max(np.clip(mapas[i][m], 0, None).sum(), 1e-12))}
                                       for m in modelos_oclusao} for i in escolhidas}, "oclusao_resumo.json")
    log("oclusão concluída")

    # 8. Figuras ---------------------------------------------------------------
    fg.paleta_e_saturacao(paleta, ind, catalogo)
    fg.segmentacao(imagens, catalogo, conversor, ft.indicadores_imagem)
    fg.indicadores_por_severidade(ind, catalogo)
    fg.esquemas(resumo, CONJUNTOS)
    fg.interpolacao(interp, ["trivial", "indicadores", "dinov2", "hibrido"])
    fg.campanhas(camp, ["trivial", "indicadores", "dinov2", "hibrido"])
    fg.confusao(confusoes[(PRINCIPAL, "blocos")], "DINOv2 + indicadores, blocos temporais", "fig_confusao_principal_blocos.png")
    fg.confusao(confusoes[("hibrido_v1", "aleatoria")], "Método original, divisão aleatória", "fig_confusao_original_aleatoria.png")
    fg.intervalos(oof_principal, catalogo)
    fg.oclusao(imagens, mapas, catalogo)

    import sklearn
    import torch
    salvar_json({"python": sys.version, "plataforma": platform.platform(), "numpy": np.__version__,
                 "pandas": pd.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__,
                 "semente": config.SEMENTE, "fonte_dinov2": fonte_dinov2}, "versoes.json")
    log(f"concluído. Resultados em {config.SAIDA}")


if __name__ == "__main__":
    main()
