"""Gera as partes do artigo que vêm dos resultados: artigo/numeros.tex (macros com cada
número citado no texto), artigo/tabelas/*.tex e artigo/figuras/*.png.

    .venv/Scripts/python.exe artigo/gerar_artigo.py

Rode depois de main.py, experimentos.py e motor.py. Nenhum número do texto é digitado à mão.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import config  # noqa: E402

ART = RAIZ / "artigo"
S = config.SAIDA
E = S / "experimentos"
M = S / "motor"
NOMES = {
    "trivial": "Miniatura 16$\\times$12 (linha de base)", "posicao": "Posição do objeto (controle)",
    "indicadores_v1": "Indicadores v1 (rascunho)", "indicadores": "Indicadores de paleta",
    "mobilenet": "MobileNetV3", "dinov2": "DINOv2", "hibrido_v1": "DINOv2 + indicadores v1",
    "hibrido": "DINOv2 + indicadores de paleta", "hibrido_mb": "MobileNetV3 + indicadores de paleta",
}
ORDEM = list(NOMES)


def br(v: float, casas: int = 1) -> str:
    """Número com vírgula decimal (padrão do artigo em português)."""
    return f"{v:.{casas}f}".replace(".", ",")


def pct(v: float, casas: int = 0) -> str:
    return br(100 * v, casas) + "\\%"


def macro(nome: str, valor: str) -> str:
    assert nome.isalpha(), nome
    return f"\\newcommand{{\\{nome}}}{{{valor}}}"


def tabela(nome: str, colunas: str, cabecalho: list[str], linhas: list[list[str]], destaque: set[int] = frozenset()) -> None:
    corpo = []
    for i, l in enumerate(linhas):
        celulas = [f"\\textbf{{{c}}}" if i in destaque else c for c in l]
        corpo.append(" & ".join(celulas) + " \\\\")
    tex = "\n".join([f"\\begin{{tabular}}{{{colunas}}}", "\\toprule", " & ".join(cabecalho) + " \\\\", "\\midrule",
                     *corpo, "\\bottomrule", "\\end{tabular}"])
    (ART / "tabelas" / f"{nome}.tex").write_text(tex + "\n", encoding="utf-8")


def main() -> None:
    (ART / "tabelas").mkdir(parents=True, exist_ok=True)
    (ART / "figuras").mkdir(parents=True, exist_ok=True)
    m: dict[str, str] = {}

    # ---------------------------------------------------------------- dados
    aud = json.loads((S / "auditoria_dataset.json").read_text(encoding="utf-8"))
    esc = {e["condicao"]: e for e in aud["escala_por_condicao"]}
    m["NImagens"] = str(aud["n_imagens"])
    m["NCores"] = str(aud["paleta"]["cores_distintas"])
    m["MaxIdxSaudavel"] = br(esc["Healthy"]["max_mean"], 3)
    m["MaxIdxSCoitenta"] = br(esc["SC80"]["max_mean"], 3)
    m["SatSCseiscentos"] = pct(esc["SC600"]["saturado_mean"], 2)
    q = aud["quase_duplicatas"]
    m["VizMesmaClasse"] = pct(q["vizinho_mais_proximo_mesma_classe"], 1)
    m["VizAdjacente"] = pct(q["vizinho_mais_proximo_quadro_adjacente"], 1)

    # ------------------------------------------------------------ reprodução
    rep = pd.read_csv(S / "reproducao_protocolo_original.csv").set_index("conjunto")
    m["FUmOriginal"] = br(rep.loc["hibrido_v1", "mul_f1_macro"], 3)
    m["MAEOriginal"] = br(rep.loc["hibrido_v1", "reg_mae_espiras"], 2)

    # ------------------------------------------------------------ validação
    r = pd.read_csv(S / "resumo_validacao.csv")
    a, b = r[r.esquema == "aleatoria"].set_index("conjunto"), r[r.esquema == "blocos"].set_index("conjunto")
    tabela("validacao", "lrrrr", ["Features", "F1 aleat.", "F1 blocos", "MAE aleat.", "MAE blocos"],
           [[NOMES[c], br(a.loc[c, "mul_f1_macro"], 3), br(b.loc[c, "mul_f1_macro"], 3),
             br(a.loc[c, "reg_mae_espiras"]), br(b.loc[c, "reg_mae_espiras"])] for c in ORDEM],
           destaque={ORDEM.index("hibrido")})
    m["FUmBlocosHib"] = br(b.loc["hibrido", "mul_f1_macro"], 3)
    m["MAEBlocosHib"] = br(b.loc["hibrido", "reg_mae_espiras"])
    m["FUmBlocosTriv"] = br(b.loc["trivial", "mul_f1_macro"], 3)
    m["MAEBlocosTriv"] = br(b.loc["trivial", "reg_mae_espiras"])
    m["FUmBlocosPos"] = br(b.loc["posicao", "mul_f1_macro"], 2)
    m["MAEBlocosDino"] = br(b.loc["dinov2", "reg_mae_espiras"])
    m["MAEBlocosInd"] = br(b.loc["indicadores", "reg_mae_espiras"])
    m["CobBlocosHib"] = br(b.loc["hibrido", "conformal_cobertura"], 3)
    m["MeiaBlocosHib"] = br(b.loc["hibrido", "conformal_meia_largura"] * config.TOTAL_ESPIRAS, 0)
    m["OODBlocosHib"] = pct(b.loc["hibrido", "fora_dominio_fracao"], 1)

    # ------------------------------------------------- severidade não vista
    it = pd.read_csv(S / "interpolacao_severidade.csv")
    g = it.groupby("conjunto", sort=False)
    res = pd.DataFrame({"mae": g.mae_espiras.mean(), "pior": g.mae_espiras.max(),
                        "nivel": g.nivel_mais_proximo_correto.mean(), "cob": g.conformal_cobertura.mean(),
                        "meia": g.conformal_meia_largura.mean() * config.TOTAL_ESPIRAS})
    res["cond_pior"] = [it.loc[it[it.conjunto == c].mae_espiras.idxmax(), "condicao_retirada"] for c in res.index]
    tabela("interpolacao", "lrrrr", ["Features", "MAE médio", "Pior caso", "Nível certo", "Cobertura IC 90\\%"],
           [[NOMES[c], br(res.loc[c, "mae"]), f"{br(res.loc[c, 'pior'])} ({res.loc[c, 'cond_pior']})",
             pct(res.loc[c, "nivel"]), pct(res.loc[c, "cob"])] for c in ORDEM],
           destaque={ORDEM.index("hibrido")})
    for c, n in [("hibrido", "Hib"), ("indicadores", "Ind"), ("dinov2", "Dino"), ("trivial", "Triv"),
                 ("hibrido_mb", "MB"), ("mobilenet", "Mob"), ("hibrido_v1", "HibVum")]:
        m[f"Interp{n}"] = br(res.loc[c, "mae"])
        m[f"Interp{n}Pior"] = br(res.loc[c, "pior"])
    m["InterpHibPiorCond"] = res.loc["hibrido", "cond_pior"]
    m["InterpHibNivel"] = pct(res.loc["hibrido", "nivel"])
    m["InterpHibCob"] = pct(res.loc["hibrido", "cob"])
    m["InterpHibMeia"] = br(res.loc["hibrido", "meia"], 0)
    hib = it[it.conjunto == "hibrido"].set_index("condicao_retirada")
    m["InterpHibSCtrezentosvinte"] = br(hib.loc["SC320", "mae_espiras"])
    tabela("interpolacao_hibrido", "l" + "r" * len(hib), ["Retirada", *hib.index],
           [["MAE (espiras)", *[br(v) for v in hib.mae_espiras]],
            ["$\\bar{\\alpha}$ previsto", *[br(v, 3) for v in hib.alfa_previsto_medio]],
            ["$\\alpha$ real", *[br(v, 3) for v in hib.alfa_real]]])

    # --------------------------------------------------- campanha não vista
    cp = pd.read_csv(S / "generalizacao_por_campanha.csv")
    ci = cp[cp.tipo == "interpolacao"]
    conjs_c = ["trivial", "posicao", "indicadores", "mobilenet", "dinov2", "hibrido", "hibrido_mb"]
    linhas = []
    for c in conjs_c:
        gc = ci[ci.conjunto == c]
        bc = (gc.mae_espiras * gc.n_teste).sum() / gc.n_teste.sum()
        ood = (gc.fora_dominio_fracao * gc.n_teste).sum() / gc.n_teste.sum()
        linhas.append([NOMES[c], br(gc[gc.campanha_retirada == "B"].mae_espiras.mean()),
                       br(gc[gc.campanha_retirada == "C"].mae_espiras.mean()), br(bc), pct(ood)])
        if c in ("hibrido", "trivial", "hibrido_mb", "dinov2"):
            m[{"hibrido": "CampHib", "trivial": "CampTriv", "hibrido_mb": "CampMB", "dinov2": "CampDino"}[c]] = br(bc)
            m[{"hibrido": "CampHibOOD", "trivial": "CampTrivOOD", "hibrido_mb": "CampMBOOD", "dinov2": "CampDinoOOD"}[c]] = pct(ood)
    tabela("campanha", "lrrrr", ["Features", "Camp. B", "Camp. C", "B + C", "Fora do domínio"], linhas,
           destaque={conjs_c.index("hibrido")})
    ext = cp[(cp.tipo == "extrapolacao") & (cp.conjunto == "hibrido")].set_index("condicao")
    m["CampAAlfa"] = br(ext.loc["Healthy", "alfa_previsto_medio"], 2)

    # ------------------------------------------------------- experimentos
    zoo = pd.read_csv(E / "zoo_resumo.csv").set_index("conjunto")
    anin = pd.read_csv(E / "selecao_aninhada.csv")
    m["AninhadaMAE"] = br(anin.mae_espiras.mean())
    m["AninhadaEscolhas"] = str(anin.escolhido.nunique())
    m["ZooN"] = str(len(zoo))
    m["ZooMelhor"] = br(zoo.mae_interp.min())
    top = zoo.sort_values("mae_interp").head(6)
    reg_nome = {"ridge": "Ridge", "pls": "PLS", "pca_ridge": "PCA+Ridge", "kernel_ridge": "Kernel Ridge", "svr": "SVR",
                "gp": "Proc. gaussiano", "media_dino_mb": "média DINOv2 e MobileNet", "media_3": "média de 3 modelos"}
    feat_nome = {**NOMES, "ind_invariantes": "Indicadores sem cena", "hibrido_inv": "DINOv2 + indicadores sem cena",
                 "empilhado": "Média de modelos"}
    tabela("zoo", "llrrr", ["Features", "Regressor", "Sev. nova", "Pior", "Camp. nova"],
           [[feat_nome.get(k.split("|")[0], k), reg_nome.get(k.split("|")[1], k), br(v.mae_interp), br(v.pior_interp),
             br(v.mae_campanha_BC)] for k, v in top.iterrows()])
    kr = zoo[zoo.index.str.endswith(("kernel_ridge", "svr"))]
    m["ZooKernelMin"] = br(kr.mae_interp.min(), 0)
    pares = pd.read_csv(E / "testes_pareados_interpolacao.csv")
    def p_de(a_, b_):
        l = pares[(pares.a == a_) & (pares.b == b_)].iloc[0]
        return l
    l = p_de("hibrido|ridge", "dinov2|ridge"); m["PHibDino"] = br(l.p_wilcoxon, 3); m["NHibDino"] = str(int(l.niveis_a_melhor))
    l = p_de("hibrido|ridge", "indicadores|ridge"); m["PHibInd"] = br(l.p_wilcoxon, 3); m["NHibInd"] = str(int(l.niveis_a_melhor))
    l = p_de("hibrido_mb|ridge", "hibrido|ridge"); m["PMBHib"] = br(l.p_wilcoxon, 2)
    ic = pd.read_csv(E / "ic_bootstrap_interpolacao.csv").set_index("conjunto")
    m["ICHibInf"], m["ICHibSup"] = br(ic.loc["hibrido|ridge", "ic95_inf"]), br(ic.loc["hibrido|ridge", "ic95_sup"])

    rob = pd.read_csv(E / "robustez_filtro.csv").set_index(["conjunto", "perturbacao"])
    m["RobIndSemFiltroRuidoFUm"] = br(rob.loc[("indicadores_sem_filtro", "ruido"), "mul_f1_macro"], 2)
    m["RobIndRuidoFUm"] = br(rob.loc[("indicadores", "ruido"), "mul_f1_macro"], 2)
    m["RobIndSemFiltroTransl"] = br(rob.loc[("indicadores_sem_filtro", "translacao"), "reg_mae_espiras"])
    m["RobIndTransl"] = br(rob.loc[("indicadores", "translacao"), "reg_mae_espiras"])
    aum = pd.read_csv(E / "aumento_resumo.csv").set_index(["conjunto", "treino", "perturbacao"])
    gen = pd.read_csv(E / "aumento_generalizacao_resumo.csv").set_index(["conjunto", "treino", "teste"])
    perts = ["nenhuma", "ruido", "desfoque", "translacao", "rotacao"]
    rot = {"nenhuma": "Limpa", "ruido": "Ruído", "desfoque": "Desfoque", "translacao": "Transl.", "rotacao": "Rotação"}
    linhas = []
    for c, nome in [("mobilenet", "MobileNetV3"), ("hibrido_mb_f", "MobileNetV3 + ind.")]:
        for t, tn in [("limpo", "sem aumento"), ("aumentado", "com aumento")]:
            linhas.append([f"{nome}, {tn}", *[br(aum.loc[(c, t, p), "reg_mae_espiras"]) for p in perts],
                           br(gen.loc[(c, t, "severidade"), "mae_espiras"]), br(gen.loc[(c, t, "campanha"), "mae_espiras"])])
    tabela("aumento", "l" + "r" * 7, ["Modelo", *[rot[p] for p in perts], "Sev. nova", "Camp. nova"], linhas, destaque={1})
    m["AumMBSevAntes"], m["AumMBSevDepois"] = br(gen.loc[("mobilenet", "limpo", "severidade"), "mae_espiras"]), br(gen.loc[("mobilenet", "aumentado", "severidade"), "mae_espiras"])
    m["AumMBCampAntes"], m["AumMBCampDepois"] = br(gen.loc[("mobilenet", "limpo", "campanha"), "mae_espiras"]), br(gen.loc[("mobilenet", "aumentado", "campanha"), "mae_espiras"])
    m["AumMBRuidoAntes"], m["AumMBRuidoDepois"] = br(aum.loc[("mobilenet", "limpo", "ruido"), "reg_mae_espiras"]), br(aum.loc[("mobilenet", "aumentado", "ruido"), "reg_mae_espiras"])
    m["AumHibSevDepois"] = br(gen.loc[("hibrido_mb_f", "aumentado", "severidade"), "mae_espiras"])

    der = pd.read_csv(E / "deriva_temporal.csv")
    m["DerivaMax"] = br(der.variacao_espiras_na_gravacao.abs().max(), 0)
    m["DerivaRhoMax"] = br(der.spearman_indice.abs().max(), 2)
    sen = pd.read_csv(E / "sensibilidade.csv")
    bp = sen[sen.variavel == "blocos_purga"]
    m["SensMAEMin"], m["SensMAEMax"] = br(bp.mae_blocos.min()), br(bp.mae_blocos.max())
    m["SensFUmMin"] = br(bp.f1_blocos.min(), 3)

    ab = pd.read_csv(E / "abstencao_resumo.csv").set_index(["conjunto", "teste"])
    m["AbstCampAceitas"] = br(ab.loc[("hibrido", "campanha"), "mae_aceitas"], 0)
    m["AbstCampRejeitadas"] = br(ab.loc[("hibrido", "campanha"), "mae_rejeitadas"], 0)
    m["AbstCampFracRejeitada"] = pct(1 - ab.loc[("hibrido", "campanha"), "fracao_aceita"])
    m["AbstCampRho"] = br(ab.loc[("hibrido", "campanha"), "spearman_score_erro"], 2)
    m["AbstSevAceitas"] = br(ab.loc[("hibrido", "severidade"), "mae_aceitas"])
    m["AbstSevRejeitadas"] = br(ab.loc[("hibrido", "severidade"), "mae_rejeitadas"])

    ocl = json.loads((S / "oclusao_resumo.json").read_text(encoding="utf-8"))
    for mod, nome in (("mobilenet", "Mob"), ("indicadores", "Ind")):
        v = [d[mod]["fracao_da_queda_dentro_da_roi"] for d in ocl.values()]
        m[f"Ocl{nome}Min"], m[f"Ocl{nome}Max"] = pct(min(v)), pct(max(v))

    # --------------------------------------------------------------- motor
    mc = pd.read_csv(M / "classificacao_11_condicoes.csv").set_index(["conjunto", "esquema"])
    ms = pd.read_csv(M / "severidade_blocos.csv").set_index("conjunto")
    mn = pd.read_csv(M / "severidade_nao_vista_resumo.csv").set_index("conjunto")
    nm = {"trivial": NOMES["trivial"], "posicao": NOMES["posicao"], "indicadores": NOMES["indicadores"],
          "mobilenet": "MobileNetV3", "hibrido_mb": NOMES["hibrido_mb"]}
    tabela("motor", "lrrrr", ["Features", "F1 (11 cond.)", "MAE blocos", "MAE sev. nova", "Pior"],
           [[nm[c], br(mc.loc[(c, "blocos"), "f1_macro"], 3), br(ms.loc[c, "mae_pontos_percentuais"], 2),
             br(mn.loc[c, "mae_medio_pp"], 1), br(mn.loc[c, "pior_pp"], 1)] for c in nm])
    m["MotorFUm"] = br(mc.loc[("hibrido_mb", "blocos"), "f1_macro"], 3)
    m["MotorFUmTriv"] = br(mc.loc[("trivial", "blocos"), "f1_macro"], 3)
    m["MotorFUmPos"] = br(mc.loc[("posicao", "blocos"), "f1_macro"], 2)
    m["MotorMAE"] = br(ms.loc["hibrido_mb", "mae_pontos_percentuais"], 2)
    m["MotorSevMob"] = br(mn.loc["mobilenet", "mae_medio_pp"], 1)
    m["MotorSevPos"] = br(mn.loc["posicao", "mae_medio_pp"], 1)
    m["MotorCores"] = str(json.loads((M / "auditoria.json").read_text(encoding="utf-8"))["paleta"]["cores_distintas"])

    # ---------------------------------------------------------- relatórios
    sv = pd.read_csv(S / "sensibilidade_verificador.csv").set_index("mutacao")
    nomes_mut = {"temperatura_inventada": "Temperatura inventada (``87\\,\\textdegree C'')", "corrente_inventada": "Corrente inventada (``3,2\\,A'')",
                 "espiras_recalculadas": "Espiras recalculadas", "recomendacao_trocada": "Recomendação trocada",
                 "limitacoes_omitidas": "Limitações omitidas", "numero_alterado": "$\\alpha$ alterado",
                 "vida_util_afirmada": "``Reduz a vida útil'' (alerta)"}
    tabela("verificador", "lr", ["Erro injetado (255 relatórios cada)", "Detectado"],
           [[nomes_mut[k], pct(sv.loc[k, "com_alerta" if k == "vida_util_afirmada" else "reprovados"], 1)] for k in nomes_mut])
    rep_ = sv.drop("vida_util_afirmada").reprovados
    m["VerifMin"], m["VerifMax"] = pct(rep_.min(), 1), pct(rep_.max(), 0)
    ver = pd.read_csv(S / "verificacao_relatorios.csv")
    m["RelAprovados"] = str(int(ver.aprovado.sum()))

    # ---------------------------------------------------------- aplicação
    meta_app = config.RAIZ / "modelos" / "diagnosticador.json"
    if meta_app.exists():
        meta = json.loads(meta_app.read_text(encoding="utf-8"))
        m["AppMeia"] = br(meta["q_conformal"] * config.TOTAL_ESPIRAS, 0)
    else:
        m["AppMeia"] = "\\todo{rode diagnosticar.py --treinar}"

    (ART / "numeros.tex").write_text("% Gerado por gerar_artigo.py a partir de saida/. Não edite à mão.\n"
                                     + "\n".join(macro(k, v) for k, v in m.items()) + "\n", encoding="utf-8")
    for f in ["fig_paleta_saturacao.png", "fig_indicadores_severidade.png", "fig_esquemas_validacao.png",
              "fig_interpolacao.png", "fig_campanhas.png", "fig_oclusao.png", "fig_segmentacao.png"]:
        shutil.copy2(S / "figuras" / f, ART / "figuras" / f)
    zipar_overleaf()
    print(f"{len(m)} números, {len(list((ART / 'tabelas').glob('*.tex')))} tabelas, figuras copiadas para {ART}")


def zipar_overleaf() -> None:
    """artigo_overleaf.zip: no Overleaf, New Project > Upload Project e escolha este arquivo."""
    import zipfile
    arquivos = [ART / "artigo.tex", ART / "numeros.tex", ART / "referencias.bib",
                *sorted((ART / "tabelas").glob("*.tex")), *sorted((ART / "figuras").glob("*.png"))]
    with zipfile.ZipFile(ART / "artigo_overleaf.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in arquivos:
            z.write(f, f.relative_to(ART).as_posix())


if __name__ == "__main__":
    main()
