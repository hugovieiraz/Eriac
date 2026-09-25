"""Gera painel/index.html a partir de saida/. Rode depois de main.py:

    .venv/Scripts/python.exe gerar_painel.py

O painel é um único HTML com os dados embutidos: abre direto no navegador, sem servidor e
sem internet. Os termogramas não são embutidos (dataset de terceiros); o painel os carrega
de ../dados/IR_trans_bmp/ quando o dataset foi baixado conforme o README.
"""
from __future__ import annotations

import json
from datetime import date

import pandas as pd

import config
import relatorio_llm as rl

MODELO = config.RAIZ / "painel" / "modelo.html"
DESTINO = config.RAIZ / "painel" / "index.html"


def _registros(df: pd.DataFrame, colunas: list[str], casas: int = 4) -> list[dict]:
    return [{c: (round(v, casas) if isinstance(v, float) else v) for c, v in zip(colunas, linha)}
            for linha in df[colunas].itertuples(index=False)]


def _csv(caminho):
    return pd.read_csv(caminho) if caminho.exists() else None


def dados_opcionais(s) -> dict:
    """Seções que dependem de experimentos.py e motor.py; ficam ocultas se faltarem."""
    e, m = s / "experimentos", s / "motor"
    saida = {}
    zoo = _csv(e / "zoo_resumo.csv")
    if zoo is not None:
        saida["zoo"] = _registros(zoo, list(zoo.columns))
        anin = _csv(e / "selecao_aninhada.csv")
        saida["aninhada"] = _registros(anin, ["condicao_retirada", "escolhido", "mae_espiras"]) if anin is not None else None
    aum, gen = _csv(e / "aumento_resumo.csv"), _csv(e / "aumento_generalizacao_resumo.csv")
    if aum is not None:
        linhas = _registros(aum, ["conjunto", "treino", "perturbacao", "reg_mae_espiras"])
        linhas = [{**r, "mae_espiras": r.pop("reg_mae_espiras")} for r in linhas]
        if gen is not None:
            linhas += [{"conjunto": r["conjunto"], "treino": r["treino"], "perturbacao": r["teste"], "mae_espiras": r["mae_espiras"]}
                       for r in _registros(gen, ["conjunto", "treino", "teste", "mae_espiras"])]
        saida["aumento"] = linhas
    der = _csv(e / "deriva_temporal.csv")
    if der is not None:
        saida["deriva"] = _registros(der, list(der.columns))
    clf, blo, nv = _csv(m / "classificacao_11_condicoes.csv"), _csv(m / "severidade_blocos.csv"), _csv(m / "severidade_nao_vista_resumo.csv")
    if clf is not None and blo is not None and nv is not None:
        saida["motor"] = {"classificacao": _registros(clf, list(clf.columns)), "blocos": _registros(blo, list(blo.columns)),
                          "nao_vista": _registros(nv, list(nv.columns))}
    return saida


def main() -> None:
    s = config.SAIDA
    resumo = pd.read_csv(s / "resumo_validacao.csv")
    interp = pd.read_csv(s / "interpolacao_severidade.csv")
    camp = pd.read_csv(s / "generalizacao_por_campanha.csv")
    rob = pd.read_csv(s / "robustez_perturbacoes.csv")
    sens = pd.read_csv(s / "sensibilidade_verificador.csv")
    oof = pd.read_csv(s / "previsoes_oof_principal.csv")
    catalogo = pd.read_csv(s / "catalogo.csv")
    registros = json.loads((s / "registros_estruturados.json").read_text(encoding="utf-8"))
    versoes = json.loads((s / "versoes.json").read_text(encoding="utf-8"))

    claude = {}
    arq_claude = s / "verificacao_relatorios_claude.csv"
    if arq_claude.exists():
        for linha in pd.read_csv(arq_claude).itertuples(index=False):
            texto = (s / "relatorios" / f"claude_{linha.arquivo.replace('.bmp', '')}.md").read_text(encoding="utf-8")
            claude[linha.arquivo] = {"texto": texto, "aprovado": bool(linha.aprovado),
                                     "falhas": str(linha.falhas) if pd.notna(linha.falhas) else "",
                                     "modelo": linha.modelo_que_respondeu}

    por_arquivo = {r["identificacao"]["arquivo"]: r for r in registros}
    pastas = catalogo.set_index("arquivo").pasta.to_dict()
    imagens = []
    for o in oof.itertuples(index=False):
        r = por_arquivo[o.arquivo]
        texto = rl.gerar_deterministico(r)
        v = rl.verificar(texto, r)
        imagens.append({
            "arquivo": o.arquivo, "pasta": pastas[o.arquivo], "condicao": o.condicao, "campanha": o.campanha,
            "alfa": round(o.alfa, 4), "prev": round(o.alfa_previsto, 4), "inf": round(o.alfa_inferior, 4),
            "sup": round(o.alfa_superior, 4), "prob": round(o.prob_defeito, 4),
            "conf": round(o.confianca_classificador, 4), "dominio": round(o.score_dominio, 3),
            "coerencia": round(o.coerencia_visual, 3), "rec": r["recomendacao"]["codigo"],
            "relatorio": texto, "aprovado": v["aprovado"], "falhas": v["falhas"],
            "claude": claude.get(o.arquivo),
        })

    dados = {
        "gerado_em": date.today().isoformat(),
        "versoes": versoes,
        "classes": config.ORDEM_CLASSES,
        "resumo": _registros(resumo, ["conjunto", "esquema", "mul_f1_macro", "mul_f1_macro_dp",
                                      "reg_mae_espiras", "reg_mae_espiras_dp", "bin_acuracia"]),
        "interpolacao": _registros(interp, ["conjunto", "condicao_retirada", "alfa_real", "alfa_previsto_medio",
                                            "alfa_previsto_dp", "mae_espiras", "nivel_mais_proximo_correto",
                                            "fora_dominio_fracao"]),
        "campanha": _registros(camp, ["conjunto", "campanha_retirada", "tipo", "condicao", "alfa_real", "n_teste",
                                      "alfa_previsto_medio", "mae_espiras", "fora_dominio_fracao"]),
        "robustez": _registros(rob, ["conjunto", "perturbacao", "mul_f1_macro", "reg_mae_espiras", "fora_dominio_fracao"]),
        "verificador": _registros(sens, ["mutacao", "n", "reprovados", "com_alerta"]),
        "imagens": imagens,
        **dados_opcionais(s),
    }
    json_dados = json.dumps(dados, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = MODELO.read_text(encoding="utf-8").replace("/*__DADOS__*/null", json_dados)
    DESTINO.write_text(html, encoding="utf-8")
    print(f"Painel gerado: {DESTINO} ({DESTINO.stat().st_size / 1024:.0f} KB, {len(imagens)} imagens)")


if __name__ == "__main__":
    main()
