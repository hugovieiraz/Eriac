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
    }
    json_dados = json.dumps(dados, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = MODELO.read_text(encoding="utf-8").replace("/*__DADOS__*/null", json_dados)
    DESTINO.write_text(html, encoding="utf-8")
    print(f"Painel gerado: {DESTINO} ({DESTINO.stat().st_size / 1024:.0f} KB, {len(imagens)} imagens)")


if __name__ == "__main__":
    main()
