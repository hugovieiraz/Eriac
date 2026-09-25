"""Diagnostica termogramas novos com o modelo de aplicação.

    .venv/Scripts/python.exe diagnosticar.py foto.bmp
    .venv/Scripts/python.exe diagnosticar.py pasta_de_imagens/ --saida resultados/
    .venv/Scripts/python.exe diagnosticar.py foto.bmp --llm claude     (requer credencial)
    .venv/Scripts/python.exe diagnosticar.py --treinar                 (retreina e salva o modelo)

Para cada imagem grava <nome>.json (registro estruturado + verificação) e <nome>.md
(relatório); ao final, resumo.csv com uma linha por imagem.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

import relatorio_llm as rl
from aplicacao import ARQUIVO_MODELO, Diagnosticador, carregar_rgb

EXTENSOES = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def listar(entradas: list[str]) -> list[Path]:
    arquivos = []
    for e in map(Path, entradas):
        if e.is_dir():
            arquivos += sorted(p for p in e.rglob("*") if p.suffix.lower() in EXTENSOES)
        elif e.suffix.lower() in EXTENSOES:
            arquivos.append(e)
        else:
            print(f"ignorado (não é imagem): {e}", file=sys.stderr)
    return arquivos


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entradas", nargs="*", help="imagens ou pastas")
    ap.add_argument("--saida", default="diagnosticos", help="pasta de saída (padrão: diagnosticos/)")
    ap.add_argument("--llm", choices=["deterministico", "claude"], default="deterministico")
    ap.add_argument("--treinar", action="store_true", help="retreina o modelo com as 255 imagens e salva")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):  # terminais do Windows usam cp1252
        sys.stdout.reconfigure(errors="replace")

    if args.treinar:
        d = Diagnosticador().treinar()
        print(f"Modelo salvo em {d.salvar()}")
        if not args.entradas:
            return
    else:
        d = Diagnosticador.carregar(ARQUIVO_MODELO)
    arquivos = listar(args.entradas)
    if not arquivos:
        ap.error("nenhuma imagem encontrada")
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    linhas = []
    for arq in arquivos:
        rgb, tam = carregar_rgb(arq)
        r = d.diagnosticar(rgb, arq.name, tam)
        texto, meta = r["relatorio"], {"backend": "deterministico"}
        if args.llm == "claude":
            texto, meta = rl.gerar_claude_seguro(r["registro"])
            meta["backend"] = "claude"
        verif = rl.verificar(texto, r["registro"])
        base = saida / arq.stem
        base.with_suffix(".md").write_text(texto, encoding="utf-8")
        base.with_suffix(".json").write_text(json.dumps(
            {"registro": r["registro"], "verificacao": verif, "geracao": meta}, ensure_ascii=False, indent=2), encoding="utf-8")
        dg, q = r["registro"]["diagnostico"], r["registro"]["controle_de_qualidade"]
        linhas.append({"arquivo": str(arq), "condicao_estimada": dg["condicao_estimada"], "alfa": dg["alfa_estimado"],
                       "alfa_inf": dg["intervalo_alfa"][0], "alfa_sup": dg["intervalo_alfa"][1],
                       "espiras": dg["espiras_em_curto_estimadas"], "classe_mais_proxima": dg["classe_nominal_mais_proxima"],
                       "dentro_do_dominio": q["dentro_do_dominio"], "score_dominio": q["score_de_dominio"],
                       "recomendacao": r["registro"]["recomendacao"]["codigo"], "relatorio_aprovado": verif["aprovado"]})
        print(f"{arq.name}: alfa={dg['alfa_estimado']:.3f} [{dg['intervalo_alfa'][0]:.3f}, {dg['intervalo_alfa'][1]:.3f}] "
              f"≈ {dg['espiras_em_curto_estimadas']} espiras · {r['registro']['recomendacao']['codigo']}"
              f"{'' if q['dentro_do_dominio'] else ' · FORA DO DOMÍNIO'} · relatório {'aprovado' if verif['aprovado'] else 'REPROVADO'}")
    pd.DataFrame(linhas).to_csv(saida / "resumo.csv", index=False)
    print(f"{len(linhas)} imagens; resultados em {saida}/")


if __name__ == "__main__":
    main()
