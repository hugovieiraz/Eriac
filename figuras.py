"""Figuras do artigo (PNG estático, 180 dpi). Paleta categórica em ordem fixa, eixos recessivos,
marcadores diferentes por série para leitura em impressão em tons de cinza."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

SUPERFICIE, TEXTO, TEXTO_2, GRADE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARCADORES = ["o", "s", "^", "D", "v", "P", "X", "*"]
SEQUENCIAL = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
    "axes.edgecolor": GRADE, "axes.labelcolor": TEXTO_2, "xtick.color": TEXTO_2, "ytick.color": TEXTO_2,
    "text.color": TEXTO, "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "lines.linewidth": 1.6, "lines.markersize": 5, "legend.frameon": False,
})

NOMES = {
    "trivial": "Trivial 16x12", "posicao": "Posição (controle)", "indicadores_v1": "Indicadores v1",
    "indicadores": "Indicadores (paleta)", "mobilenet": "MobileNetV3", "dinov2": "DINOv2",
    "hibrido_v1": "DINOv2 + ind. v1", "hibrido": "DINOv2 + ind. (paleta)",
}


def _salvar(fig, nome: str) -> None:
    fig.tight_layout()
    fig.savefig(config.FIGURAS / nome, dpi=180)
    plt.close(fig)


def paleta_e_saturacao(paleta: np.ndarray, ind: pd.DataFrame, catalogo: pd.DataFrame) -> None:
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 3.6), gridspec_kw={"height_ratios": [1, 2.2]})
    a1.imshow(paleta[None, :, :], aspect="auto", extent=(0, 1, 0, 1))
    a1.set_yticks([])
    a1.grid(False)
    a1.set_xlabel("Índice ordinal de paleta recuperado (0 = início, 1 = fim; sem unidade de temperatura)")
    a1.set_title(f"Paleta da câmera: {len(paleta)} cores ordenadas sem rótulo")
    sat = ind.groupby(catalogo.condicao, sort=False).fracao_saturada.mean().reindex(config.ORDEM_CLASSES)
    a2.bar(range(9), sat.to_numpy() * 100, color=SERIES[0], width=0.7)
    a2.set_xticks(range(9), config.ORDEM_CLASSES)
    a2.set_ylabel("% da ROI no topo da paleta")
    a2.set_title("Saturação: pixels no fim da paleta (informação cortada)")
    for i, v in enumerate(sat.to_numpy() * 100):
        if v >= 0.5:
            a2.text(i, v, f"{v:.1f}", ha="center", va="bottom", color=TEXTO_2, fontsize=8)
    _salvar(fig, "fig_paleta_saturacao.png")


def segmentacao(imagens, catalogo, conversor, indicadores_imagem) -> None:
    escolhidas = [catalogo.index[catalogo.nivel == n][len(catalogo.index[catalogo.nivel == n]) // 2] for n in (0, 4, 8)]
    fig, eixos = plt.subplots(3, 3, figsize=(8, 6.2))
    for linha, i in enumerate(escolhidas):
        rgb = imagens[i]
        _, mascara, quente = indicadores_imagem(rgb, conversor)
        indice = conversor.indice(rgb)
        sobreposicao = rgb.copy()
        sobreposicao[~mascara] = (sobreposicao[~mascara] * 0.25).astype(np.uint8)
        sobreposicao[quente] = [255, 255, 255]
        for eixo, img, titulo in zip(eixos[linha], (rgb, indice, sobreposicao),
                                     (f"{catalogo.condicao[i]}: RGB original", "Índice de paleta (0-1)",
                                      "ROI e região relativamente quente")):
            eixo.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=1 if img.ndim == 2 else None)
            eixo.set_title(titulo, fontsize=8.5)
            eixo.axis("off")
    _salvar(fig, "fig_segmentacao.png")


def indicadores_por_severidade(ind: pd.DataFrame, catalogo: pd.DataFrame) -> None:
    colunas = [("indice_media", "Índice médio de paleta na ROI"), ("fracao_area_quente", "Fração de área relativamente quente"),
               ("gradiente_medio", "Gradiente médio do índice"), ("fundo_mediana", "Mediana do índice no fundo")]
    df = ind.assign(alfa=catalogo.alfa, campanha=catalogo.campanha)
    fig, eixos = plt.subplots(2, 2, figsize=(8, 5.6))
    for eixo, (col, rotulo) in zip(eixos.flat, colunas):
        for k, (camp, g) in enumerate(df.groupby("campanha")):
            s = g.groupby("alfa")[col].agg(["mean", "std"]).reset_index()
            eixo.errorbar(s.alfa, s["mean"], yerr=s["std"], marker=MARCADORES[k], color=SERIES[k],
                          capsize=2, label=f"Campanha {camp}", linestyle="-")
        eixo.set_xlabel("Severidade declarada α")
        eixo.set_title(rotulo, fontsize=9)
    eixos[0, 0].legend(fontsize=7.5)
    _salvar(fig, "fig_indicadores_severidade.png")


def esquemas(resumo: pd.DataFrame, ordem: list[str]) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    y = np.arange(len(ordem))
    for k, (esq, rotulo) in enumerate((("aleatoria", "Aleatória por imagem"), ("blocos", "Blocos temporais + purga"))):
        r = resumo[resumo.esquema == esq].set_index("conjunto").reindex(ordem)
        a1.barh(y + (k - 0.5) * 0.38, r.mul_f1_macro, height=0.36, color=SERIES[k], label=rotulo)
        a2.barh(y + (k - 0.5) * 0.38, r.reg_mae_espiras, height=0.36, color=SERIES[k], label=rotulo)
    for eixo in (a1, a2):
        eixo.set_yticks(y, [NOMES[c] for c in ordem])
        eixo.invert_yaxis()
    a2.set_yticklabels([])
    a1.set_xlabel("F1-macro (9 classes)")
    a1.set_xlim(0, 1.02)
    a2.set_xlabel("MAE de severidade (espiras)")
    a1.set_title("Classificação", pad=22)
    a2.set_title("Regressão de α", pad=22)
    manipuladores, rotulos = a1.get_legend_handles_labels()
    fig.legend(manipuladores, rotulos, loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 0.93))
    _salvar(fig, "fig_esquemas_validacao.png")


def interpolacao(tabela: pd.DataFrame, conjuntos: list[str]) -> None:
    fig, eixo = plt.subplots(figsize=(5.2, 4.4))
    eixo.plot([0, 1], [0, 1], color=TEXTO_2, linewidth=1, linestyle="--", label="Ideal")
    for k, c in enumerate(conjuntos):
        t = tabela[tabela.conjunto == c]
        eixo.errorbar(t.alfa_real, t.alfa_previsto_medio, yerr=t.alfa_previsto_dp, marker=MARCADORES[k],
                      color=SERIES[k], linestyle="none", capsize=2, markersize=6, label=NOMES[c])
    eixo.set_xlabel("α real da condição retirada do treino")
    eixo.set_ylabel("α previsto (média ± dp)")
    eixo.set_title("Severidade não vista no treinamento")
    eixo.legend(fontsize=7.5)
    _salvar(fig, "fig_interpolacao.png")


def campanhas(tabela: pd.DataFrame, conjuntos: list[str]) -> None:
    fig, eixo = plt.subplots(figsize=(5.2, 4.4))
    for faixa in tabela[tabela.tipo == "extrapolacao"].groupby("campanha_retirada").alfa_real.agg(["min", "max"]).itertuples():
        eixo.axvspan(faixa.min - 0.03, faixa.max + 0.03, color=GRADE, alpha=0.6, linewidth=0)
    eixo.text(0.83, 0.05, "extrapolação\n(campanha D)", ha="center", fontsize=7.5, color=TEXTO_2)
    eixo.text(0.0, 0.62, "extrapolação\n(campanha A)", ha="center", fontsize=7.5, color=TEXTO_2)
    eixo.plot([0, 1], [0, 1], color=TEXTO_2, linewidth=1, linestyle="--", label="Ideal")
    for k, c in enumerate(conjuntos):
        t = tabela[tabela.conjunto == c]
        eixo.plot(t.alfa_real, t.alfa_previsto_medio, marker=MARCADORES[k], color=SERIES[k],
                  linestyle="none", markersize=6, label=NOMES[c])
    eixo.set_xlabel("α real (campanha inteira fora do treino)")
    eixo.set_ylabel("α previsto médio")
    eixo.set_title("Campanha de aquisição não vista")
    eixo.legend(fontsize=7.5)
    _salvar(fig, "fig_campanhas.png")


def confusao(matriz: np.ndarray, titulo: str, nome: str) -> None:
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq", SEQUENCIAL)
    fig, eixo = plt.subplots(figsize=(5.6, 4.8))
    eixo.imshow(matriz, cmap=cmap)
    eixo.grid(False)
    eixo.set_xticks(range(9), config.ORDEM_CLASSES, rotation=45, ha="right")
    eixo.set_yticks(range(9), config.ORDEM_CLASSES)
    eixo.set_xlabel("Previsto")
    eixo.set_ylabel("Real")
    eixo.set_title(titulo)
    limite = matriz.max() * 0.55
    for i in range(9):
        for j in range(9):
            if matriz[i, j]:
                eixo.text(j, i, matriz[i, j], ha="center", va="center", fontsize=7.5,
                          color="#ffffff" if matriz[i, j] > limite else TEXTO)
    _salvar(fig, nome)


def intervalos(oof: pd.DataFrame, catalogo: pd.DataFrame) -> None:
    df = oof.merge(catalogo[["alfa"]], left_on="indice", right_index=True)
    rng = np.random.default_rng(config.SEMENTE)
    x = df.alfa + rng.uniform(-0.018, 0.018, len(df))
    fig, eixo = plt.subplots(figsize=(6, 4.4))
    eixo.vlines(x, df.alfa_inferior, df.alfa_superior, color=SERIES[0], alpha=0.35, linewidth=1)
    eixo.plot(x, df.alfa_previsto, "o", color=SERIES[0], markersize=3, label="α previsto e intervalo de 90%")
    eixo.plot([0, 1], [0, 1], color=TEXTO_2, linestyle="--", linewidth=1, label="Ideal")
    eixo.set_xlabel("α declarado")
    eixo.set_ylabel("α previsto")
    eixo.set_title("Previsões fora da dobra (blocos temporais)")
    eixo.legend(fontsize=7.5, loc="upper left")
    _salvar(fig, "fig_intervalos_conformais.png")


def oclusao(imagens, mapas: dict[int, dict[str, np.ndarray]], catalogo) -> None:
    indices = list(mapas)
    modelos = list(next(iter(mapas.values())))
    fig, eixos = plt.subplots(len(indices), 1 + len(modelos), figsize=(3 * (1 + len(modelos)), 2.4 * len(indices)))
    for linha, i in enumerate(indices):
        eixos[linha, 0].imshow(imagens[i])
        eixos[linha, 0].set_title(f"{catalogo.condicao[i]}", fontsize=8.5)
        cinza = imagens[i].astype(float) @ [0.299, 0.587, 0.114]
        for col, m in enumerate(modelos, start=1):
            mapa = mapas[i][m]
            lim = max(np.abs(mapa).max(), 1e-6)
            eixos[linha, col].imshow(cinza, cmap="gray", vmin=0, vmax=255)
            h = eixos[linha, col].imshow(mapa, cmap="RdBu_r", vmin=-lim, vmax=lim, alpha=0.7,
                                         extent=(0, imagens.shape[2], imagens.shape[1], 0))
            eixos[linha, col].set_title(f"{NOMES[m]}: queda de α ao ocultar", fontsize=8)
            fig.colorbar(h, ax=eixos[linha, col], fraction=0.035)
        for eixo in eixos[linha]:
            eixo.axis("off")
    _salvar(fig, "fig_oclusao.png")
