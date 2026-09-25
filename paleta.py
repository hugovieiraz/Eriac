"""Recuperação da paleta da câmera e conversão RGB -> índice ordinal de paleta.

As 255 imagens usam exatamente 253 cores, e essas cores formam uma única curva no espaço
RGB (paleta tipo "ironbow": azul escuro -> magenta -> laranja -> amarelo -> branco). A
ordem das cores ao longo da curva é recuperada sem rótulo nenhum: árvore geradora mínima
das cores e caminho mais longo dessa árvore (o "diâmetro"), orientado do escuro para o claro.

O índice resultante (0 = início da paleta, 1 = fim) é ORDINAL: diz qual cor está mais
adiante na escala da câmera. Sem a faixa de temperatura configurada na câmera, ele não é
convertido em graus Celsius, e nenhuma diferença de índice é tratada como ΔT.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


def _codificar(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.int64)
    return (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]


def cores_unicas(imagens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    codigos, contagem = np.unique(_codificar(imagens).ravel(), return_counts=True)
    cores = np.stack([codigos >> 16, (codigos >> 8) & 255, codigos & 255], axis=1)
    return cores.astype(np.uint8), contagem


def recuperar_paleta(imagens: np.ndarray) -> tuple[np.ndarray, dict]:
    """Devolve as cores ordenadas (m, 3) e um dicionário de auditoria."""
    cores, contagem = cores_unicas(imagens)
    distancias = cdist(cores.astype(float), cores.astype(float))
    arvore = minimum_spanning_tree(distancias)
    d0 = shortest_path(arvore, directed=False, indices=0)
    a = int(np.argmax(d0))
    da, pred = shortest_path(arvore, directed=False, indices=a, return_predecessors=True)
    b = int(np.argmax(da))
    caminho = [b]
    while caminho[-1] != a:
        caminho.append(int(pred[caminho[-1]]))
    ordenadas = cores[caminho]
    luminancia = ordenadas.astype(float) @ [0.299, 0.587, 0.114]
    if luminancia[0] > luminancia[-1]:
        ordenadas = ordenadas[::-1]
        luminancia = luminancia[::-1]
    fora = np.min(cdist(cores.astype(float), ordenadas.astype(float)), axis=1)
    passos = np.linalg.norm(np.diff(ordenadas.astype(float), axis=0), axis=1)
    auditoria = {
        "cores_distintas": int(len(cores)),
        "cores_no_caminho": int(len(ordenadas)),
        "distancia_maxima_fora_do_caminho": float(fora.max()),
        "fracao_passos_luminancia_crescente": float(np.mean(np.diff(luminancia) >= 0)),
        "passo_rgb_mediano": float(np.median(passos)),
        "passo_rgb_maximo": float(passos.max()),
        "cor_inicial": ordenadas[0].tolist(),
        "cor_final": ordenadas[-1].tolist(),
        "pixels_por_cor_minimo": int(contagem.min()),
    }
    return ordenadas, auditoria


class ConversorPaleta:
    """RGB -> índice em [0, 1]. Cor exata usa tabela; cor fora da paleta (imagem perturbada)
    usa a cor de paleta mais próxima."""

    def __init__(self, paleta: np.ndarray):
        self.paleta = paleta
        self.m = len(paleta)
        self._tabela = {int(c): i for i, c in enumerate(_codificar(paleta))}
        self._arvore = cKDTree(paleta.astype(float))

    def indice(self, rgb: np.ndarray) -> np.ndarray:
        codigos = _codificar(rgb)
        unicos, inverso = np.unique(codigos.ravel(), return_inverse=True)
        idx = np.empty(len(unicos), dtype=np.float32)
        faltando = []
        for k, c in enumerate(unicos):
            i = self._tabela.get(int(c))
            if i is None:
                faltando.append(k)
            else:
                idx[k] = i
        if faltando:
            faltando = np.array(faltando)
            u = unicos[faltando]
            rgb_f = np.stack([u >> 16, (u >> 8) & 255, u & 255], axis=1).astype(float)
            idx[faltando] = self._arvore.query(rgb_f)[1]
        return (idx[inverso].reshape(codigos.shape) / (self.m - 1)).astype(np.float32)
