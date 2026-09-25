"""Catálogo do dataset, campanhas de aquisição e blocos temporais.

Termos:
- ordem: posição do quadro dentro da pasta, pelo número no nome do arquivo.
- bloco: pedaço contíguo no tempo dentro de cada classe (0..N_BLOCOS-1). A validação por
  blocos testa sempre um trecho que o modelo não viu, em vez de quadros vizinhos quase
  idênticos aos do treino.
- campanha: sequência de gravação. A numeração dos arquivos continua de uma pasta para a
  seguinte quando elas foram gravadas em sequência (p2 termina em 026, p3 começa em 027).
"""
from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd
from PIL import Image

import config

_NUMERO = re.compile(r"p\d(\d+)\.bmp$", re.IGNORECASE)


def numero_do_arquivo(nome: str) -> int:
    """'p5042.bmp' -> 42 (o primeiro dígito é o índice da pasta, o resto é a numeração)."""
    m = _NUMERO.search(nome)
    if m is None:
        raise ValueError(f"nome fora do padrão pNxxx.bmp: {nome}")
    return int(m.group(1))


def atribuir_blocos(n: int, n_blocos: int = config.N_BLOCOS) -> np.ndarray:
    """n quadros ordenados -> n_blocos pedaços contíguos. n=22 dá [5, 5, 4, 4, 4]."""
    if n < n_blocos:
        raise ValueError(f"{n} quadros não cabem em {n_blocos} blocos")
    partes = np.array_split(np.arange(n), n_blocos)
    return np.concatenate([np.full(len(p), i) for i, p in enumerate(partes)])


def inferir_campanhas(faixas: dict[str, tuple[int, int]]) -> dict[str, str]:
    """Agrupa pastas cuja numeração é contínua (início = fim da anterior + 1).

    `faixas` mapeia pasta -> (primeiro número, último número), na ordem das pastas.
    """
    campanhas, rotulo, anterior_fim = {}, "@", None
    for pasta, (inicio, fim) in faixas.items():
        if anterior_fim is None or inicio != anterior_fim + 1:
            rotulo = chr(ord(rotulo) + 1)
        campanhas[pasta] = rotulo
        anterior_fim = fim
    return campanhas


def construir_catalogo() -> pd.DataFrame:
    linhas = []
    for nivel, (pasta, (condicao, espiras)) in enumerate(config.CLASSES.items()):
        arquivos = sorted((config.DATASET / pasta).glob("*.bmp"), key=lambda p: numero_do_arquivo(p.name))
        blocos = atribuir_blocos(len(arquivos))
        for ordem, (caminho, bloco) in enumerate(zip(arquivos, blocos)):
            linhas.append({
                "arquivo": caminho.name,
                "pasta": pasta,
                "caminho": str(caminho),
                "condicao": condicao,
                "nivel": nivel,
                "espiras": espiras,
                "alfa": espiras / config.TOTAL_ESPIRAS,
                "numero": numero_do_arquivo(caminho.name),
                "ordem": ordem,
                "bloco": int(bloco),
                "sha256": hashlib.sha256(caminho.read_bytes()).hexdigest(),
            })
    cat = pd.DataFrame(linhas)
    if len(cat) != 255:
        raise RuntimeError(f"Esperadas 255 imagens; encontradas {len(cat)}.")
    faixas = cat.groupby("pasta", sort=False).numero.agg(["min", "max"])
    campanhas = inferir_campanhas({p: (int(r["min"]), int(r["max"])) for p, r in faixas.iterrows()})
    cat["campanha"] = cat.pasta.map(campanhas)
    return cat


def carregar_imagens(catalogo: pd.DataFrame) -> np.ndarray:
    """Lê todas as imagens uma única vez: uint8 (n, 240, 320, 3)."""
    return np.stack([np.asarray(Image.open(c).convert("RGB")) for c in catalogo.caminho])


def vizinhos_para_purgar(catalogo: pd.DataFrame, teste: np.ndarray, purga: int = config.PURGA) -> np.ndarray:
    """Índices de treino a remover: até `purga` quadros de cada lado de cada trecho de teste,
    dentro da mesma pasta. O conjunto de teste fica intacto."""
    teste_set = set(teste.tolist())
    remover = []
    for _, grupo in catalogo.groupby("pasta", sort=False):
        indices = grupo.sort_values("ordem").index.to_numpy()
        no_teste = np.array([i in teste_set for i in indices])
        for pos in np.flatnonzero(no_teste):
            for d in range(1, purga + 1):
                for viz in (pos - d, pos + d):
                    if 0 <= viz < len(indices) and not no_teste[viz]:
                        remover.append(indices[viz])
    return np.unique(np.array(remover, dtype=int))
