"""Conjuntos de features. Todos recebem as imagens RGB já carregadas e devolvem (n, d).

- trivial: imagem em cinza reduzida a 16x12. Linha de base: se algo sofisticado não a
  supera, não aprendeu nada além do brilho global.
- posicao: caixa da região segmentada (centro x, centro y, largura, altura). Controle: não
  tem padrão térmico nenhum, só onde o objeto está no quadro.
- indicadores: descritores relativos calculados sobre o índice ordinal de paleta, depois de
  um filtro de mediana 5x5 (config.FILTRO_MEDIANA) que absorve ruído de sensor.
- indicadores_v1: os 44 indicadores do rascunho original (lidos do CSV da execução original).
- mobilenet: embeddings congelados da MobileNetV3-small (ImageNet), pesos locais.
- dinov2: embeddings DINOv2 ViT-S/14 da execução original (conferidos por SHA-256).
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import pandas as pd

import config
from paleta import ConversorPaleta

LIMIAR_QUENTE = 0.65      # índice de paleta normalizado; sem interpretação em °C
LIMIAR_SATURADO = 0.995   # topo da paleta: pixel no branco, informação cortada


# ------------------------------------------------------------ segmentação ----
def maior_componente(mascara: np.ndarray) -> np.ndarray:
    n, rotulos, stats, _ = cv2.connectedComponentsWithStats(mascara.astype(np.uint8), 8)
    if n <= 1:
        return np.ones(mascara.shape, dtype=bool)
    return rotulos == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def mascara_transformador(rgb: np.ndarray) -> np.ndarray:
    """Mesma segmentação visual do rascunho: distância CIELAB à mediana dos quatro cantos,
    limiar no percentil 60, morfologia e maior componente conectada. Não é o contorno
    físico validado do enrolamento."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    k = max(8, min(h, w) // 12)
    cantos = np.concatenate((lab[:k, :k], lab[:k, -k:], lab[-k:, :k], lab[-k:, -k:]), axis=0).reshape(-1, 3)
    distancia = np.linalg.norm(lab - np.median(cantos, axis=0), axis=2)
    mascara = distancia >= max(float(np.percentile(distancia, 60)), 7.0)
    nucleo = np.ones((5, 5), dtype=np.uint8)
    mascara = cv2.morphologyEx(mascara.astype(np.uint8), cv2.MORPH_CLOSE, nucleo, iterations=2)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, nucleo, iterations=1).astype(bool)
    mascara = maior_componente(mascara)
    if mascara.mean() < 0.05 or mascara.mean() > 0.90:
        yy, xx = np.ogrid[:h, :w]
        mascara = (xx > 0.10 * w) & (xx < 0.90 * w) & (yy > 0.08 * h) & (yy < 0.92 * h)
    return mascara


def _entropia(valores: np.ndarray, bins: int = 32) -> float:
    hist, _ = np.histogram(valores, bins=bins, range=(0.0, 1.0))
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0


# ------------------------------------------------------------ indicadores ----
def indicadores_imagem(rgb: np.ndarray, conversor: ConversorPaleta, filtro_mediana: int = config.FILTRO_MEDIANA,
                       limiar_quente: float = LIMIAR_QUENTE) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """`filtro_mediana` (3 ou 5) aplica um filtro de mediana ao mapa de índice antes dos
    indicadores: ruído de sensor vira cores fora da paleta, e a mediana as absorve."""
    mascara = mascara_transformador(rgb)
    indice = conversor.indice(rgb)
    if filtro_mediana:
        indice = cv2.medianBlur(indice.astype(np.float32), filtro_mediana)
    roi = indice[mascara]
    quente = mascara & (indice >= limiar_quente)
    if quente.sum() < 5:
        quente = mascara & (indice >= np.percentile(roi, 90))
    h, w = mascara.shape
    yy, xx = np.indices(mascara.shape)
    pesos = np.where(quente, indice, 0.0)
    soma = pesos.sum()
    if soma == 0:
        cx, cy = float(xx[mascara].mean()), float(yy[mascara].mean())
    else:
        cx, cy = float((xx * pesos).sum() / soma), float((yy * pesos).sum() / soma)
    dx, dy = xx[quente] / w - cx / w, yy[quente] / h - cy / h
    gx = cv2.Sobel(indice, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(indice, cv2.CV_32F, 0, 1, ksize=3)
    gradiente = np.sqrt(gx ** 2 + gy ** 2)[mascara]
    n_comp, _, stats, _ = cv2.connectedComponentsWithStats(quente.astype(np.uint8), 8)
    areas = stats[1:, cv2.CC_STAT_AREA] if n_comp > 1 else np.array([])
    centrado = roi - roi.mean()
    desvio = roi.std()
    fundo = indice[~mascara] if (~mascara).any() else np.array([0.0])
    f = {
        "fracao_area_roi": float(mascara.mean()),
        "fracao_area_quente": float(quente.sum() / mascara.sum()),
        "fracao_saturada": float(np.mean(roi >= LIMIAR_SATURADO)),
        "indice_media": float(roi.mean()),
        "indice_p50": float(np.percentile(roi, 50)),
        "indice_p90": float(np.percentile(roi, 90)),
        "indice_p95": float(np.percentile(roi, 95)),
        "indice_p99": float(np.percentile(roi, 99)),
        "indice_max": float(roi.max()),
        "indice_desvio": float(desvio),
        "indice_assimetria": float(np.mean(centrado ** 3) / (desvio ** 3 + 1e-12)),
        "indice_integrado": float(roi.sum() / mascara.size),
        "contraste_roi_fundo": float(roi.mean() - np.median(fundo)),
        "fundo_mediana": float(np.median(fundo)),
        "hotspot_x": cx / w,
        "hotspot_y": cy / h,
        "hotspot_distancia_centro": float(math.hypot(cx / w - 0.5, cy / h - 0.5)),
        "hotspot_dispersao": float(np.sqrt(np.mean(dx * dx + dy * dy))) if quente.any() else 0.0,
        "gradiente_medio": float(gradiente.mean()),
        "gradiente_p90": float(np.percentile(gradiente, 90)),
        "entropia": _entropia(roi),
        "componentes_quentes": float(len(areas)),
        "maior_componente_quente": float(areas.max() / mascara.sum()) if len(areas) else 0.0,
    }
    hist, _ = np.histogram(roi, bins=16, range=(0.0, 1.0))
    for i, v in enumerate(hist / hist.sum()):
        f[f"hist_indice_{i:02d}"] = float(v)
    return f, mascara, quente


def indicadores(imagens: np.ndarray, conversor: ConversorPaleta, filtro_mediana: int = config.FILTRO_MEDIANA,
                limiar_quente: float = LIMIAR_QUENTE) -> pd.DataFrame:
    return pd.DataFrame([indicadores_imagem(img, conversor, filtro_mediana, limiar_quente)[0] for img in imagens])


# Indicadores que dependem do enquadramento ou do fundo, e portanto da campanha de gravação.
INDICADORES_DE_CENA = ["fundo_mediana", "fracao_area_roi", "hotspot_x", "hotspot_y"]


# ------------------------------------------------------ baseline e controle ----
def trivial(imagens: np.ndarray) -> np.ndarray:
    cinza = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in imagens]
    return np.stack([cv2.resize(c, (16, 12), interpolation=cv2.INTER_AREA).ravel() / 255.0 for c in cinza])


def posicao(imagens: np.ndarray) -> np.ndarray:
    linhas = []
    for img in imagens:
        m = mascara_transformador(img)
        ys, xs = np.nonzero(m)
        h, w = m.shape
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        linhas.append([(x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h])
    return np.array(linhas)


# ------------------------------------------------------------- redes -------
_MOBILENET = None


def _mobilenet():
    """MobileNetV3-small com pesos ImageNet já presentes no cache local do torch."""
    global _MOBILENET
    if _MOBILENET is None:
        import torch
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        torch.manual_seed(config.SEMENTE)
        modelo = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        modelo.classifier = torch.nn.Identity()
        _MOBILENET = modelo.eval()
    return _MOBILENET


def mobilenet(imagens: np.ndarray, lote: int = 32) -> np.ndarray:
    import torch

    modelo = _mobilenet()
    media = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    desvio = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    saidas = []
    with torch.inference_mode():
        for i in range(0, len(imagens), lote):
            x = np.stack([cv2.resize(im, (224, 224), interpolation=cv2.INTER_CUBIC) for im in imagens[i:i + lote]])
            x = ((x.astype(np.float32) / 255.0 - media) / desvio).transpose(0, 3, 1, 2)
            saidas.append(modelo(torch.from_numpy(np.ascontiguousarray(x))).numpy())
    return np.concatenate(saidas)


def dinov2_original(catalogo: pd.DataFrame) -> np.ndarray:
    """Vetores DINOv2 da execução original, reordenados pelo SHA-256 de cada imagem."""
    cat_original = pd.read_csv(config.REFERENCIA / "catalogo_imagens.csv")
    vetores = np.load(config.REFERENCIA / "vetores_dinov2.npy")
    posicao_por_hash = {h: i for i, h in enumerate(cat_original.sha256)}
    faltando = set(catalogo.sha256) - set(posicao_por_hash)
    if faltando:
        raise RuntimeError(f"{len(faltando)} imagens sem vetor DINOv2 correspondente.")
    return vetores[[posicao_por_hash[h] for h in catalogo.sha256]]


def _dinov2_cache_valido(catalogo: pd.DataFrame) -> bool:
    meta = config.CACHE / "dinov2_meta.json"
    if not meta.exists() or not (config.CACHE / "dinov2_limpo.npy").exists():
        return False
    import json
    return json.loads(meta.read_text(encoding="utf-8"))["sha256_imagens_em_ordem"] == catalogo.sha256.tolist()


def dinov2(catalogo: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Vetores regenerados por extrair_dinov2.py, se existirem; senão, os da execução original."""
    if _dinov2_cache_valido(catalogo):
        return np.load(config.CACHE / "dinov2_limpo.npy"), "regenerado (extrair_dinov2.py)"
    return dinov2_original(catalogo), "execução original (vetores_dinov2.npy)"


def dinov2_perturbado(catalogo: pd.DataFrame, tipo: str) -> np.ndarray | None:
    arq = config.CACHE / f"dinov2_{tipo}.npy"
    return np.load(arq) if _dinov2_cache_valido(catalogo) and arq.exists() else None


def indicadores_v1_original(catalogo: pd.DataFrame) -> pd.DataFrame:
    """Os 44 indicadores exatamente como a execução original os calculou."""
    orig = pd.read_csv(config.REFERENCIA / "indicadores_relativos.csv")
    cat_cols = ["file", "folder", "filename", "condition", "turns_short", "alpha",
                "severity_level", "width", "height", "mode", "sha256"]
    nomes = [c for c in orig.columns if c not in cat_cols]
    orig = orig.set_index("sha256")
    return orig.loc[catalogo.sha256, nomes].reset_index(drop=True)


# ---------------------------------------------------------- perturbações ----
def perturbar(imagens: np.ndarray, tipo: str, semente: int = config.SEMENTE, intensidade: float = 1.0) -> np.ndarray:
    """Perturbações de teste. `intensidade` escala todas (1 = a do teste; o aumento de dados
    de treino usa intensidades diferentes para não treinar exatamente no que é testado)."""
    rng = np.random.default_rng(semente)
    saida = []
    for img in imagens:
        if tipo == "ruido":
            x = np.clip(img.astype(np.float32) + rng.normal(0, 8 * intensidade, img.shape), 0, 255).astype(np.uint8)
        elif tipo == "desfoque":
            x = cv2.GaussianBlur(img, (0, 0), 1.5 * intensidade)
        elif tipo == "translacao":
            m = np.float32([[1, 0, 12 * intensidade], [0, 1, 8 * intensidade]])
            x = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        elif tipo == "rotacao":
            m = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 4 * intensidade, 1.0)
            x = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        else:
            raise ValueError(tipo)
        saida.append(x)
    return np.stack(saida)


PERTURBACOES = ["ruido", "desfoque", "translacao", "rotacao"]


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

