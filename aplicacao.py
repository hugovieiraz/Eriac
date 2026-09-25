"""Modelo de aplicação: treinado uma vez com as 255 imagens, salvo em disco e usado para
diagnosticar termogramas novos (linha de comando em diagnosticar.py, interface em interface/).

Receita: embeddings da MobileNetV3-small treinados com aumento de dados (cada imagem de
treino ganha cópias com ruído, desfoque, translação e rotação). Nos experimentos
(saida/experimentos/aumento_*.csv) foi a receita local mais robusta a imagens degradadas e a
que menos errou com campanha de gravação não vista. Roda inteira na máquina do usuário; o
DINOv2 exigiria baixar código pelo torch.hub. Os indicadores de paleta não entram no modelo:
servem ao relatório e à coerência visual.

O intervalo de α é o conformal calibrado retirando, um de cada vez, os níveis interiores de
severidade (treino com aumento, resíduos medidos só nas imagens originais do nível
retirado). É mais largo que o erro dentro de uma gravação, porque uma imagem nova vem, por
definição, de uma gravação que o modelo não viu.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image

import avaliacao as av
import config
import dados
import features as ft
import relatorio_llm as rl
from paleta import ConversorPaleta, recuperar_paleta

ARQUIVO_MODELO = config.RAIZ / "modelos" / "diagnosticador.joblib"
FILTRO = config.FILTRO_MEDIANA
TAMANHO = (320, 240)


def carregar_rgb(caminho_ou_bytes) -> tuple[np.ndarray, tuple[int, int]]:
    """Lê BMP/PNG/JPG, devolve RGB 320x240 e o tamanho original."""
    if isinstance(caminho_ou_bytes, (bytes, bytearray)):
        import io
        img = Image.open(io.BytesIO(caminho_ou_bytes))
    else:
        img = Image.open(caminho_ou_bytes)
    img = img.convert("RGB")
    original = img.size
    if img.size != TAMANHO:
        img = img.resize(TAMANHO, Image.Resampling.BICUBIC)
    return np.asarray(img), original


@dataclass
class Diagnosticador:
    paleta: np.ndarray | None = None
    modelos: dict = field(default_factory=dict)
    q_conformal: float = 0.0
    dominio: av.DetectorDominio | None = None
    ind_treino: pd.DataFrame | None = None
    alfa_treino: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------ features
    def _features(self, imagens: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
        ind = ft.indicadores(imagens, ConversorPaleta(self.paleta), filtro_mediana=FILTRO)
        return ft.mobilenet(imagens), ind

    # ------------------------------------------------------------ treino
    def treinar(self) -> "Diagnosticador":
        catalogo = dados.construir_catalogo()
        imagens = dados.carregar_imagens(catalogo)
        self.paleta, aud = recuperar_paleta(imagens)
        X, ind = self._features(imagens)
        copias = [ft.mobilenet(ft.perturbar(imagens, t, semente=config.SEMENTE + 1, intensidade=0.6))
                  for t in ft.PERTURBACOES]
        rep = 1 + len(copias)
        Xa = np.vstack([X] + copias)
        niveis, alfa, blocos = catalogo.nivel.to_numpy(), catalogo.alfa.to_numpy(), catalogo.bloco.to_numpy()
        na, aa, ba = np.tile(niveis, rep), np.tile(alfa, rep), np.tile(blocos, rep)
        por_bloco = av.divisoes_internas_por_grupo(ba)
        self.modelos["binaria"], _ = av.ajustar("binaria", Xa, (na > 0).astype(int), por_bloco)
        self.modelos["multiclasse"], _ = av.ajustar("multiclasse", Xa, na, por_bloco)
        self.modelos["regressao"], a = av.ajustar("regressao", Xa, aa, av.divisoes_internas_por_grupo(na))
        self.q_conformal = self._quantil_niveis_interiores(Xa, aa, na, len(X))
        self.dominio = av.DetectorDominio().ajustar(Xa)
        self.ind_treino, self.alfa_treino = ind, alfa
        self.meta = {"treinado_em": datetime.now().isoformat(timespec="seconds"), "n_imagens": len(catalogo),
                     "receita": "MobileNetV3-small com aumento de dados", "ridge_alpha": a,
                     "q_conformal": self.q_conformal, "paleta_cores": aud["cores_distintas"],
                     "intervalo": "conformal 90% com resíduos de níveis interiores retirados um a um"}
        return self

    def _quantil_niveis_interiores(self, Xa, aa, na, n_originais: int) -> float:
        """Quantil conformal: retira cada nível interior (com suas cópias) do treino e mede o
        resíduo só nas imagens originais desse nível."""
        original = np.arange(len(Xa)) < n_originais
        alfa_nivel = pd.Series(aa[original]).groupby(na[original]).first()
        residuos = []
        for g, a in alfa_nivel.items():
            outros = alfa_nivel.drop(g)
            if not (outros.min() < a < outros.max()):
                continue
            m = av.clone(self.modelos["regressao"]).fit(Xa[na != g], aa[na != g])
            teste = original & (na == g)
            residuos.append(np.abs(np.clip(m.predict(Xa[teste]), 0, 1) - aa[teste]))
        res = np.concatenate(residuos)
        nivel = min(1.0, np.ceil((len(res) + 1) * config.NIVEL_INTERVALO) / len(res))
        return float(np.quantile(res, nivel, method="higher"))

    def salvar(self, caminho: Path = ARQUIVO_MODELO) -> Path:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, caminho, compress=3)
        caminho.with_suffix(".json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
        return caminho

    @staticmethod
    def carregar(caminho: Path = ARQUIVO_MODELO, treinar_se_faltar: bool = True) -> "Diagnosticador":
        if caminho.exists():
            return joblib.load(caminho)
        if not treinar_se_faltar:
            raise FileNotFoundError(caminho)
        d = Diagnosticador().treinar()
        d.salvar(caminho)
        return d

    # ------------------------------------------------------------ uso
    def diagnosticar(self, rgb: np.ndarray, arquivo: str = "imagem", tamanho_original=TAMANHO) -> dict:
        X, ind = self._features(rgb[None])
        prob = float(self.modelos["binaria"].predict_proba(X)[0, 1])
        conf = float(self.modelos["multiclasse"].predict_proba(X)[0].max())
        alfa = float(np.clip(self.modelos["regressao"].predict(X)[0], 0, 1))
        score = float(self.dominio.pontuar(X)[0])
        coer = float(av.coerencia_visual(self.ind_treino, self.alfa_treino, ind, np.array([alfa]))[0])
        oof = SimpleNamespace(prob_defeito=prob, alfa_previsto=alfa, alfa_inferior=max(0.0, alfa - self.q_conformal),
                              alfa_superior=min(1.0, alfa + self.q_conformal), confianca_classificador=conf,
                              score_dominio=score, coerencia_visual=coer)
        registro = rl.registro_diagnostico(SimpleNamespace(arquivo=arquivo), oof, ind.iloc[0],
                                           resolucao=list(tamanho_original))
        texto = rl.gerar_deterministico(registro)
        return {"registro": registro, "relatorio": texto, "verificacao": rl.verificar(texto, registro),
                "indicadores": ind.iloc[0].to_dict()}

    def oclusao(self, rgb: np.ndarray) -> dict:
        """Mapa de oclusão da regressão de α: vermelho onde ocultar a região derruba α
        (ela puxava a estimativa para cima), azul onde a eleva. Custa ~165 passadas da rede."""
        from matplotlib import colormaps

        mapa = ft.mapa_oclusao(rgb, self.modelos["regressao"], ft.mobilenet)
        mascara = ft.mascara_transformador(rgb)
        positivo = np.clip(mapa, 0, None)
        limite = max(float(np.abs(mapa).max()), 1e-6)
        cor = colormaps["RdBu_r"]((mapa / limite + 1) / 2)[..., :3]
        cinza = (rgb.astype(float) @ [0.299, 0.587, 0.114]) / 255
        imagem = (0.35 * np.stack([cinza] * 3, axis=-1) + 0.65 * cor) * 255
        return {"imagem": imagem.astype(np.uint8),
                "fracao_na_roi": float(positivo[mascara].sum() / max(positivo.sum(), 1e-12)),
                "escala_espiras": limite * config.TOTAL_ESPIRAS}

    def mapas(self, rgb: np.ndarray) -> dict[str, np.ndarray]:
        """Imagens auxiliares para a interface: índice de paleta e ROI com região quente."""
        conv = ConversorPaleta(self.paleta)
        _, mascara, quente = ft.indicadores_imagem(rgb, conv, FILTRO)
        indice = cv2.medianBlur(conv.indice(rgb).astype(np.float32), FILTRO)
        cinza = (np.clip(indice, 0, 1) * 255).astype(np.uint8)
        sobre = rgb.copy()
        sobre[~mascara] = (sobre[~mascara] * 0.25).astype(np.uint8)
        sobre[quente] = [255, 255, 255]
        return {"indice": np.stack([cinza] * 3, axis=-1), "roi": sobre}
