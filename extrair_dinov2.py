"""Regenera os vetores DINOv2 ViT-S/14 (limpos e perturbados) no cache do projeto.

Este script NÃO roda automaticamente: o torch.hub baixa e executa o código do repositório
facebookresearch/dinov2. Rode-o conscientemente, uma vez:

    .venv/Scripts/python.exe extrair_dinov2.py

Saídas em cache/: dinov2_limpo.npy, dinov2_<perturbacao>.npy e dinov2_meta.json (SHA-256 das
imagens na ordem das linhas, SHA-256 dos pesos, commit do repositório do hub). Quando esses
arquivos existem, main.py os usa no lugar dos vetores da execução original e inclui DINOv2 e o
modelo híbrido no teste de robustez.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import config
import dados
import features as ft

REPO = "facebookresearch/dinov2"


def extrair(modelo, imagens: np.ndarray, lote: int = 16) -> np.ndarray:
    media = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    desvio = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensores = []
    for img in imagens:
        x = Image.fromarray(img).resize((224, 224), Image.Resampling.BICUBIC)
        t = torch.from_numpy(np.asarray(x).copy()).permute(2, 0, 1).float() / 255.0
        tensores.append((t - media) / desvio)
    saida = []
    with torch.inference_mode():
        for i in range(0, len(tensores), lote):
            saida.append(modelo(torch.stack(tensores[i:i + lote])).numpy())
    return np.concatenate(saida)


def main() -> None:
    torch.manual_seed(config.SEMENTE)
    catalogo = dados.construir_catalogo()
    imagens = dados.carregar_imagens(catalogo)
    modelo = torch.hub.load(REPO, "dinov2_vits14", pretrained=True, trust_repo=True).eval()
    hub = Path(torch.hub.get_dir())
    pesos = sorted(hub.rglob("dinov2_vits14*.pth"))
    repo_dir = next(hub.glob("facebookresearch_dinov2_*"), None)
    commit = None
    if repo_dir is not None and (repo_dir / ".git").exists():
        commit = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    config.CACHE.mkdir(exist_ok=True)
    np.save(config.CACHE / "dinov2_limpo.npy", extrair(modelo, imagens))
    for tipo in ft.PERTURBACOES:
        np.save(config.CACHE / f"dinov2_{tipo}.npy", extrair(modelo, ft.perturbar(imagens, tipo)))
    meta = {
        "repositorio": REPO, "commit_do_hub": commit, "diretorio_hub": str(repo_dir),
        "pesos": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in pesos},
        "sha256_imagens_em_ordem": catalogo.sha256.tolist(), "torch": torch.__version__,
    }
    (config.CACHE / "dinov2_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Vetores DINOv2 salvos em", config.CACHE)


if __name__ == "__main__":
    main()
