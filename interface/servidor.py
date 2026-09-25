"""Interface web local para diagnosticar termogramas. Só a biblioteca padrão do Python.

    .venv/Scripts/python.exe interface/servidor.py            (abre http://127.0.0.1:8000)
    .venv/Scripts/python.exe interface/servidor.py --porta 8765 --sem-navegador

O servidor escuta só em 127.0.0.1: nada fica acessível a outras máquinas da rede.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import config  # noqa: E402
from aplicacao import Diagnosticador, carregar_rgb  # noqa: E402

PAGINA = Path(__file__).with_name("index.html")
LIMITE_BYTES = 15 * 1024 * 1024
TRAVA = threading.Lock()  # o modelo não é usado por duas requisições ao mesmo tempo


def png_base64(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def listar_exemplos() -> list[dict]:
    """Três quadros (início, meio, fim) de cada condição, se o dataset estiver baixado."""
    exemplos = []
    for pasta, (condicao, espiras) in config.CLASSES.items():
        arquivos = sorted((config.DATASET / pasta).glob("*.bmp"))
        for arq in [arquivos[i] for i in (0, len(arquivos) // 2, -1)] if arquivos else []:
            exemplos.append({"pasta": pasta, "arquivo": arq.name, "condicao": condicao, "espiras": espiras})
    return exemplos


class Handler(BaseHTTPRequestHandler):
    modelo: Diagnosticador
    exemplos: list[dict]

    def log_message(self, fmt, *args):  # silencia o log por requisição
        pass

    def _responder(self, codigo: int, corpo: bytes, tipo: str) -> None:
        self.send_response(codigo)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(corpo)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(corpo)

    def _json(self, obj, codigo: int = 200) -> None:
        self._responder(codigo, json.dumps(obj, ensure_ascii=False, default=float).encode("utf-8"), "application/json; charset=utf-8")

    def _exemplo(self, consulta: dict) -> Path | None:
        pasta, arquivo = consulta.get("pasta", [""])[0], consulta.get("arquivo", [""])[0]
        permitido = {(e["pasta"], e["arquivo"]) for e in self.exemplos}
        return config.DATASET / pasta / arquivo if (pasta, arquivo) in permitido else None

    def _diagnosticar(self, rgb: np.ndarray, nome: str, tamanho) -> dict:
        with TRAVA:
            r = self.modelo.diagnosticar(rgb, nome, tamanho)
            mapas = self.modelo.mapas(rgb)
        r["imagens"] = {"original": png_base64(rgb), "indice": png_base64(mapas["indice"]), "roi": png_base64(mapas["roi"])}
        return r

    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            return self._responder(200, PAGINA.read_bytes(), "text/html; charset=utf-8")
        if url.path == "/api/info":
            return self._json({"modelo": self.modelo.meta, "exemplos": self.exemplos, "classes": config.ORDEM_CLASSES,
                               "alfas": config.ALFA_NOMINAL, "total_espiras": config.TOTAL_ESPIRAS,
                               "faixas": [{"ate": a, "codigo": c} for a, c, _ in config.FAIXAS_RECOMENDACAO]})
        if url.path == "/api/miniatura":
            caminho = self._exemplo(parse_qs(url.query))
            if caminho is None:
                return self._json({"erro": "exemplo desconhecido"}, 404)
            img = Image.open(caminho).convert("RGB")
            img.thumbnail((160, 120))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return self._responder(200, buf.getvalue(), "image/png")
        return self._json({"erro": "rota não encontrada"}, 404)

    def do_POST(self):
        url = urlparse(self.path)
        try:
            if url.path == "/api/diagnosticar":
                n = int(self.headers.get("Content-Length", 0))
                if not 0 < n <= LIMITE_BYTES:
                    return self._json({"erro": "Envie uma imagem de até 15 MB."}, 413)
                nome = unquote(self.headers.get("X-Nome-Arquivo", "imagem"))[:120]
                try:
                    rgb, tam = carregar_rgb(self.rfile.read(n))
                except Exception:
                    return self._json({"erro": "Não consegui ler o arquivo como imagem (use BMP, PNG ou JPG)."}, 400)
                return self._json(self._diagnosticar(rgb, nome, tam))
            if url.path == "/api/diagnosticar_exemplo":
                caminho = self._exemplo(parse_qs(url.query))
                if caminho is None:
                    return self._json({"erro": "exemplo desconhecido"}, 404)
                rgb, tam = carregar_rgb(caminho)
                return self._json(self._diagnosticar(rgb, caminho.name, tam))
            return self._json({"erro": "rota não encontrada"}, 404)
        except Exception as exc:  # erro inesperado: mostra na interface em vez de derrubar o servidor
            traceback.print_exc()
            return self._json({"erro": f"Falha no diagnóstico: {exc}"}, 500)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--porta", type=int, default=8000)
    ap.add_argument("--sem-navegador", action="store_true")
    args = ap.parse_args()
    print("Carregando o modelo (na primeira vez ele é treinado, leva cerca de 1 minuto)...", flush=True)
    Handler.modelo = Diagnosticador.carregar()
    Handler.exemplos = listar_exemplos()
    servidor = ThreadingHTTPServer(("127.0.0.1", args.porta), Handler)
    endereco = f"http://127.0.0.1:{args.porta}"
    print(f"Interface pronta em {endereco}  (Ctrl+C para encerrar)", flush=True)
    if not args.sem_navegador:
        webbrowser.open(endereco)
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
