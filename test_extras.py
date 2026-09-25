"""Testes das peças acrescentadas depois do MVP. Nenhum precisa do dataset nem da internet."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import avaliacao as av
import relatorio_llm as rl
from modelos import PLS1D, REGRESSORES, MediaDeModelos


def _sintetico(n_por_nivel: int = 20, niveis: int = 7, ruido: float = 0.02, semente: int = 0):
    """α linear em uma feature, mais features de ruído: o regressor ideal é trivial."""
    rng = np.random.default_rng(semente)
    alfas = np.linspace(0, 1, niveis)
    nivel = np.repeat(np.arange(niveis), n_por_nivel)
    alfa = alfas[nivel]
    X = np.column_stack([alfa + rng.normal(0, ruido, len(alfa)), rng.normal(size=(len(alfa), 5))])
    ordem = np.tile(np.arange(n_por_nivel), niveis)
    cat = pd.DataFrame({"nivel": nivel, "alfa": alfa, "pasta": [f"p{k}" for k in nivel], "ordem": ordem,
                        "bloco": ordem * 5 // n_por_nivel, "campanha": np.where(nivel < 4, "A", "B")})
    return X, cat, alfas


def test_niveis_interiores_exclui_as_pontas():
    _, cat, _ = _sintetico()
    assert av.niveis_interiores(cat) == [1, 2, 3, 4, 5]


def test_interpolacao_recupera_relacao_linear():
    X, cat, alfas = _sintetico()
    prob = av.Problema(tuple(f"N{i}" for i in range(7)), tuple(alfas), 600.0)
    r = av.interpolacao(X, cat, "sintetico", problema=prob)
    assert r.mae_espiras.max() < 20          # ruído de 0,02 em α ≈ 10 espiras de erro médio
    assert r.nivel_mais_proximo_correto.min() > 0.9
    assert r.conformal_cobertura.mean() > 0.8


def test_conformal_interior_cobre_severidade_nova():
    X, cat, _ = _sintetico(ruido=0.05)
    modelo = REGRESSORES["ridge"][0].set_params(modelo__alpha=1.0)
    q = av.quantil_conformal_interior(modelo, X, cat.alfa.to_numpy(), cat.nivel.to_numpy())
    assert 0.05 < q < 0.2  # da ordem do ruído, não do intervalo inteiro


@pytest.mark.parametrize("nome", ["ridge", "pls", "pca_ridge", "kernel_ridge", "svr", "gp"])
def test_regressores_candidatos_ajustam_e_preveem(nome):
    X, cat, _ = _sintetico()
    base, grade = REGRESSORES[nome]
    params = {k: v[0] for k, v in grade.items()}
    m = base.set_params(**params).fit(X, cat.alfa)
    assert m.predict(X).shape == (len(X),)


def test_pls1d_devolve_vetor():
    X, cat, _ = _sintetico()
    assert PLS1D(2).fit(X, cat.alfa).predict(X).ndim == 1


def test_media_de_modelos_e_a_media_dos_membros():
    X, cat, _ = _sintetico()
    base = REGRESSORES["ridge"][0].set_params(modelo__alpha=1.0)
    media = MediaDeModelos([(slice(0, 3), base), (slice(3, 6), base)]).fit(X, cat.alfa)
    a = base.fit(X[:, :3], cat.alfa).predict(X[:, :3])
    b = base.fit(X[:, 3:6], cat.alfa).predict(X[:, 3:6])
    np.testing.assert_allclose(media.predict(X), (a + b) / 2)


def test_validar_com_problema_generico():
    X, cat, alfas = _sintetico()
    prob = av.Problema(tuple(f"N{i}" for i in range(7)), tuple(alfas), 100.0)
    ind = pd.DataFrame(np.abs(X[:, :4]), columns=av.INDICADORES_COERENCIA)
    r = av.validar(X, cat, ind, "blocos", "sintetico", problema=prob)
    assert r["confusao"].shape == (7, 7)
    assert r["dobras"].mul_f1_macro.mean() > 0.8


def test_registro_usa_resolucao_informada():
    oof = SimpleNamespace(prob_defeito=0.9, alfa_previsto=0.5, alfa_inferior=0.45, alfa_superior=0.55,
                          confianca_classificador=0.8, score_dominio=0.5, coerencia_visual=0.9)
    ind = SimpleNamespace(fracao_area_quente=0.1, indice_media=0.2, fracao_saturada=0.0)
    r = rl.registro_diagnostico(SimpleNamespace(arquivo="x.png"), oof, ind, resolucao=[640, 480])
    assert r["identificacao"]["resolucao_pixels"] == [640, 480]
    assert rl.verificar(rl.gerar_deterministico(r), r)["aprovado"]


def test_carregar_rgb_redimensiona_e_guarda_tamanho(tmp_path):
    from aplicacao import carregar_rgb
    caminho = tmp_path / "t.png"
    Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8)).save(caminho)
    rgb, tam = carregar_rgb(caminho)
    assert rgb.shape == (240, 320, 3) and tam == (640, 480)
    rgb2, _ = carregar_rgb(caminho.read_bytes())
    assert rgb2.shape == (240, 320, 3)


def test_interface_so_serve_exemplos_listados():
    import importlib.util
    import config
    spec = importlib.util.spec_from_file_location("servidor", config.RAIZ / "interface" / "servidor.py")
    srv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srv)
    h = SimpleNamespace(exemplos=[{"pasta": "p1_Noload", "arquivo": "p1001.bmp"}])
    ok = srv.Handler._exemplo(h, {"pasta": ["p1_Noload"], "arquivo": ["p1001.bmp"]})
    assert ok is not None and ok.name == "p1001.bmp"
    assert srv.Handler._exemplo(h, {"pasta": [".."], "arquivo": ["config.py"]}) is None
    assert srv.Handler._exemplo(h, {"pasta": ["p1_Noload"], "arquivo": ["../../config.py"]}) is None
