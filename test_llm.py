"""Testes do backend Claude com um SDK simulado: conferem o que é enviado à API e como a
resposta, a recusa e os erros são tratados, sem rede nem credencial."""
from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

import pytest

import relatorio_llm as rl


class _Erro(Exception):
    pass


class _StatusErro(_Erro):
    def __init__(self, status_code=500):
        super().__init__("erro")
        self.status_code = status_code


def _sdk_falso(resposta=None, erro=None):
    chamadas = []

    class Mensagens:
        def create(self, **kw):
            chamadas.append(kw)
            if erro is not None:
                raise erro
            return resposta

    class Anthropic:
        def __init__(self):
            self.beta = SimpleNamespace(messages=Mensagens())

    mod = types.ModuleType("anthropic")
    mod.Anthropic = Anthropic
    mod.APIConnectionError = type("APIConnectionError", (_Erro,), {})
    mod.APIStatusError = _StatusErro
    mod.RateLimitError = type("RateLimitError", (_StatusErro,), {})
    mod.AuthenticationError = type("AuthenticationError", (_StatusErro,), {})
    return mod, chamadas


def _resposta(texto: str, stop="end_turn"):
    return SimpleNamespace(model="claude-opus-5", stop_reason=stop, _request_id="req_teste",
                           usage=SimpleNamespace(input_tokens=900, output_tokens=300),
                           content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=texto)])


def _registro():
    oof = SimpleNamespace(prob_defeito=0.99, alfa_previsto=0.5401, alfa_inferior=0.5132, alfa_superior=0.5669,
                          confianca_classificador=0.917, score_dominio=0.56, coerencia_visual=0.89)
    ind = SimpleNamespace(fracao_area_quente=0.116, indice_media=0.156, fracao_saturada=0.0)
    return rl.registro_diagnostico(SimpleNamespace(arquivo="p5042.bmp"), oof, ind)


def test_requisicao_e_resposta(monkeypatch):
    r = _registro()
    texto_ok = rl.gerar_deterministico(r)
    sdk, chamadas = _sdk_falso(_resposta(texto_ok))
    monkeypatch.setitem(sys.modules, "anthropic", sdk)
    texto, meta = rl.gerar_claude(r)
    assert texto == texto_ok and rl.verificar(texto, r)["aprovado"]
    kw = chamadas[0]
    assert kw["model"] == "claude-opus-5"
    assert kw["thinking"] == {"type": "adaptive"}
    assert "server-side-fallback-2026-07-01" in kw["betas"] and kw["extra_body"] == {"fallbacks": "default"}
    assert kw["system"] == rl.PROMPT_SISTEMA
    enviado = json.loads(kw["messages"][0]["content"])
    assert enviado == r  # o registro vai inteiro, e só ele (sem rótulo verdadeiro)
    assert meta["request_id"] == "req_teste" and meta["modelo_que_respondeu"] == "claude-opus-5"


def test_recusa_vira_relatorio_vazio_reprovado(monkeypatch):
    sdk, _ = _sdk_falso(_resposta("", stop="refusal"))
    monkeypatch.setitem(sys.modules, "anthropic", sdk)
    r = _registro()
    texto, meta = rl.gerar_claude(r)
    assert texto == "" and meta["stop_reason"] == "refusal"
    assert not rl.verificar(texto, r)["aprovado"]


@pytest.mark.parametrize("erro, trecho", [("AuthenticationError", "credencial"), ("RateLimitError", "limite"),
                                          ("APIConnectionError", "conexão")])
def test_erro_da_api_nao_interrompe_o_lote(monkeypatch, erro, trecho):
    sdk, _ = _sdk_falso()
    excecao = getattr(sdk, erro)() if erro != "APIConnectionError" else sdk.APIConnectionError("x")
    sdk_com_erro, _ = _sdk_falso(erro=excecao)
    for nome in ("APIConnectionError", "APIStatusError", "RateLimitError", "AuthenticationError"):
        setattr(sdk_com_erro, nome, getattr(sdk, nome))
    monkeypatch.setitem(sys.modules, "anthropic", sdk_com_erro)
    texto, meta = rl.gerar_claude_seguro(_registro())
    assert texto == "" and trecho in meta["erro"]


def test_texto_do_llm_com_temperatura_e_reprovado(monkeypatch):
    r = _registro()
    sdk, _ = _sdk_falso(_resposta(rl.gerar_deterministico(r) + "\nO enrolamento atinge 95 °C."))
    monkeypatch.setitem(sys.modules, "anthropic", sdk)
    texto, _ = rl.gerar_claude(r)
    v = rl.verificar(texto, r)
    assert not v["aprovado"] and "unidade de temperatura no texto" in v["falhas"]
