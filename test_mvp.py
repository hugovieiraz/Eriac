"""Testes das peças que, se quebradas, invalidariam os números do artigo sem dar erro."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import avaliacao as av
import config
import dados
import relatorio_llm as rl
from paleta import ConversorPaleta


@pytest.fixture(scope="module")
def catalogo():
    return dados.construir_catalogo()


# ------------------------------------------------------------------ dados ----
def test_blocos_contiguos_e_balanceados():
    assert np.bincount(dados.atribuir_blocos(22)).tolist() == [5, 5, 4, 4, 4]
    b = dados.atribuir_blocos(40)
    assert np.all(np.diff(b) >= 0)  # contíguos: nunca voltam


def test_campanhas_inferidas_da_numeracao(catalogo):
    obtido = catalogo.drop_duplicates("pasta").set_index("pasta").campanha.to_dict()
    assert obtido == config.CAMPANHAS_ESPERADAS


def test_divisao_por_blocos_sem_sobreposicao_e_com_purga(catalogo):
    vistos = []
    for _, tr, te, _ in av.divisoes_externas("blocos", catalogo):
        assert not set(tr) & set(te)
        vistos.extend(te)
        assert set(catalogo.nivel[te]) == set(range(9))  # toda dobra testa todas as classes
        # nenhum quadro de treino é vizinho imediato de um quadro de teste na mesma pasta
        chave = set(zip(catalogo.pasta[te], catalogo.ordem[te]))
        for p, o in zip(catalogo.pasta[tr], catalogo.ordem[tr]):
            assert (p, o - 1) not in chave and (p, o + 1) not in chave
    assert sorted(vistos) == list(range(len(catalogo)))  # cada imagem testada uma vez


# ----------------------------------------------------------------- paleta ----
def test_conversor_paleta_exato_e_vizinho():
    paleta = np.array([[0, 0, 100], [100, 0, 100], [200, 100, 0], [255, 255, 255]], dtype=np.uint8)
    conv = ConversorPaleta(paleta)
    img = np.array([[[0, 0, 100], [255, 255, 255]], [[198, 101, 2], [100, 0, 100]]], dtype=np.uint8)
    np.testing.assert_allclose(conv.indice(img), [[0, 1], [2 / 3, 1 / 3]], atol=1e-6)


# ------------------------------------------------------------- avaliação ----
def test_classe_mais_proxima_respeita_espacamento_desigual():
    # 0,99 fica mais perto de SC600 (1,0) que de SC560 (0,933); round(0,99*7,5)=7 errava.
    assert av.classe_mais_proxima(np.array([0.99]))[0] == 8
    assert av.classe_mais_proxima(np.array([0.95]))[0] == 7
    assert av.classe_mais_proxima(np.array([0.0, 0.534]))[1] == 4


def test_coerencia_so_usa_treino():
    treino = pd.DataFrame(np.ones((20, 4)), columns=av.INDICADORES_COERENCIA)
    treino.iloc[:, 0] = np.arange(20)
    teste = pd.DataFrame([[1e6, 1, 1, 1]], columns=av.INDICADORES_COERENCIA)
    # Um valor absurdo no teste não pode "puxar" o perfil esperado: a coerência cai para ~0.
    c = av.coerencia_visual(treino, np.linspace(0, 1, 20), teste, np.array([0.5]))
    assert c[0] < 1e-3


def test_detector_de_dominio():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    det = av.DetectorDominio().ajustar(X)
    assert np.mean(det.pontuar(X[:50]) > 1) < 0.1
    assert np.all(det.pontuar(X[:5] + 50) > 1)


# ------------------------------------------------------------- relatório ----
class _Oof:
    prob_defeito = 0.99
    alfa_previsto = 0.5401
    alfa_inferior = 0.5132
    alfa_superior = 0.5669
    confianca_classificador = 0.917
    score_dominio = 0.56
    coerencia_visual = 0.89


class _Ind:
    fracao_area_quente = 0.116
    indice_media = 0.156
    fracao_saturada = 0.0


def _registro(**muda):
    oof = _Oof()
    for k, v in muda.items():
        setattr(oof, k, v)
    return rl.registro_diagnostico(pd.Series({"arquivo": "p5042.bmp"}), oof, _Ind())


def test_rotulo_verdadeiro_nao_vai_para_o_llm():
    texto = str(_registro())
    assert "SC320" in texto  # só como classe nominal mais próxima estimada
    assert "condicao_real" not in texto and "alfa_real" not in texto


def test_relatorio_deterministico_passa_no_verificador():
    r = _registro()
    v = rl.verificar(rl.gerar_deterministico(r), r)
    assert v["aprovado"], v


@pytest.mark.parametrize("tipo", ["numero_alterado", "temperatura_inventada", "corrente_inventada",
                                  "espiras_recalculadas", "recomendacao_trocada", "limitacoes_omitidas"])
def test_verificador_pega_mutacoes(tipo):
    r = _registro()
    mutado = rl.mutacoes(rl.gerar_deterministico(r), r)[tipo]
    assert not rl.verificar(mutado, r)["aprovado"]


def test_verificador_aceita_arredondamento_e_percentual():
    r = _registro()
    base = rl.gerar_deterministico(r)
    assert rl.verificar(base + "\nEm resumo, alfa de aproximadamente 0,54 e confiança de 92%.", r)["aprovado"]


def test_limitacao_conhecida_numero_trocado_de_campo_passa():
    """O verificador confere se o número EXISTE no registro, não a que campo se refere.
    Trocar o limite superior do intervalo pelo alfa estimado passa. Documentado no relatório."""
    r = _registro()
    texto = rl.gerar_deterministico(r) + "\nO limite superior do intervalo é 0,540."
    assert rl.verificar(texto, r)["aprovado"]


def test_fora_do_dominio_troca_a_recomendacao():
    r = _registro(score_dominio=1.8)
    assert r["recomendacao"]["codigo"] == "fora_do_dominio"
    assert "fora do dom" in rl.gerar_deterministico(r).lower()


def test_modulos_discordantes_mandam_para_especialista():
    r = _registro(prob_defeito=0.2)  # classificador diz saudável, regressão diz alfa 0,54
    assert r["recomendacao"]["codigo"] == "inconsistente"
