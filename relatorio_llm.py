"""Camada de relatório: registro estruturado -> texto técnico -> verificação automática.

Divisão de responsabilidades:
- Os números (condição, alfa, intervalo, confiança, domínio, coerência) vêm dos modelos.
- A recomendação vem de REGRAS determinísticas (config.FAIXAS_RECOMENDACAO).
- O LLM só redige. Ele não recebe a imagem, os pesos, nem o rótulo verdadeiro.
- O verificador confere o texto contra o registro: todo número citado precisa existir no
  registro, nenhuma unidade de temperatura pode aparecer, a ação recomendada tem de ser a da
  regra e as limitações têm de estar presentes.

Backends:
- deterministico: preenche um modelo de texto fixo. Roda sem rede e serve de referência.
- claude: chama a API da Anthropic (requer credencial no ambiente e o pacote `anthropic`).
"""
from __future__ import annotations

import json
import re
import unicodedata

import numpy as np

import config

VERSAO_PROMPT = "2026-09-24.1"
MODELO_CLAUDE = "claude-opus-5"

LIMITACOES = [
    "Imagem não radiométrica: as cores indicam posição relativa na paleta da câmera, não temperatura.",
    "Temperatura por pixel indisponível; nenhum valor de temperatura é calculado.",
    "Tempo de aquecimento, carga e corrente no laço em curto não foram informados pelo dataset.",
    "Modelo avaliado apenas no transformador de bancada do dataset, com curtos artificiais.",
]
# Trecho mínimo que precisa aparecer no texto para cada limitação (comparação sem acento).
MARCAS_LIMITACAO = ["nao radiometric", "temperatura por pixel", "tempo de aquecimento", "bancada"]

PROMPT_SISTEMA = """Você redige relatórios técnicos de inspeção termográfica para engenheiros de manutenção de transformadores.

Você recebe um registro JSON produzido por um pipeline de visão computacional. Todos os valores já foram calculados. Seu papel é apenas explicá-los em português técnico, claro e conciso.

Regras obrigatórias:
1. Use somente informações do registro. Não calcule, não converta e não estime nenhum valor novo.
2. Cite números exatamente como aparecem no registro (arredondar para menos casas decimais é permitido).
3. Nunca mencione temperatura em graus, kelvin ou qualquer unidade térmica: as imagens não são radiométricas.
4. A ação recomendada deve ser transcrita literalmente do campo recomendacao.acao. Não acrescente outras ações.
5. Deixe claro que alfa, espiras e probabilidades são estimativas de modelo, com o intervalo informado.
6. Se controle_de_qualidade.dentro_do_dominio for false ou concordancia_entre_modulos for false, diga isso no início do relatório.
7. Transcreva todas as limitações listadas.

Estrutura: 1) Identificação; 2) Resultado do diagnóstico; 3) Controle de qualidade; 4) Recomendação; 5) Limitações. Sem introdução nem despedida."""


# ------------------------------------------------------------------ regras ----
LIMIAR_DEFEITO_ALFA = config.FAIXAS_RECOMENDACAO[0][0]


def recomendacao(alfa: float, dentro_dominio: bool, concordancia: bool) -> dict:
    """Faixa escolhida pelo alfa estimado. Fora do domínio ou com módulos discordantes, a
    regra não escolhe faixa nenhuma e manda para especialista."""
    if not dentro_dominio:
        return {"codigo": "fora_do_dominio", "acao": config.ACAO_FORA_DOMINIO}
    if not concordancia:
        return {"codigo": "inconsistente", "acao": config.ACAO_INCONSISTENTE}
    for limite, codigo, acao in config.FAIXAS_RECOMENDACAO:
        if alfa < limite:
            return {"codigo": codigo, "acao": acao}
    raise ValueError(alfa)


def registro_diagnostico(cat_linha, oof_linha, ind_linha, resolucao: list[int] | None = None) -> dict:
    """Monta o registro que o LLM recebe. O rótulo verdadeiro NÃO entra aqui."""
    defeito = oof_linha.prob_defeito >= 0.5
    alfa = round(float(oof_linha.alfa_previsto), 3)
    inf, sup = round(float(oof_linha.alfa_inferior), 3), round(float(oof_linha.alfa_superior), 3)
    dentro = bool(oof_linha.score_dominio <= 1.0)
    # Concordância: o classificador binário e a regressão contam a mesma história.
    concordancia = bool(defeito == (alfa >= LIMIAR_DEFEITO_ALFA))
    rec = recomendacao(alfa, dentro, concordancia)
    return {
        "identificacao": {
            "arquivo": cat_linha.arquivo,
            "equipamento": "transformador monofásico de bancada",
            "tipo_de_imagem": "termograma RGB não radiométrico",
            "resolucao_pixels": resolucao or [320, 240],
        },
        "diagnostico": {
            "condicao_estimada": "curto entre espiras" if defeito else "sem indício de curto",
            "probabilidade_de_defeito": round(float(oof_linha.prob_defeito), 3),
            "alfa_estimado": alfa,
            "nivel_do_intervalo_percentual": int(round(config.NIVEL_INTERVALO * 100)),
            "intervalo_alfa": [inf, sup],
            "espiras_totais_declaradas": config.TOTAL_ESPIRAS,
            "espiras_em_curto_estimadas": int(round(alfa * config.TOTAL_ESPIRAS)),
            "intervalo_espiras": [int(round(inf * config.TOTAL_ESPIRAS)), int(round(sup * config.TOTAL_ESPIRAS))],
            "classe_nominal_mais_proxima": config.ORDEM_CLASSES[int(np.abs(np.array(config.ALFA_NOMINAL) - alfa).argmin())],
            "confianca_do_classificador": round(float(oof_linha.confianca_classificador), 3),
        },
        "controle_de_qualidade": {
            "dentro_do_dominio": dentro,
            "score_de_dominio": round(float(oof_linha.score_dominio), 2),
            "limiar_do_score_de_dominio": 1.0,
            "coerencia_visual": round(float(oof_linha.coerencia_visual), 2),
            "concordancia_entre_modulos": concordancia,
        },
        "indicadores_relativos": {
            "fracao_area_relativamente_quente": round(float(ind_linha.fracao_area_quente), 3),
            "indice_medio_de_paleta": round(float(ind_linha.indice_media), 3),
            "fracao_saturada_da_paleta": round(float(ind_linha.fracao_saturada), 3),
        },
        "recomendacao": {**rec, "origem": "regra determinística com faixas ilustrativas, não normativas"},
        "limitacoes": LIMITACOES,
        "proveniencia": {
            "identificacao": "observado",
            "espiras_totais_declaradas": "fornecido pelo dataset",
            "diagnostico": "estimado por modelo",
            "controle_de_qualidade": "calculado",
            "indicadores_relativos": "calculado a partir da imagem",
            "recomendacao": "regra determinística",
        },
    }


# --------------------------------------------------------------- backends ----
def _fmt(x: float, casas: int) -> str:
    return f"{x:.{casas}f}".replace(".", ",")


def gerar_deterministico(r: dict) -> str:
    d, q, ind, rec = r["diagnostico"], r["controle_de_qualidade"], r["indicadores_relativos"], r["recomendacao"]
    alertas = []
    if not q["dentro_do_dominio"]:
        alertas.append("ATENÇÃO: a imagem está fora do domínio de treinamento do modelo.")
    if not q["concordancia_entre_modulos"]:
        alertas.append("ATENÇÃO: os módulos de classificação e de severidade discordam.")
    linhas = [*alertas,
              "1) Identificação",
              f"Arquivo {r['identificacao']['arquivo']}; {r['identificacao']['equipamento']}; "
              f"{r['identificacao']['tipo_de_imagem']} de {r['identificacao']['resolucao_pixels'][0]} x "
              f"{r['identificacao']['resolucao_pixels'][1]} pixels.",
              "2) Resultado do diagnóstico",
              f"Condição estimada: {d['condicao_estimada']} (probabilidade de defeito estimada pelo classificador: "
              f"{_fmt(d['probabilidade_de_defeito'], 3)}).",
              f"Severidade estimada: alfa = {_fmt(d['alfa_estimado'], 3)}, com intervalo de "
              f"{d['nivel_do_intervalo_percentual']}% de {_fmt(d['intervalo_alfa'][0], 3)} a {_fmt(d['intervalo_alfa'][1], 3)}. "
              f"Isso corresponde a cerca de {d['espiras_em_curto_estimadas']} espiras em curto de "
              f"{d['espiras_totais_declaradas']} declaradas (intervalo de {d['intervalo_espiras'][0]} a "
              f"{d['intervalo_espiras'][1]}). Classe nominal mais próxima: {d['classe_nominal_mais_proxima']}; "
              f"confiança do classificador multiclasse: {_fmt(d['confianca_do_classificador'], 3)}.",
              "3) Controle de qualidade",
              f"Dentro do domínio de treinamento: {'sim' if q['dentro_do_dominio'] else 'não'} "
              f"(score {_fmt(q['score_de_dominio'], 2)}; acima de {_fmt(q['limiar_do_score_de_dominio'], 1)} "
              f"indica fora do domínio). "
              f"Coerência visual relativa: {_fmt(q['coerencia_visual'], 2)}. "
              f"Concordância entre módulos: {'sim' if q['concordancia_entre_modulos'] else 'não'}. "
              f"Indicadores relativos: fração de área relativamente quente {_fmt(ind['fracao_area_relativamente_quente'], 3)}; "
              f"índice médio de paleta {_fmt(ind['indice_medio_de_paleta'], 3)}; "
              f"fração saturada da paleta {_fmt(ind['fracao_saturada_da_paleta'], 3)}.",
              "4) Recomendação",
              rec["acao"],
              f"Origem: {rec['origem']}.",
              "5) Limitações",
              *[f"- {lim}" for lim in r["limitacoes"]]]
    return "\n".join(linhas)


def gerar_claude(r: dict) -> tuple[str, dict]:
    """Chama a API da Anthropic. Fallback de recusa ativado no servidor (fallbacks="default")."""
    import anthropic

    cliente = anthropic.Anthropic()
    resposta = cliente.beta.messages.create(
        model=MODELO_CLAUDE,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        extra_body={"fallbacks": "default"},
        thinking={"type": "adaptive"},
        system=PROMPT_SISTEMA,
        messages=[{"role": "user", "content": json.dumps(r, ensure_ascii=False, indent=2)}],
    )
    meta = {"modelo_solicitado": MODELO_CLAUDE, "modelo_que_respondeu": resposta.model,
            "stop_reason": resposta.stop_reason, "request_id": resposta._request_id,
            "tokens_entrada": resposta.usage.input_tokens, "tokens_saida": resposta.usage.output_tokens,
            "versao_prompt": VERSAO_PROMPT}
    if resposta.stop_reason == "refusal":
        return "", meta
    texto = "\n".join(b.text for b in resposta.content if b.type == "text")
    return texto, meta


def gerar_claude_seguro(r: dict) -> tuple[str, dict]:
    """Como gerar_claude, mas uma falha da API vira um relatório vazio com o erro anotado
    (o verificador o reprova) em vez de interromper o lote. O SDK já repete sozinho erros
    transitórios (429, 5xx, conexão) duas vezes antes de chegar aqui."""
    import anthropic

    base = {"modelo_solicitado": MODELO_CLAUDE, "versao_prompt": VERSAO_PROMPT}
    try:
        return gerar_claude(r)
    except anthropic.AuthenticationError:
        return "", {**base, "erro": "credencial ausente ou inválida"}
    except anthropic.RateLimitError:
        return "", {**base, "erro": "limite de requisições atingido"}
    except anthropic.APIStatusError as e:
        return "", {**base, "erro": f"erro da API ({e.status_code})"}
    except anthropic.APIConnectionError:
        return "", {**base, "erro": "falha de conexão"}


# -------------------------------------------------------------- verificador ----
_NUMERO = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)?(?![\w])")
_NUMERACAO = re.compile(r"^\s*(#+\s*)?\d+[.)]?\s", re.MULTILINE)


def _sem_acento(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", t.lower()) if unicodedata.category(c) != "Mn")


def _numeros_do_registro(r: dict) -> list[float]:
    valores: list[float] = []

    def andar(x):
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)):
            valores.append(float(x))
        elif isinstance(x, dict):
            for v in x.values():
                andar(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                andar(v)
        elif isinstance(x, str):
            for m in _NUMERO.findall(x):
                valores.append(float(m.replace(",", ".")))

    andar(r)
    return valores


def _numero_permitido(texto_num: str, permitidos: list[float], percentual: bool = False) -> bool:
    valor = float(texto_num.replace(",", "."))
    casas = len(texto_num.split(",")[-1]) if "," in texto_num else (
        len(texto_num.split(".")[-1]) if "." in texto_num else 0)
    for p in permitidos:
        # 0,95 pode aparecer como 95%, mas só quando o texto traz o sinal de porcentagem.
        for candidato in ((p, p * 100) if percentual else (p,)):
            if abs(round(candidato, casas) - valor) < 1e-9 or abs(candidato - valor) < 1e-9:
                return True
    return False


def verificar(texto: str, r: dict) -> dict:
    """Confere o texto contra o registro. `aprovado` só é verdadeiro sem nenhuma falha."""
    falhas, alertas = [], []
    if not texto.strip():
        return {"aprovado": False, "falhas": ["texto vazio (recusa ou erro)"], "alertas": []}
    corpo = _NUMERACAO.sub(" ", texto)
    permitidos = _numeros_do_registro(r)
    for m in _NUMERO.finditer(corpo):
        percentual = corpo[m.end():m.end() + 2].lstrip().startswith("%")
        if not _numero_permitido(m.group(), permitidos, percentual):
            falhas.append(f"número ausente do registro: {m.group()}")
    if re.search(config.UNIDADES_PROIBIDAS, texto, flags=re.IGNORECASE):
        falhas.append("unidade de temperatura no texto")
    normal = _sem_acento(re.sub(r"\s+", " ", texto))
    if _sem_acento(re.sub(r"\s+", " ", r["recomendacao"]["acao"])) not in normal:
        falhas.append("ação recomendada diferente da regra")
    for marca in MARCAS_LIMITACAO:
        if marca not in normal:
            falhas.append(f"limitação ausente: '{marca}'")
    q = r["controle_de_qualidade"]
    if not q["dentro_do_dominio"] and "fora do dominio" not in normal:
        falhas.append("não avisou que a imagem está fora do domínio")
    for frase in re.split(r"(?<=[.!?\n])\s+", texto):
        f = _sem_acento(frase)
        for termo in config.TERMOS_FORA_ESCOPO:
            if _sem_acento(termo) in f and not any(_sem_acento(m) in f for m in config.MARCAS_NEGACAO):
                alertas.append(f"tema fora do escopo afirmado: '{termo}'")
    return {"aprovado": not falhas, "falhas": falhas, "alertas": alertas}


# ------------------------------------------------ teste de sensibilidade ----
def mutacoes(texto: str, r: dict) -> dict[str, str]:
    """Erros típicos de LLM injetados de propósito, para medir se o verificador os pega."""
    d = r["diagnostico"]
    alfa_txt = _fmt(d["alfa_estimado"], 3)
    # Desloca para o lado que não satura em 0 ou 1 (1,000 coincidiria com o limiar do score).
    sinal = -1 if d["alfa_estimado"] > 0.5 else 1
    alfa_errado = _fmt(d["alfa_estimado"] + sinal * 0.137, 3)
    espiras_erradas = d["espiras_em_curto_estimadas"] + sinal * 43
    outras = [a for _, _, a in config.FAIXAS_RECOMENDACAO if a != r["recomendacao"]["acao"]]
    corte = texto.find("5) Limitações")
    return {
        "numero_alterado": texto.replace(alfa_txt, alfa_errado, 1),
        "temperatura_inventada": texto + "\nA região mais quente atinge aproximadamente 87 °C.",
        "corrente_inventada": texto + "\nA corrente estimada no laço em curto é de 3,2 A.",
        "espiras_recalculadas": texto + f"\nPor proporção, estimam-se {espiras_erradas} espiras afetadas.",
        "recomendacao_trocada": texto.replace(r["recomendacao"]["acao"], outras[0]),
        "limitacoes_omitidas": texto[:corte] if corte > 0 else texto,
        "vida_util_afirmada": texto + "\nO defeito reduz a vida útil do equipamento de forma significativa.",
    }
