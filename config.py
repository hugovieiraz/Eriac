"""Constantes do experimento. Nada aqui depende de dados: só caminhos, rótulos e escolhas fixadas antes de rodar."""
from __future__ import annotations

from pathlib import Path

RAIZ = Path(__file__).resolve().parent
DATASET = RAIZ / "dados" / "IR_trans_bmp"
REFERENCIA = RAIZ / "referencia_original" / "resultados_transformador"
SAIDA = RAIZ / "saida"
FIGURAS = SAIDA / "figuras"
CACHE = RAIZ / "cache"

SEMENTE = 2026
TOTAL_ESPIRAS = 600

# pasta -> (condição, espiras em curto)
CLASSES = {
    "p1_Noload": ("Healthy", 0),
    "p2_80": ("SC80", 80),
    "p3_160": ("SC160", 160),
    "p4_240": ("SC240", 240),
    "p5_320": ("SC320", 320),
    "p6_400": ("SC400", 400),
    "p7_480": ("SC480", 480),
    "p8_560": ("SC560", 560),
    "p9_600": ("SC600", 600),
}
ORDEM_CLASSES = [c for c, _ in CLASSES.values()]
ALFA_NOMINAL = [n / TOTAL_ESPIRAS for _, n in CLASSES.values()]

# Campanhas de aquisição inferidas da numeração dos arquivos: a numeração continua de uma
# pasta para a seguinte (p2 termina em 026 e p3 começa em 027; p4 -> p5; p6 -> p7 -> p8 -> p9).
# dados.inferir_campanhas() reconstrói isto a partir dos nomes e os testes conferem.
CAMPANHAS_ESPERADAS = {
    "p1_Noload": "A",
    "p2_80": "B", "p3_160": "B",
    "p4_240": "C", "p5_320": "C",
    "p6_400": "D", "p7_480": "D", "p8_560": "D", "p9_600": "D",
}

N_BLOCOS = 5   # blocos temporais contíguos por classe
FILTRO_MEDIANA = 5  # filtro de mediana no mapa de índice de paleta (robustez a ruído)
PURGA = 1      # quadros vizinhos à fronteira do bloco de teste removidos do treino

# Grades de hiperparâmetros, escolhidos por validação interna (aninhada).
GRADE_C = [0.01, 0.1, 1.0, 10.0]
GRADE_RIDGE = [0.1, 1.0, 10.0, 100.0]

# Protocolo original do rascunho (hiperparâmetros fixos), usado só para reprodução.
C_ORIGINAL = 1.0
RIDGE_ORIGINAL = 10.0

NIVEL_INTERVALO = 0.90  # cobertura nominal do intervalo conformal de alfa

# Faixas de recomendação. ILUSTRATIVAS: não vêm de norma e devem ser calibradas com
# especialista antes de qualquer uso. O LLM só redige a ação escolhida por estas regras.
FAIXAS_RECOMENDACAO = [
    # (limite superior de alfa estimado, código, ação)
    (0.05, "sem_indicio",
     "Sem indício visual de curto entre espiras. Manter a periodicidade usual de inspeção."),
    (0.25, "baixa",
     "Indício de curto de baixa severidade. Reduzir o intervalo de inspeção termográfica e "
     "confirmar com ensaios elétricos (relação de transformação, corrente de excitação)."),
    (0.60, "intermediaria",
     "Severidade intermediária. Programar ensaios elétricos confirmatórios e avaliar a "
     "retirada programada do equipamento."),
    (1.01, "elevada",
     "Severidade elevada. Encaminhar para avaliação imediata por especialista e considerar "
     "a retirada de operação."),
]
ACAO_FORA_DOMINIO = (
    "Imagem fora do domínio de treinamento. O resultado numérico não deve ser usado; "
    "repetir a aquisição e encaminhar a especialista."
)
ACAO_INCONSISTENTE = (
    "Resultados inconsistentes entre os módulos de classificação e de severidade. "
    "Encaminhar a especialista antes de qualquer decisão."
)

# Unidades de temperatura que o relatório textual não pode conter: as imagens não são
# radiométricas. Valores elétricos inventados são pegos pela checagem de números, que só
# aceita números presentes no registro estruturado.
UNIDADES_PROIBIDAS = r"[°º]|\d\s*(graus?|celsius|kelvin|fahrenheit|k)\b"

# Temas fora do escopo. Aparecer é permitido só em frase de negação/limitação
# ("não estima vida útil"); fora disso o verificador emite alerta.
TERMOS_FORA_ESCOPO = [
    "vida útil", "vida remanescente", "envelhecimento", "perdas joule",
    "temperatura interna", "temperatura absoluta", "elevação de temperatura",
]
MARCAS_NEGACAO = ["não", "sem ", "indisponív", "desconhecid", "nenhum", "nem "]
