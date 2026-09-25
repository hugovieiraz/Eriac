# Relatório: MVP do artigo ERIAC (curto entre espiras por termografia)

Execução de 24/09/2026. Python 3.11.9, scikit-learn 1.9.1, torch 2.14 (CPU), semente 2026.
O pipeline completo roda em cerca de 95 s e é determinístico: duas execuções seguidas deram
números idênticos. Todos os números abaixo saem de `saida/`.

---

## Resumo executivo

1. **O resultado original foi reproduzido exatamente.** Com o protocolo do rascunho, o
   híbrido dá F1-macro 0,9970 ± 0,0079 e MAE 0,0103 ± 0,0013 (6,16 espiras), os mesmos
   números da Tabela II.
2. **A validação por blocos temporais quase não derruba os números**: F1 0,996 → 0,982 e
   MAE 5,8 → 6,8 espiras no híbrido. Isso não é boa notícia. **Uma miniatura de 16x12
   pixels em cinza atinge F1 0,974 e MAE 9,8 espiras** no mesmo protocolo, e o DINOv2
   sozinho (10,8 espiras) fica *pior* que ela. Dentro de uma mesma sessão de gravação, o
   problema está no teto e não discrimina métodos.
3. **Os testes que discriminam são a severidade não vista e a campanha não vista.** Com uma
   severidade intermediária inteira fora do treino, o híbrido (DINOv2 + indicadores de
   paleta) é o melhor dos 8 conjuntos: **16,8 espiras de erro médio, 31,7 no pior caso
   (SC240), nível nominal certo em 94% das imagens**. Os indicadores sozinhos erram 43,5 e a
   miniatura trivial erra 34,7. É aqui, e não na validação cruzada, que o DINOv2 se justifica.
4. **O SC320 era o melhor caso.** O número honesto de interpolação é a média de 16,8 a 19,5
   espiras (depende da versão dos indicadores), não 6,2.
5. **As imagens se agrupam em 4 campanhas de gravação**, e o enquadramento da câmera muda
   entre elas. Só a posição da caixa do transformador (4 números, nenhum padrão térmico)
   classifica as 9 condições com F1 0,50, contra 0,11 do acaso. **Retirando uma campanha
   inteira**, o híbrido erra 31,6 espiras em média na interpolação e não consegue
   extrapolar. Em compensação, o detector de domínio marca 70% dessas imagens como "fora do
   domínio", e a camada de relatório troca a recomendação por "encaminhar a especialista".
6. **A condição saudável existe em uma única campanha (A).** Detectar "saudável" é
   indistinguível de detectar "campanha A". A imagem saudável é mais clara que a do SC80.
7. **A camada LLM ficou auditável.** Os números vêm do modelo, a recomendação vem de regras e
   o texto passa por um verificador. Nos 255 relatórios, o verificador pega 99,6% a 100% de
   seis tipos de erro injetados de propósito (número trocado, temperatura inventada, corrente
   inventada, espiras recalculadas, recomendação trocada, limitações omitidas).
8. **Duas coisas não foram executadas aqui.** A chamada à API do Claude não rodou porque não
   há credencial no ambiente; o código está pronto. A re-extração do DINOv2 foi bloqueada pelo
   ambiente, porque o `torch.hub` baixa e executa código do GitHub. Usei os vetores DINOv2 da
   sua execução original, conferidos imagem a imagem por SHA-256 (255/255), e deixei o
   `extrair_dinov2.py` para você rodar.

---

## 1. Plano de ação: cada furo e como foi contornado

| # | Furo no rascunho | Contorno no MVP | Status |
|---|---|---|---|
| 1 | Validação por imagem com quadros quase idênticos dos dois lados | Blocos temporais contíguos por classe, com purga de 1 quadro vizinho no treino; o esquema aleatório fica só como comparação | ✅ |
| 2 | Hiperparâmetros fixos (C=1, Ridge=10) | Validação aninhada: C e α do Ridge escolhidos em validação interna por grupo | ✅ |
| 3 | Sem linha de base nem controle | Miniatura trivial 16x12 e controle de posição (só a caixa do objeto) | ✅ |
| 4 | Sem ablação: não se sabia se o DINOv2 contribuía | 8 conjuntos: trivial, posição, indicadores v1, indicadores de paleta, MobileNetV3, DINOv2, híbrido v1, híbrido de paleta; comparação pareada por dobra | ✅ |
| 5 | SC320 destacado como se fosse típico | Tabela completa das 7 condições retiradas, com média e pior caso, para todos os conjuntos | ✅ |
| 6 | Campanhas de gravação não identificadas | Campanhas inferidas da numeração dos arquivos e teste de retirar uma campanha inteira | ✅ novo |
| 7 | `round(α·7,5)` para achar a classe (classes não são igualmente espaçadas) | Classe de α nominal mais próximo; teste unitário cobre o caso 0,99 → SC600 | ✅ |
| 8 | Coerência visual comparava com o dataset inteiro, incluindo a própria imagem | Coerência calculada só com o treino da dobra | ✅ |
| 9 | Média aritmética de matiz HSV (variável circular) | Substituída pelo índice ordinal de paleta, que dispensa o HSV | ✅ |
| 10 | O rótulo verdadeiro ia no JSON entregue ao LLM | Rótulo removido do registro; um teste garante isso | ✅ |
| 11 | LLM sem prompt versionado, sem modelo registrado, testado em uma imagem | Prompt versionado, backend Claude com metadados, backend determinístico de referência, verificador nas 255 imagens | ✅ (Claude não executado) |
| 12 | O resumo prometia recomendações, o rascunho não tinha | Recomendações por regra determinística; o LLM só transcreve | ✅ (faixas ilustrativas) |
| 13 | Nenhuma medida de incerteza | Intervalo conformal de 90% para α, com cobertura medida | ✅ novo |
| 14 | Nenhuma proteção contra imagem estranha | Detector de domínio (kNN) que troca a recomendação | ✅ novo |
| 15 | Robustez desconhecida | Ruído, desfoque, translação e rotação no teste | ✅ parcial (DINOv2 depende da re-extração) |
| 16 | Explicabilidade | Mapas de oclusão | ✅ |
| 17 | DINOv2 sem versão fixada | Vetores originais conferidos por SHA-256; `extrair_dinov2.py` registra o commit do hub e o hash dos pesos | ⏳ você roda |
| 18 | "Framework generalista" sem teste em outro equipamento | Não feito. O dataset do motor já está no `thermal_fault_lab` | ⏳ próximo passo |

---

## 2. O que foi construído

Pasta `C:\Users\hv392\projetos\eriac_transformador\`:

| Arquivo | Papel |
|---|---|
| `config.py` | Caminhos, rótulos, campanhas esperadas, grades de hiperparâmetros, faixas de recomendação, unidades proibidas |
| `dados.py` | Catálogo com SHA-256, campanhas inferidas, blocos temporais, purga |
| `paleta.py` | Recupera as 253 cores da paleta sem rótulo e converte RGB em índice ordinal |
| `features.py` | Segmentação (a mesma do rascunho), 39 indicadores de paleta, trivial, posição, MobileNetV3, DINOv2, perturbações |
| `avaliacao.py` | Validação aninhada, 4 esquemas (aleatório, blocos, interpolação, campanha), conformal, domínio, coerência, comparação pareada |
| `relatorio_llm.py` | Registro estruturado, regras de recomendação, prompt, backends determinístico e Claude, verificador, mutações |
| `figuras.py` | 10 figuras do artigo |
| `main.py` | Orquestra tudo (`--llm claude`, `--n-llm`, `--sem-robustez`) |
| `extrair_dinov2.py` | Regenera DINOv2 limpo e perturbado, com proveniência |
| `test_mvp.py` | 19 testes; todos passam |

---

## 3. Achados sobre os dados

**Paleta.** As 255 imagens usam exatamente 253 cores, e todas ficam sobre uma única curva no
espaço RGB: a distância máxima de uma cor ao caminho recuperado é 0. A paleta vai de azul
escuro `(5, 0, 94)` a branco `(255, 255, 255)`, passando por magenta, laranja e amarelo. Cada
pixel, portanto, tem um **índice ordinal exato**. Ele é mais defensável que o L do CIELAB (que
não é monótono em todas as paletas) e continua sem nenhuma conversão para °C.

**Campanhas.** A numeração continua de uma pasta para a outra: p2 termina em 026 e p3 começa
em 027. O mesmo vale para p4 → p5 e para p6 → p7 → p8 → p9. Isso dá 4 campanhas:
A = {Healthy}, B = {SC80, SC160}, C = {SC240, SC320} e D = {SC400, SC480, SC560, SC600}. O
enquadramento muda entre elas (veja `fig_indicadores_severidade.png`: o índice do fundo salta
de campanha para campanha).

**A condição saudável é mais clara que o SC80.** O índice máximo médio por imagem é 0,136 no
Healthy e 0,088 no SC80; depois cresce de forma monótona até 0,9995 no SC600. Isso é um
efeito de campanha (ambiente, câmera ou cena), não do defeito.

**Saturação desprezível.** Só o SC600 toca o fim da paleta: 0,03% da ROI em média, no máximo
0,11%.

**Quase-duplicatas.** O vizinho mais próximo de cada imagem no espaço DINOv2 é da mesma classe
em 98,8% dos casos, é o quadro imediatamente adjacente em 35,7% e está a até 3 quadros em
59,2%. O cosseno mediano com o quadro adjacente é 0,980, e com a imagem mais parecida de outra
classe é 0,966.

---

## 4. Resultados

### 4.1 Validação cruzada: aleatória × blocos temporais

Validação aninhada, 5 dobras. No esquema de blocos, cada dobra testa um trecho contíguo de
todas as 9 classes.

| Conjunto | Dim. | F1 aleatória | **F1 blocos** | MAE aleatória (espiras) | **MAE blocos (espiras)** |
|---|---:|---:|---:|---:|---:|
| Trivial 16x12 | 192 | 0,991 | **0,974** | 7,7 | **9,8** |
| Posição (controle) | 4 | 0,605 | **0,499** | 91,2 | **90,9** |
| Indicadores v1 (rascunho) | 44 | 0,995 | **0,995** | 4,7 | **5,5** |
| Indicadores de paleta | 39 | 1,000 | **1,000** | 5,2 | **6,7** |
| MobileNetV3 | 576 | 1,000 | **0,980** | 6,3 | **8,1** |
| DINOv2 | 384 | 0,987 | **0,982** | 9,0 | **10,8** |
| DINOv2 + ind. v1 | 428 | 0,996 | **0,996** | 6,3 | **7,2** |
| **DINOv2 + ind. paleta** | 423 | 0,996 | **0,982** | 5,8 | **6,8** |

Comparação pareada no esquema de blocos (mesmas 5 dobras):

- O híbrido **reduz o MAE em 4,0 espiras em relação ao DINOv2 sozinho, nas 5 dobras**, e
  empata em F1.
- Em relação aos indicadores sozinhos, o híbrido não ganha nada: F1 −0,019 (0 vitórias, 3
  empates) e MAE +0,1 espira.
- **Os indicadores sozinhos superam a miniatura trivial por 3,1 espiras, nas 5 dobras.**
- **O DINOv2 sozinho perde para a miniatura trivial** por 1,0 espira e perde para a MobileNetV3
  nas 5 dobras (por 2,8 espiras).

A classificação binária fica em 1,000 para quase tudo. Como a condição saudável é uma
campanha só (seção 3), isso não sustenta nenhuma conclusão.

### 4.2 Severidade não vista (o teste que discrimina)

Cada condição intermediária sai inteira do treino e só a regressão é avaliada. O
hiperparâmetro é escolhido por validação interna também agrupada por nível.

| Conjunto | MAE médio (espiras) | Pior caso | Nível certo | Fora do domínio |
|---|---:|---|---:|---:|
| Trivial 16x12 | 34,7 | 60,8 (SC240) | 50% | 21% |
| Posição | 114,0 | 243,5 (SC560) | 20% | 2% |
| Indicadores v1 | 33,0 | 80,0 (SC80) | 58% | 75% |
| Indicadores de paleta | 43,5 | 99,0 (SC160) | 51% | 72% |
| MobileNetV3 | 18,0 | 27,7 (SC320) | 92% | 44% |
| DINOv2 | 21,4 | 34,5 (SC240) | 85% | 32% |
| DINOv2 + ind. v1 | 19,5 | 39,7 (SC80) | 91% | 36% |
| **DINOv2 + ind. paleta** | **16,8** | **31,7 (SC240)** | **94%** | 33% |

No híbrido de paleta, o erro por condição retirada é: SC80 28,6 · SC160 16,9 · SC240 31,7 ·
SC320 8,7 · SC400 13,8 · SC480 7,9 · SC560 10,1 espiras.

Os indicadores sozinhos erram feio nas pontas da interpolação: no SC80 preveem α = 0.
Combinados com a representação profunda, dão o melhor resultado. Esta é a evidência que
sustenta o método híbrido no artigo.

Nota sobre o híbrido v1: com Ridge fixo em 10, como no rascunho, a média das 7 condições dá
16,1 espiras; com a escolha aninhada, 19,5. O SC320 continua em 6,2 nos dois casos.

### 4.3 Campanha não vista (o teste mais duro)

Uma campanha inteira sai do treino, levando junto o enquadramento dela. As campanhas B e C
são interpolação; A e D são extrapolação.

| Conjunto | MAE campanha B | MAE campanha C | **MAE médio (B+C)** | Fora do domínio (B+C) |
|---|---:|---:|---:|---:|
| Trivial | 92,1 | 88,7 | 91,0 | 30% |
| Posição | 142,3 | 131,7 | 135,9 | 0% |
| Indicadores de paleta | 78,1 | 25,4 | 52,9 | 100% |
| MobileNetV3 | 46,5 | 18,9 | 33,0 | 100% |
| DINOv2 | 23,5 | 57,2 | 39,6 | 60% |
| **DINOv2 + ind. paleta** | 26,7 | 38,1 | **31,6** | 70% |

Na extrapolação o modelo falha, como esperado. Retirada a campanha A, o saudável é previsto
com α ≈ 0,29. Retirada a campanha D, tudo satura em α = 1. **A trivial desaba de 9,8 para 91
espiras** quando a campanha muda: ela memorizava a sessão. O híbrido perde muito menos. O
detector de domínio marca 60% a 100% dessas imagens nos conjuntos com indicadores ou redes
(só 30% na trivial e 0% no controle de posição). É justamente esse sinal que impede um
relatório confiante sobre uma imagem de outra campanha.

### 4.4 Incerteza, domínio e robustez

**Intervalo conformal de 90%** (híbrido, blocos): **cobertura medida de 0,905**, com meia-largura
média de 0,029 em α (17 espiras). Veja `fig_intervalos_conformais.png`.

**Domínio:** no esquema de blocos, 1,5% das imagens (4 de 255: 2 Healthy e 2 SC80) ficam acima
do limiar.

**Robustez** (modelos treinados em imagens limpas, testados em imagens perturbadas):

| Conjunto | Perturbação | F1 | MAE (espiras) | Fora do domínio |
|---|---|---:|---:|---:|
| Indicadores | nenhuma | 1,000 | 6,7 | 2% |
| Indicadores | ruído σ = 8 | **0,377** | 36,7 | **95%** |
| Indicadores | translação 12 px | 0,995 | 18,7 | 7% |
| MobileNetV3 | nenhuma | 0,980 | 8,1 | 4% |
| MobileNetV3 | ruído σ = 8 | **0,683** | 36,7 | **100%** |
| MobileNetV3 | desfoque σ = 1,5 | **0,584** | 24,8 | **94%** |
| Trivial | translação 12 px | 0,903 | **58,0** | 26% |

Quando a imagem degrada ao ponto de derrubar o modelo, **o detector de domínio avisa em 94% a
100% dos casos**. Tabela completa em `saida/robustez_perturbacoes.csv`.

### 4.5 Oclusão

Entre 77% e 100% da queda do α previsto ao ocultar um quadrado da imagem acontece **dentro da
ROI do transformador**, tanto na MobileNet quanto nos indicadores. Os modelos olham o
enrolamento, não o fundo. Veja `fig_oclusao.png` e `saida/oclusao_resumo.json`.

### 4.6 Camada de relatório (LLM)

Arquitetura:

```
imagem → modelos → registro JSON (sem rótulo verdadeiro)
                 → regra determinística escolhe a recomendação
                 → LLM redige
                 → verificador confere o texto contra o registro
```

- **255 de 255 relatórios determinísticos aprovados** pelo verificador.
- **Sensibilidade do verificador** (255 relatórios × 7 tipos de erro injetado):

| Erro injetado | Reprovado |
|---|---:|
| Temperatura inventada ("87 °C") | 100% |
| Corrente inventada ("3,2 A") | 100% |
| Espiras recalculadas | 100% |
| Recomendação trocada | 100% |
| Limitações omitidas | 100% |
| α alterado | 99,6% |
| "Reduz a vida útil" | 0% reprovado, **100% com alerta** |

- **Limitação conhecida:** o verificador confere se o número *existe* no registro, não a
  *que campo* ele se refere. Um teste documenta isso. O único α alterado que escapou coincidia
  com outro número do registro.
- **Recomendações** (faixas ilustrativas, não normativas): 20 sem indício, 29 baixa,
  78 intermediária, 124 elevada, 4 fora do domínio. Todas as 22 imagens saudáveis caem em "sem
  indício" ou "fora do domínio". O SC160 (α = 0,267) cai em "intermediária" porque o limite
  ilustrativo da faixa baixa é 0,25; o limite precisa ser calibrado com especialista.

Exemplo (`p5042.bmp`, a mesma imagem do rascunho), em `saida/relatorios/`:

> Severidade estimada: alfa = 0,540, com intervalo de 90% de 0,513 a 0,567. Isso corresponde a
> cerca de 324 espiras em curto de 600 declaradas (intervalo de 308 a 340). Classe nominal mais
> próxima: SC320 […] Dentro do domínio de treinamento: sim (score 0,56) […]
> Recomendação: Severidade intermediária. Programar ensaios elétricos confirmatórios […]

Backend Claude: modelo `claude-opus-5`, raciocínio adaptativo e fallback de recusa no servidor
(`fallbacks: "default"`, beta `server-side-fallback-2026-07-01`). Cada relatório é salvo com o
modelo que respondeu, o `request_id`, os tokens e a versão do prompt, e passa pelo mesmo
verificador.

---

## 5. O que muda no artigo

### 5.1 Resumo sugerido (substitui o atual)

> Este trabalho avalia uma abordagem híbrida para identificação de curtos entre espiras e
> estimativa de severidade em um transformador monofásico de bancada a partir de 255
> termogramas RGB não radiométricos (condição saudável e oito níveis artificiais de curto).
> Representações DINOv2 congeladas foram combinadas a indicadores relativos calculados sobre o
> índice ordinal da paleta da câmera, recuperado sem rótulos. A avaliação usa blocos temporais
> com purga, retirada integral de uma severidade e retirada integral de uma campanha de
> aquisição, além de linha de base trivial e controle de enquadramento. Dentro das sessões de
> gravação, até uma miniatura 16x12 pixels atinge F1-macro de 0,974, o que torna esse protocolo
> pouco discriminante. Para severidades ausentes do treinamento, o método híbrido obteve erro
> médio de 16,8 espiras (pior caso 31,7) e acertou o nível nominal em 94% das imagens, contra
> 43,5 espiras dos indicadores isolados e 34,7 da linha de base. Um intervalo conformal de 90%
> apresentou cobertura de 0,905. Um LLM redige o relatório a partir do diagnóstico estruturado,
> com recomendações definidas por regras e verificação automática que detectou 99,6% a 100% dos
> erros injetados. Os resultados não expressam temperatura nem validação entre equipamentos.

### 5.2 Mudanças por seção

- **Seção 2 (Dados):** acrescente as 4 campanhas, a paleta de 253 cores, o fato de a condição
  saudável estar em uma campanha só e o saudável ser mais claro que o SC80.
- **Seção 3 (Metodologia):** índice ordinal de paleta no lugar do L do CIELAB; blocos
  temporais com purga; validação aninhada; trivial e posição; conformal; detector de domínio;
  verificador.
- **Tabela II:** substitua pela tabela 4.1 (com as duas colunas de esquema e os controles).
- **Tabela III:** mostre todas as condições com média e pior caso para pelo menos trivial,
  indicadores, DINOv2 e híbrido (tabela 4.2), e cite o SC320 como exemplo, não como resultado.
- **Nova tabela:** campanha não vista (4.3).
- **Seção 4.4 (LLM):** arquitetura com regras e verificador, e a tabela de sensibilidade.
- **Figuras sugeridas:** `fig_interpolacao.png`, `fig_campanhas.png`,
  `fig_esquemas_validacao.png`, `fig_intervalos_conformais.png`, `fig_oclusao.png`
  (em `saida/figuras/`).
- **Conclusões:** a contribuição do DINOv2 aparece na generalização para severidade não vista,
  não na validação cruzada; a detecção do saudável não é separável da campanha.

### 5.3 Alinhamento com o resumo já enviado

| Promessa do resumo enviado | Situação |
|---|---|
| LLM gera recomendações | ✅ via regras; o LLM só redige, o que é mais defensável |
| "Padrões térmicos" | ⚠️ troque por "padrões térmicos relativos na paleta da câmera" |
| "Framework generalista" | ⚠️ o pipeline é genérico; falta aplicar ao motor (próximo passo) |
| Identificar e estimar a severidade | ✅ com incerteza e aviso de domínio |

---

## 6. O que falta (em ordem de impacto)

1. **Rodar `extrair_dinov2.py`** (alguns minutos em CPU). Completa a robustez do DINOv2 e do
   híbrido e registra o commit e o hash dos pesos.
2. **Rodar `main.py --llm claude --n-llm 20`** com credencial da Anthropic, para a tabela
   "relatórios reais aprovados pelo verificador". É o dado que falta na seção do LLM.
3. **Calibrar as faixas de recomendação** com um especialista ou com a literatura. As de agora
   são ilustrativas.
4. **Aplicar o mesmo pipeline ao motor** (`thermal_fault_lab/dados`), para sustentar o
   "framework generalista".
5. **Referências:** o rascunho tem 2. Faltam termografia em transformadores, detecção de curto
   entre espiras, conformal prediction, detecção de fora do domínio e LLMs em manutenção.
6. **Uma segunda campanha com imagens saudáveis**, se o grupo tiver acesso à bancada. Isso
   resolveria o maior fator de confusão.

---

## 7. Como reproduzir

```bash
cd C:\Users\hv392\projetos\eriac_transformador
../thermal_fault_lab/.venv/Scripts/python.exe -m pytest -q test_mvp.py
../thermal_fault_lab/.venv/Scripts/python.exe main.py
```

Principais arquivos de saída: `resumo_validacao.csv`, `comparacao_pareada_blocos.csv`,
`interpolacao_severidade.csv`, `interpolacao_resumo.csv`, `generalizacao_por_campanha.csv`,
`robustez_perturbacoes.csv`, `reproducao_protocolo_original.csv`, `auditoria_dataset.json`,
`registros_estruturados.json`, `verificacao_relatorios.csv`, `sensibilidade_verificador.csv`,
`previsoes_oof_principal.csv`, `figuras/`, `relatorios/`.
