# Relatório: curto entre espiras por termografia (versão 2)

25/09/2026. A versão 1 (MVP) está no histórico do Git. Esta versão acrescenta a busca e a
seleção de modelos, indicadores robustos, aumento de dados, estatística, análise temporal,
sensibilidade, o teste no motor de indução, a aplicação (linha de comando e interface), o
artigo em LaTeX e a integração contínua. O que mudou está em `CHANGELOG.md`.

Todos os números vêm de `saida/` e são reproduzíveis (semente 2026, execuções
determinísticas). Ambiente: Python 3.11.9, scikit-learn 1.9.1, torch 2.14 (CPU).

---

## Resumo executivo

1. **O resultado do rascunho foi reproduzido exatamente**: F1-macro 0,9970 e MAE de 6,16
   espiras com o protocolo original.
2. **Dentro de uma sessão de gravação, o problema está no teto.** Uma miniatura de 16×12
   pixels em cinza atinge F1 0,974 e erro de 9,8 espiras; o DINOv2 sozinho erra mais (10,8).
   Esse teste não distingue métodos.
3. **O teste que distingue é a severidade não vista.** O híbrido DINOv2 + indicadores de
   paleta erra **14,8 espiras de 600** em média (IC 95%: 9,6 a 20,5), com pior caso de 25,8,
   e acerta o nível nominal em **97%** das imagens. Ele supera o DINOv2 sozinho nas 7
   severidades (p = 0,016) e os indicadores sozinhos em 6 de 7 (p = 0,031).
4. **Buscar o "melhor modelo" não ajudou.** Entre 44 combinações de features e regressores,
   a seleção aninhada (que escolhe sem ver o teste) erra 18,5 espiras, mais que o híbrido
   definido de antemão (14,8).
5. **Campanha de gravação nova é o limite do dataset.** Com uma campanha inteira fora do
   treino, o híbrido erra 33,6 espiras e não extrapola. O detector de domínio marca 67% dessas
   imagens, e o relatório passa a mandar para especialista.
6. **Duas melhorias de robustez se confirmaram.** O filtro de mediana no índice de paleta
   levou o F1 dos indicadores com ruído de 0,38 para 0,66. O aumento de dados levou a
   MobileNet, com ruído, de 36,7 para 16,5 espiras e, de forma exploratória, reduziu o erro
   com campanha nova de 33,0 para 19,4.
7. **O pipeline funciona sem ajustes no motor de indução**: F1 0,997 nas 11 condições. Mas lá
   o teste de severidade não vista não discrimina métodos, porque o controle de posição erra
   menos que as redes (3,8 contra 4,1 pontos percentuais).
8. **A camada de LLM é auditável**: o verificador detecta de 99,2% a 100% dos erros
   injetados. **Os relatórios reais do Claude ainda não foram gerados** (falta credencial).
9. **Há uma ferramenta pronta**: `diagnosticar.py` e uma interface web local, com o modelo
   MobileNet treinado com aumento de dados e intervalo de ±22 espiras calibrado para
   severidade nova.

---

## 1. O que o dataset revela

| Achado | Número | Consequência |
|---|---|---|
| A paleta da câmera tem 253 cores numa única curva | distância máxima ao caminho: 0 | cada pixel vira um índice ordinal exato (sem °C) |
| A numeração dos arquivos revela 4 campanhas | A = saudável; B = 80, 160; C = 240, 320; D = 400 a 600 | enquadramento e fundo mudam entre campanhas |
| O saudável é mais claro que o SC80 | índice máximo 0,136 contra 0,088 | detectar o saudável = detectar a campanha A |
| Quadros vizinhos são quase idênticos | vizinho mais próximo da mesma classe em 98,8% | a divisão aleatória infla os números |
| Saturação no topo da paleta | 0,03% da região no SC600 | desprezível |
| O equipamento aquece ou esfria durante algumas gravações | Spearman até 0,91 | mas o α previsto varia no máximo 18 espiras |

---

## 2. Resultados

### 2.1 Validação dentro das sessões

Validação aninhada, 5 dobras. F1-macro das 9 condições; MAE em espiras de 600.

| Features | F1 aleatória | **F1 blocos** | MAE aleatória | **MAE blocos** |
|---|---:|---:|---:|---:|
| Miniatura 16×12 (linha de base) | 0,991 | **0,974** | 7,7 | **9,8** |
| Posição do objeto (controle) | 0,605 | **0,499** | 91,2 | **90,9** |
| Indicadores v1 (rascunho) | 0,995 | **0,995** | 4,7 | **5,5** |
| Indicadores de paleta | 1,000 | **1,000** | 4,0 | **5,3** |
| MobileNetV3 | 1,000 | **0,980** | 6,3 | **8,1** |
| DINOv2 | 0,987 | **0,982** | 9,0 | **10,8** |
| DINOv2 + indicadores v1 | 0,996 | **0,996** | 6,3 | **7,2** |
| **DINOv2 + indicadores de paleta** | 0,996 | **0,982** | 5,7 | **6,9** |
| MobileNetV3 + indicadores de paleta | 1,000 | **0,980** | 5,2 | **6,9** |

Intervalo conformal de 90% do híbrido: cobertura 0,929, meia-largura de 17 espiras.

### 2.2 Severidade não vista (o teste principal)

Cada severidade intermediária sai inteira do treino.

| Features | MAE médio | Pior caso | Nível certo | Cobertura IC 90% |
|---|---:|---|---:|---:|
| Miniatura 16×12 | 34,7 | 60,8 (SC240) | 49% | 89% |
| Posição (controle) | 114,0 | 243,5 (SC560) | 20% | 86% |
| Indicadores v1 | 33,0 | 80,0 (SC80) | 58% | 86% |
| Indicadores de paleta | 38,8 | 80,0 (SC80) | 56% | 86% |
| MobileNetV3 | 18,0 | 27,7 (SC320) | 92% | 91% |
| DINOv2 | 21,4 | 34,5 (SC240) | 85% | 84% |
| DINOv2 + indicadores v1 | 19,5 | 39,7 (SC80) | 91% | 83% |
| **DINOv2 + indicadores de paleta** | **14,8** | **25,8 (SC80)** | **97%** | **91%** |
| MobileNetV3 + indicadores de paleta | 17,0 | 22,9 (SC320) | 98% | 93% |

Híbrido, por severidade retirada: SC80 25,8 · SC160 14,2 · SC240 25,3 · SC320 6,0 ·
SC400 12,0 · SC480 6,7 · SC560 13,6 espiras. O intervalo de 90% para severidade nova (±37
espiras em média) cobriu 91% dos casos.

**Testes pareados (Wilcoxon sobre os 7 níveis; o menor p possível é 0,016):**

| Comparação | Diferença média | Níveis em que o 1º vence | p |
|---|---:|---:|---:|
| Híbrido × DINOv2 | −6,6 espiras | 7 de 7 | 0,016 |
| Híbrido × indicadores | −24,0 espiras | 6 de 7 | 0,031 |
| MobileNet + ind. × híbrido | +2,2 espiras | 3 de 7 | 0,69 (sem diferença detectável) |

### 2.3 Busca e seleção de modelos

Foram avaliadas 44 combinações de 7 conjuntos de features e 6 regressores (Ridge, PLS,
PCA + Ridge, Kernel Ridge, SVR, processo gaussiano), mais 2 médias de modelos.

| Critério | Erro na severidade nova |
|---|---:|
| Melhor combinação, escolhida olhando o teste (otimista) | 13,6 espiras |
| **Seleção aninhada** (escolhe sem ver a severidade retirada) | **18,5 espiras** |
| Híbrido definido de antemão | 14,8 espiras |

A seleção aninhada escolheu 5 combinações diferentes ao longo dos 7 níveis. Com tão poucos
níveis, buscar o melhor modelo não melhora de forma confiável. Os regressores com núcleo
(Kernel Ridge, SVR) extrapolam mal (acima de 41 espiras). A média de DINOv2 e MobileNet foi a
melhor com campanha nova (19,5 espiras), mas isso é leitura do teste, não resultado
confirmado.

### 2.4 Campanha não vista

| Features | Camp. B | Camp. C | **B + C** | Fora do domínio |
|---|---:|---:|---:|---:|
| Miniatura 16×12 | 92,1 | 88,7 | **91,0** | 33% |
| Posição (controle) | 142,3 | 131,7 | **135,9** | 4% |
| Indicadores de paleta | 59,3 | 51,2 | **54,5** | 99% |
| MobileNetV3 | 46,5 | 18,9 | **33,0** | 100% |
| DINOv2 | 23,5 | 57,2 | **39,6** | 62% |
| **DINOv2 + indicadores de paleta** | 29,2 | 39,6 | **33,6** | 67% |
| MobileNetV3 + indicadores de paleta | 46,2 | 16,0 | **31,3** | 100% |

Extrapolação: sem a campanha A, o saudável é previsto com α ≈ 0,27; sem a campanha D, tudo
satura em α = 1.

### 2.5 Robustez e aumento de dados

**Filtro de mediana no índice de paleta** (adotado como padrão):

| Indicadores | F1 limpa | F1 com ruído | Erro com translação |
|---|---:|---:|---:|
| Sem filtro | 1,000 | 0,38 | 18,7 espiras |
| **Com filtro 5×5** | 1,000 | **0,66** | **6,3 espiras** |

**Aumento de dados** (treino com cópias mais fracas que as perturbações do teste):

| Modelo | Limpa | Ruído | Desfoque | Transl. | Rotação | Sev. nova | Camp. nova |
|---|---:|---:|---:|---:|---:|---:|---:|
| MobileNetV3, sem aumento | 8,1 | 36,7 | 24,8 | 14,5 | 12,5 | 18,0 | 33,0 |
| **MobileNetV3, com aumento** | 6,8 | 16,5 | 8,2 | 9,6 | 8,1 | **13,9** | **19,4** |
| MobileNetV3 + ind., sem aumento | 6,9 | 34,0 | 20,1 | 13,9 | 9,8 | 17,0 | 31,3 |
| MobileNetV3 + ind., com aumento | 4,1 | 6,3 | 4,5 | 4,9 | 4,2 | 19,2 | 34,7 |

Com aumento, a MobileNet ganha robustez e também generaliza melhor para severidade e campanha
novas; a combinação com indicadores fica muito robusta dentro da sessão, mas generaliza pior.
Como são 4 variantes escolhidas olhando esses testes, o ganho da MobileNet é exploratório.
Foi por isso que ela virou o modelo de aplicação (seção 3), e não o número principal.

### 2.6 Sensibilidade e explicabilidade

- **Limiar da região quente:** 0,65 e 0,75 dão resultados iguais; 0,55 piora a severidade nova.
- **Blocos e purga:** com 3, 5 ou 8 blocos e purga de 0, 1 ou 3 quadros, o erro dentro das
  sessões fica entre 6,1 e 9,0 espiras.
- **Oclusão:** na MobileNet, 78% a 85% da queda de α ocorre ao ocultar o transformador. Nos
  indicadores, 47% a 100%: no SC600, parte do efeito vem de fora da região segmentada.

### 2.7 Motor de indução (segundo equipamento)

369 imagens, 11 condições. A paleta também foi recuperada sem rótulos (242 cores). Severidade
do estator = fração total de espiras em curto (hipótese nossa: % por fase × fases ÷ 3).

| Features | F1 (11 cond.) | MAE blocos (p.p.) | MAE sev. nova (p.p.) | Pior (p.p.) |
|---|---:|---:|---:|---:|
| Miniatura 16×12 | 0,942 | 1,47 | 6,0 | 17,5 |
| Posição (controle) | 0,542 | 4,45 | **3,8** | 8,8 |
| Indicadores de paleta | 0,992 | 1,31 | 5,1 | 10,6 |
| MobileNetV3 | 0,988 | 0,97 | 4,1 | 10,1 |
| MobileNetV3 + indicadores | **0,997** | **0,85** | 4,4 | 10,7 |

No motor, a numeração é uma única sequência que acompanha a severidade, e o enquadramento
muda aos poucos: o controle de posição interpola tão bem quanto as redes. Então o "framework
generalista" se sustenta para classificação e severidade dentro das sessões, não para
generalização. O pior caso (A50, curto de 50% em uma fase) é previsto como mais severo do que
a fração de espiras indica, o que sugere que o aquecimento não é linear nessa fração.

### 2.8 Camada de relatório

Os 255 relatórios gerados pelo modelo de texto fixo foram aprovados pelo verificador.
Recomendações: 20 sem indício, 29 baixa, 78 intermediária, 122 elevada, 6 fora do domínio.

| Erro injetado (255 relatórios cada) | Detectado |
|---|---:|
| Temperatura inventada ("87 °C") | 100% |
| Corrente inventada ("3,2 A") | 100% |
| Espiras recalculadas | 100% |
| Recomendação trocada | 100% |
| Limitações omitidas | 100% |
| α alterado | 99,2% |
| "Reduz a vida útil" | alerta em 100% |

Limitação conhecida: o verificador confere se cada número **existe** no registro, não a que
campo ele se refere.

---

## 3. Aplicação

| Peça | Uso |
|---|---|
| `diagnosticar.py` | uma imagem ou uma pasta → JSON, relatório `.md`, `resumo.csv` |
| `interface/servidor.py` | interface web local: arrastar um termograma, ver α com intervalo, índice de paleta, região quente, recomendação, relatório e verificador |
| `aplicacao.py` | treina, salva (`modelos/`) e carrega o modelo |

Modelo: MobileNetV3-small com aumento de dados; intervalo de 90% de ±22 espiras, calibrado
retirando níveis de severidade. Testes feitos: um exemplo do SC320 dá α = 0,539 (323
espiras); um termograma do **motor** enviado ao modelo do transformador foi marcado **fora
do domínio**, e a recomendação virou "encaminhar a especialista"; um arquivo que não é imagem
gera mensagem de erro clara; tentativas de ler arquivos fora da pasta de exemplos são
bloqueadas.

## 4. Artigo em LaTeX

`artigo/artigo.tex` traz o texto completo, com 8 tabelas, 5 figuras e 19 referências.
Todos os números do texto (90 macros) vêm de `artigo/numeros.tex`, gerado dos resultados.
Para compilar, envie `artigo/artigo_overleaf.zip` ao Overleaf (veja `artigo/LEIA-ME.md`).
Não havia TeX instalado para compilar aqui: o arquivo passou por uma checagem estática
(chaves e ambientes balanceados, macros definidas, citações presentes no `.bib`, colunas das
tabelas), mas **a primeira compilação no Overleaf pode apontar algum ajuste**.

## 5. Engenharia

- 34 testes (`test_mvp.py`, `test_extras.py`). Os que precisam do dataset são pulados sem
  ele; os demais usam dados sintéticos.
- GitHub Actions (`.github/workflows/testes.yml`): a cada envio, instala as dependências,
  baixa o dataset do espelho oficial e roda os testes.
- `requirements-ci.txt` com versões fixadas.

## 6. O que depende de você

1. **Rodar `extrair_dinov2.py`** (o `torch.hub` foi bloqueado aqui). Isso completa a
   robustez do DINOv2 e do híbrido e registra o commit e o hash dos pesos.
2. **Rodar `main.py --llm claude --n-llm 20`** com credencial da Anthropic. É o dado que
   falta na seção de relatórios do artigo.
3. **Completar o artigo**: autores, 3 a 5 referências do setor e os resultados do LLM
   (marcados em vermelho com `TODO`). Conferir o modelo oficial e o limite de páginas do ERIAC.
4. **Calibrar as faixas de recomendação** com um especialista.
5. **Escolher a licença** do repositório.
6. Se houver acesso à bancada: **uma segunda campanha com o transformador saudável**. É o que
   resolveria o maior fator de confusão.

## 7. Reproduzir

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe main.py
.venv/Scripts/python.exe experimentos.py
.venv/Scripts/python.exe motor.py
.venv/Scripts/python.exe gerar_painel.py
.venv/Scripts/python.exe artigo/gerar_artigo.py
```
