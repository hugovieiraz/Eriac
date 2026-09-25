# Eriac: curto entre espiras por termografia

Código, resultados e artigo do trabalho *Diagnóstico da Severidade de Curtos entre Espiras em
Transformadores: Uma Solução Baseada em Termografia, Visão Computacional e LLMs* (XXI ERIAC,
CE A2). A partir de termogramas RGB não radiométricos de um transformador de bancada, o
pipeline estima se há curto e quantas espiras estão em curto, com intervalo de incerteza,
aviso de imagem fora do domínio e relatório técnico verificado.

| Quero... | Onde |
|---|---|
| Ler os resultados | [`RELATORIO.md`](RELATORIO.md) |
| Explorar os resultados | `painel/index.html` (abre com duplo clique, sem internet) |
| Diagnosticar um termograma | `interface/servidor.py` ou `diagnosticar.py` |
| Compilar o artigo | [`artigo/LEIA-ME.md`](artigo/LEIA-ME.md) (pronto para o Overleaf) |
| Ver o que mudou | [`CHANGELOG.md`](CHANGELOG.md) e [`PLANO.md`](PLANO.md) |
| Saber para que o modelo serve e não serve | [`CARTAO_DO_MODELO.md`](CARTAO_DO_MODELO.md) |

## Instalação

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Baixe os dados do transformador (o `tar` do Windows 11 abre `.rar`):

```bash
mkdir -p dados/IR_trans_bmp
curl -L -o dados/IR_trans_bmp.rar https://raw.githubusercontent.com/mohnaj-nit/thermal-images-equip/master/IR_trans_bmp.rar
tar -xf dados/IR_trans_bmp.rar -C dados/IR_trans_bmp
```

Para o teste no motor de indução (opcional):

```bash
mkdir -p dados/IR_Motor_bmp
curl -L -o dados/IR-Motor-bmp.rar https://raw.githubusercontent.com/mohnaj-nit/thermal-images-equip/master/IR-Motor-bmp.rar
tar -xf dados/IR-Motor-bmp.rar -C dados/IR_Motor_bmp
```

Cite os dados: M. Najafi, Y. Baleghi e S. M. Mirimani, Mendeley Data, 2020,
DOI 10.17632/8mg8mkc7k5.1 (transformador) e 10.17632/m4sbt8hbvk.1 (motor). As imagens não
vão para este repositório.

## Usar em um termograma novo

Interface no navegador (só a biblioteca padrão do Python; escuta apenas em 127.0.0.1):

```bash
.venv/Scripts/python.exe interface/servidor.py
```

Linha de comando, para uma imagem ou uma pasta (grava JSON, relatório e `resumo.csv`):

```bash
.venv/Scripts/python.exe diagnosticar.py pasta_de_imagens/ --saida resultados/
```

No primeiro uso, o modelo é treinado com as 255 imagens (cerca de 1 minuto) e salvo em
`modelos/`. O modelo de aplicação é a MobileNetV3 treinada com aumento de dados, a receita
local mais robusta nos experimentos; o intervalo de 90% é calibrado para severidade nova.

## Reproduzir os resultados

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe main.py
.venv/Scripts/python.exe experimentos.py
.venv/Scripts/python.exe motor.py
.venv/Scripts/python.exe gerar_painel.py
.venv/Scripts/python.exe artigo/gerar_artigo.py
```

`main.py` leva cerca de 90 s em CPU, `experimentos.py` cerca de 10 min e `motor.py` cerca de
1 min. Tudo sai em `saida/`. As execuções são determinísticas (semente 2026).

Opcionais, que dependem de você:

- `main.py --llm claude --n-llm 20` gera relatórios pelo Claude (requer `pip install anthropic`
  e uma credencial da Anthropic no ambiente) e passa cada um pelo verificador.
- `extrair_dinov2.py` regenera os vetores DINOv2 (limpos, perturbados e de aumento de dados).
  Ele baixa e executa o código do repositório `facebookresearch/dinov2` via `torch.hub`.
  Enquanto não for rodado, o pipeline usa os vetores DINOv2 da execução original, conferidos
  por SHA-256 em `referencia_original/`.

## Arquivos

| Arquivo | Papel |
|---|---|
| `config.py` | Caminhos, rótulos, campanhas, grades, filtro de mediana, faixas de recomendação |
| `dados.py` | Catálogo, campanhas inferidas da numeração, blocos temporais, purga |
| `paleta.py` | Recupera a paleta da câmera sem rótulos e converte RGB em índice ordinal |
| `features.py` | Segmentação, 39 indicadores de paleta, trivial, posição, MobileNet, DINOv2, perturbações |
| `modelos.py` | Regressores candidatos (Ridge, PLS, PCA+Ridge, Kernel Ridge, SVR, processo gaussiano, médias) |
| `avaliacao.py` | Validação aninhada, 4 esquemas de divisão, conformal, domínio, coerência |
| `relatorio_llm.py` | Registro estruturado, regras, backends determinístico e Claude, verificador |
| `main.py` | Pipeline principal (8+1 conjuntos de features) |
| `experimentos.py` | Seleção de modelos, seleção aninhada, robustez, aumento de dados, estatística, deriva, sensibilidade |
| `motor.py` | O mesmo pipeline no motor de indução |
| `aplicacao.py` | Modelo de aplicação: treina, salva e diagnostica imagens novas |
| `diagnosticar.py` | Linha de comando |
| `interface/` | Interface web local |
| `gerar_painel.py`, `painel/` | Painel interativo dos resultados |
| `artigo/` | Artigo em LaTeX, tabelas e números gerados dos resultados |
| `figuras.py` | Figuras do artigo |
| `extrair_dinov2.py` | Regenera DINOv2 (manual) |
| `test_mvp.py`, `test_extras.py` | 34 testes; os que precisam do dataset são pulados sem ele |
| `.github/workflows/testes.yml` | Roda os testes no GitHub a cada envio, baixando o dataset |
