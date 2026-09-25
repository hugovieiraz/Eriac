# eriac_transformador

Pipeline reprodutível do artigo *Diagnóstico da Severidade de Curtos entre Espiras em
Transformadores* (XXI ERIAC, CE A2). Reescreve `analise_transformador.py`, fecha os furos
metodológicos do rascunho e acrescenta a camada de relatório com verificação automática.

## Como rodar

Crie um ambiente virtual e instale as dependências (torch em CPU é suficiente):

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Na máquina do autor, o ambiente virtual do `thermal_fault_lab` já tem tudo; troque
`.venv/Scripts/python.exe` por `../thermal_fault_lab/.venv/Scripts/python.exe`. Depois de
baixar os dados (seção abaixo):

```bash
.venv/Scripts/python.exe -m pytest -q test_mvp.py
```

```bash
.venv/Scripts/python.exe main.py
```

A execução completa leva poucos minutos em CPU. Tudo sai em `saida/`.

Relatórios pelo Claude (requer `pip install anthropic` e uma credencial da Anthropic no
ambiente, como `ANTHROPIC_API_KEY`):

```bash
.venv/Scripts/python.exe main.py --llm claude --n-llm 20
```

Regenerar os vetores DINOv2 (baixa e executa o código do repositório `facebookresearch/dinov2`
via `torch.hub`; rode conscientemente). Depois disso, o `main.py` usa esses vetores e inclui
DINOv2 e o modelo híbrido no teste de robustez:

```bash
.venv/Scripts/python.exe extrair_dinov2.py
```

## Dados

`dados/IR_trans_bmp/`: 255 BMP do transformador, espelho oficial dos autores
(`github.com/mohnaj-nit/thermal-images-equip`, arquivo `IR_trans_bmp.rar`). Os SHA-256 das
255 imagens batem com o catálogo da execução original. As imagens não vão para o repositório;
para baixar e extrair (o `tar` do Windows 11 abre `.rar`):

```bash
mkdir -p dados/IR_trans_bmp
curl -L -o dados/IR_trans_bmp.rar https://raw.githubusercontent.com/mohnaj-nit/thermal-images-equip/master/IR_trans_bmp.rar
tar -xf dados/IR_trans_bmp.rar -C dados/IR_trans_bmp
```

Cite o dataset: M. Najafi, Y. Baleghi e S. M. Mirimani, *Thermal images_1-phase_dry
type_Transformer*, Mendeley Data, v1, 2020, DOI 10.17632/8mg8mkc7k5.1.

`referencia_original/`: resultados da execução original do rascunho (zip do Drive). Daqui
vêm os vetores DINOv2 (conferidos imagem a imagem por SHA-256) e os 44 indicadores v1.

## Arquivos

| Arquivo | Papel |
|---|---|
| `config.py` | Caminhos, rótulos, campanhas, grades de hiperparâmetros, faixas de recomendação |
| `dados.py` | Catálogo, campanhas inferidas da numeração, blocos temporais, purga |
| `paleta.py` | Recupera as 253 cores da paleta e converte RGB em índice ordinal |
| `features.py` | trivial, posição, indicadores (paleta), MobileNet, DINOv2, perturbações |
| `avaliacao.py` | Validação aninhada, esquemas de divisão, conformal, domínio, coerência |
| `relatorio_llm.py` | Registro estruturado, regras de recomendação, backends, verificador |
| `figuras.py` | Figuras do artigo |
| `main.py` | Orquestra tudo |
| `extrair_dinov2.py` | Regenera DINOv2 (manual) |
| `test_mvp.py` | 18 testes das peças que invalidariam números em silêncio |

Os resultados e a discussão estão em `RELATORIO.md`.
