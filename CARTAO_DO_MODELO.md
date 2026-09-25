# Cartão do modelo de aplicação

O modelo usado por `diagnosticar.py` e pela interface (`interface/servidor.py`).

## O que ele faz

Recebe um termograma RGB colorido (BMP, PNG ou JPG) e devolve:

- a condição estimada (curto entre espiras ou sem indício), com probabilidade;
- a severidade α (fração de espiras em curto) e o número de espiras, com intervalo de 90%;
- o nível nominal mais próximo (Healthy, SC80, …, SC600);
- um escore de domínio que diz se a imagem se parece com as de treinamento;
- uma recomendação escolhida por regras e um relatório textual conferido por um verificador.

## Como foi construído

| Item | Valor |
|---|---|
| Dados de treino | 255 termogramas de um transformador monofásico de bancada (Najafi *et al.*, 2020) |
| Features | embeddings MobileNetV3-small (ImageNet, congelada) |
| Treino | regressão logística (condição) e Ridge (α), com hiperparâmetros por validação interna |
| Aumento de dados | cada imagem ganha cópias com ruído, desfoque, translação e rotação |
| Intervalo | conformal de 90%, calibrado retirando cada severidade intermediária do treino (±22 espiras) |
| Domínio | distância aos 5 vizinhos de treino; acima do percentil 99 do treino = fora do domínio |

## Desempenho medido (`RELATORIO.md`)

| Situação | Erro médio em α |
|---|---:|
| Mesma gravação, trecho não visto | 6,8 espiras de 600 |
| Severidade ausente do treino | 13,9 espiras |
| Campanha de gravação ausente do treino | 19,4 espiras |
| Com ruído forte na imagem | 16,5 espiras |

Os três últimos números vêm de uma comparação entre 4 variantes, escolhida olhando esses
mesmos testes; trate-os como estimativas otimistas até que sejam confirmados com dados novos.

## Uso previsto

Pesquisa e demonstração: apoiar a triagem de termogramas do mesmo tipo de equipamento, na
mesma câmera e paleta, indicando quando vale a pena fazer ensaios elétricos confirmatórios.

## Fora do uso previsto

- **Medir temperatura.** As imagens não são radiométricas; o modelo não mede nem estima °C.
- **Decidir sozinho** retirar um equipamento de operação. As faixas de recomendação são
  ilustrativas e não foram calibradas com especialista nem com norma.
- **Outros equipamentos, câmeras ou paletas.** O modelo foi treinado em um único
  transformador de bancada. Imagens diferentes tendem a ser marcadas como fora do domínio,
  mas o detector não é garantia: com severidade nova numa campanha conhecida, ele separa mal
  os acertos dos erros.
- **Transformadores em campo.** Não há nenhuma avaliação em equipamento real em operação.

## Limitações conhecidas

- A condição saudável existe em uma única campanha de gravação; o modelo não tem como
  separar "saudável" de "campanha A".
- Severidades abaixo de 80 ou acima de 600 espiras, e combinações de defeitos, nunca foram
  vistas; o modelo não extrapola.
- O tempo de aquecimento e a carga não são conhecidos; dentro de uma gravação, a imagem
  muda enquanto o equipamento aquece.
- O verificador do relatório confere se cada número existe no diagnóstico, não a que campo
  ele se refere.
