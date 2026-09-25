# Plano de ação: do MVP ao trabalho completo

Plano de 25/09/2026. Cada fase termina em um commit. O status de cada item fica em
`RELATORIO.md` e no `CHANGELOG.md`.

## Princípio que guia tudo

Nenhuma melhoria entra no artigo sem uma avaliação que não "espie" o teste. Comparar dez
modelos no mesmo teste e reportar o melhor infla o resultado. Por isso, a escolha entre
modelos é feita **dentro** do treino (seleção aninhada), e o número reportado é o do
procedimento completo de escolha, não o do vencedor.

## Fase A: resultados mais fortes e mais bem fundamentados

| # | Item | Pergunta que responde |
|---|---|---|
| A1 | Novos modelos: MobileNet + indicadores (híbrido 100% local), regressões com kernel, PLS, processo gaussiano, combinação de modelos | Dá para errar menos na severidade e na campanha não vistas? |
| A2 | Seleção aninhada por severidade não vista | Qual o erro honesto de "escolher o melhor modelo e aplicar"? |
| A3 | Indicadores robustos a ruído (filtro de mediana antes da paleta) e sem os indicadores de fundo (confundidos com campanha) | O colapso dos indicadores com ruído é corrigível? |
| A4 | Treino com aumento de dados (ruído, desfoque, translação) | A robustez melhora sem piorar o resto? |
| A5 | Intervalos de confiança (bootstrap) e testes pareados entre métodos | As diferenças entre métodos são maiores que o ruído? |
| A6 | Cobertura do intervalo conformal com severidade não vista | O intervalo de 90% continua valendo quando a severidade é nova? |
| A7 | Análise temporal dentro de cada gravação | O α previsto deriva enquanto o equipamento aquece? |
| A8 | Sensibilidade às escolhas (limiar, nº de blocos, purga) | Os resultados dependem de escolhas arbitrárias? |
| A9 | Mesmo pipeline no motor de indução (369 imagens) | O "framework generalista" do resumo se sustenta? |

## Fase B: aplicação

| # | Item |
|---|---|
| B1 | Modelo final treinado e salvo, com `diagnosticar.py` (uma imagem ou uma pasta → JSON, relatório, verificação, CSV) |
| B2 | Interface web local, sem dependências novas: enviar termograma, ver mapa de paleta, α com intervalo, aviso de domínio, recomendação e relatório |
| B3 | Painel atualizado com as análises novas |

## Fase C: artigo

| # | Item |
|---|---|
| C1 | Artigo completo em LaTeX, pronto para o Overleaf, com tabelas geradas automaticamente dos resultados |
| C2 | Bibliografia com referências verificáveis |

## Fase D: engenharia

| # | Item |
|---|---|
| D1 | Testes que rodam sem o dataset (dados sintéticos) e integração contínua no GitHub Actions |
| D2 | Versões fixadas, CHANGELOG, README revisado |

## Fora do alcance desta sessão (dependem de você)

- Rodar `extrair_dinov2.py` (o ambiente bloqueia o `torch.hub`).
- Relatórios reais pelo Claude (falta credencial).
- Compilar o PDF do LaTeX (não há TeX instalado; o Overleaf resolve).
- Escolher a licença do repositório.
