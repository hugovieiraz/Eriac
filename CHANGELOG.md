# Histórico de mudanças

## 0.3.0: 25/09/2026

**Resultados**
- Indicadores de paleta passam a usar filtro de mediana 5×5 no mapa de índice. Robustez a
  ruído (F1 0,38 → 0,66) e a translação (18,7 → 6,3 espiras); severidade não vista do
  híbrido 16,8 → 14,8 espiras.
- Novo conjunto no pipeline principal: MobileNetV3 + indicadores (híbrido 100% local).
- Intervalo conformal para severidade nova, calibrado retirando níveis interiores
  (`avaliacao.quantil_conformal_interior`), com cobertura medida.
- `experimentos.py`: 44 combinações de features e regressores, seleção aninhada, testes de
  Wilcoxon e IC bootstrap, robustez com e sem filtro, aumento de dados (robustez e
  generalização), deriva temporal e sensibilidade (limiar, blocos, purga).
- `motor.py`: o mesmo pipeline no motor de indução (classificação de 11 condições e
  severidade do estator).

**Aplicação**
- `aplicacao.py`: modelo MobileNetV3 treinado com aumento de dados, salvo em `modelos/`.
- `diagnosticar.py`: linha de comando para uma imagem ou uma pasta.
- `interface/`: interface web local (só biblioteca padrão), com exemplos do dataset.

**Artigo e documentação**
- `artigo/`: artigo completo em LaTeX com 90 números e 8 tabelas gerados dos resultados,
  bibliografia e zip pronto para o Overleaf.
- Painel com seções de seleção de modelos, aumento de dados, deriva temporal e motor.
- `RELATORIO.md` reescrito; `PLANO.md` com o plano de ação.

**Engenharia**
- `avaliacao.py` parametrizado por `Problema` (classes, α nominal, escala) e por regressor.
- `modelos.py` com os regressores candidatos.
- `test_extras.py`: 15 testes novos com dados sintéticos; testes que precisam do dataset
  são pulados sem ele.
- GitHub Actions rodando os testes a cada envio; `requirements-ci.txt` com versões fixadas.
- `registro_diagnostico` aceita a resolução real da imagem.

## 0.2.0: 24/09/2026
- Painel interativo (`painel/index.html`) e `gerar_painel.py`.

## 0.1.0: 24/09/2026
- MVP: reprodução do protocolo original, índice de paleta, blocos temporais, severidade e
  campanha não vistas, linha de base e controle, conformal, domínio, oclusão, camada de
  relatório com regras e verificador, figuras e relatório.
