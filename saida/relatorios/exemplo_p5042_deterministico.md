1) Identificação
Arquivo p5042.bmp; transformador monofásico de bancada; termograma RGB não radiométrico de 320 x 240 pixels.
2) Resultado do diagnóstico
Condição estimada: curto entre espiras (probabilidade de defeito estimada pelo classificador: 0,990).
Severidade estimada: alfa = 0,541, com intervalo de 90% de 0,514 a 0,569. Isso corresponde a cerca de 325 espiras em curto de 600 declaradas (intervalo de 308 a 341). Classe nominal mais próxima: SC320; confiança do classificador multiclasse: 0,917.
3) Controle de qualidade
Dentro do domínio de treinamento: sim (score 0,56; acima de 1,0 indica fora do domínio). Coerência visual relativa: 0,89. Concordância entre módulos: sim. Indicadores relativos: fração de área relativamente quente 0,113; índice médio de paleta 0,155; fração saturada da paleta 0,000.
4) Recomendação
Severidade intermediária. Programar ensaios elétricos confirmatórios e avaliar a retirada programada do equipamento.
Origem: regra determinística com faixas ilustrativas, não normativas.
5) Limitações
- Imagem não radiométrica: as cores indicam posição relativa na paleta da câmera, não temperatura.
- Temperatura por pixel indisponível; nenhum valor de temperatura é calculado.
- Tempo de aquecimento, carga e corrente no laço em curto não foram informados pelo dataset.
- Modelo avaliado apenas no transformador de bancada do dataset, com curtos artificiais.