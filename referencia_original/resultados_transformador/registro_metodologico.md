# Registro metodológico e de execução

## Escopo

Este pacote analisa o subconjunto de 255 BMP do transformador do dataset *thermal-images-equip*. Os BMP foram tratados como imagens RGB colorizadas sem radiometria. Nenhum campo, figura ou modelo deste pacote representa temperatura em graus Celsius, elevação de temperatura, corrente de curto, perdas Joule ou temperatura interna do enrolamento.

## Rótulos e cálculos determinísticos

Para cada condição, \(\alpha=N_f/600\). A saída de regressão é limitada ao intervalo \([0,1]\), e o número estimado de espiras é calculado por \(\widehat{N}_f=600\widehat{\alpha}\). A eventual quantidade \(220\widehat{\alpha}\) não é calculada ou reportada como medição: dependeria da hipótese não verificada de distribuição uniforme de 220 V pelas 600 espiras.

## Segmentação e indicadores

O fundo visual é estimado pela mediana CIELAB dos quatro cantos; a ROI é a maior região conectada com distância visual ao fundo acima do percentil 60, com operações morfológicas. Se a máscara for degenerada, aplica-se uma janela central fixa. O limiar de região relativamente aquecida é claridade CIELAB normalizada maior ou igual a 0,65; se menos de cinco pixels o satisfizerem, emprega-se o percentil 90 da ROI. Esses limiares operam na escala visual RGB/Lab e não possuem interpretação em Celsius.

Foram calculados 44 indicadores: roi_area_fraction, hot_area_fraction, visual_intensity_mean, visual_intensity_p50, visual_intensity_p90, visual_intensity_p95, visual_intensity_integrated, hotspot_centroid_x, hotspot_centroid_y, hotspot_distance_center, hotspot_dispersion, visual_intensity_skewness, relative_gradient_mean, relative_gradient_p90, visual_entropy, hot_components, largest_hot_component_fraction, hsv_hue_mean, hsv_saturation_mean, hsv_value_mean, rgb_r_hist_0, rgb_r_hist_1, rgb_r_hist_2, rgb_r_hist_3, rgb_r_hist_4, rgb_r_hist_5, rgb_r_hist_6, rgb_r_hist_7, rgb_g_hist_0, rgb_g_hist_1, rgb_g_hist_2, rgb_g_hist_3, rgb_g_hist_4, rgb_g_hist_5, rgb_g_hist_6, rgb_g_hist_7, rgb_b_hist_0, rgb_b_hist_1, rgb_b_hist_2, rgb_b_hist_3, rgb_b_hist_4, rgb_b_hist_5, rgb_b_hist_6, rgb_b_hist_7.

## Modelos e avaliação

O backbone é DINOv2 ViT-S/14 pré-treinado, congelado, com redimensionamento RGB para 224 x 224 e normalização ImageNet. Foram ajustados regressão logística para tarefas binária e multiclasse e Ridge com penalidade 10 para \(\alpha\). Avaliam-se DINOv2 e DINOv2 mais indicadores relativos em 5 folds estratificados repetidos três vezes, com sementes fixas. Os intervalos de confiança percentis são bootstrap sobre os 15 valores por fold; são descritivos da variação de folds, não intervalos de generalização externa.

O dataset não fornece identificadores de ensaios independentes ou séries temporais. Arquivos com numeração próxima podem ser correlacionados; portanto, a validação cruzada por imagem pode superestimar desempenho em relação a novas campanhas experimentais. O teste de severidade não vista retira uma classe intermediária integralmente e avalia somente a regressão de \(\alpha\), pois um classificador não pode prever de forma válida uma classe ausente no treinamento.

## LLM

Os arquivos JSON são registros de entrada para um agente de relatório. O agente deve apenas explicar valores existentes, identificar suas categorias (observado, fornecido pelo dataset, calculado ou estimado) e repetir limitações. Ele não pode recalcular, alterar predições nem criar medições.
