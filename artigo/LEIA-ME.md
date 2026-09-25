# Artigo em LaTeX

`artigo.tex` é o texto completo do artigo do XXI ERIAC. Os números citados no texto vêm de
`numeros.tex` e as tabelas de `tabelas/`; ambos são gerados a partir de `saida/` por
`gerar_artigo.py`. Nenhum número é digitado à mão: se os resultados mudarem, rode o gerador
de novo e recompile.

## Compilar no Overleaf (recomendado)

1. Em overleaf.com, **New Project > Upload Project** e escolha `artigo_overleaf.zip`.
2. Em **Menu**, confira: Compiler = pdfLaTeX, Main document = `artigo.tex`.
3. Clique em **Recompile**.

## Compilar localmente

Com uma distribuição TeX (MiKTeX ou TeX Live):

```bash
latexmk -pdf artigo.tex
```

## Regenerar números, tabelas e figuras

Da raiz do projeto, depois de `main.py`, `experimentos.py` e `motor.py`:

```bash
.venv/Scripts/python.exe artigo/gerar_artigo.py
```

## O que ainda depende de você

Os pontos pendentes aparecem em vermelho no PDF, marcados com `TODO`:

- autores, entidade e e-mail;
- 3 a 5 referências específicas de termografia em transformadores e de curto entre espiras
  (há um comentário no topo de `referencias.bib`);
- os resultados dos relatórios reais do LLM (`main.py --llm claude`), na seção 4.7.

Antes de submeter, confira o limite de páginas e o modelo oficial do ERIAC: este arquivo
segue o estilo do rascunho em Word (seções numeradas em caixa alta, tabelas em algarismos
romanos), mas não o modelo oficial, que não estava disponível.
