# Dashboard de produção e clima do milho

Painel exploratório em Streamlit com dados da PAM/IBGE e observações meteorológicas do INMET, agregadas por UF e safra.

## Executar localmente

No PowerShell, a partir da pasta do projeto:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

O painel usa os CSVs preparados em `data/processed`. Para recriar a sequência completa de preparação:

```powershell
python scripts\preparar_ibge_milho.py
python scripts\preparar_inmet_diario.py
python scripts\agregar_clima_uf_safra.py
python scripts\resumir_clima_ciclo_milho.py
python scripts\avaliar_estimativa_milho.py
```

Na página **Mapa interativo por UF**, é possível explorar produção, área plantada,
rendimento, indicadores climáticos e cobertura de dados por estado. O mapa permite
comparar o valor do ano selecionado com o ano anterior ou com a média dos anos
anteriores do período filtrado. Os limites estaduais estão em
`assets/brazil-states.geojson`, com atribuição e licença em `assets/ATTRIBUTION.txt`.

Na barra lateral, **Baixar tabelas CSV (ZIP)** exporta cada tabela filtrada em um
arquivo CSV separado, junto com um arquivo de texto que registra os filtros aplicados.
A exportação acompanha a página e os filtros selecionados. **Baixar relatório desta
página (PDF)** exporta a página aberta com os gráficos e tabelas; os gráficos são
desenhados em vetor.

As observações brutas baixadas do INMET ficam em `data/raw/inmet/dadoshistoricos`. O painel apresenta 2025 como preliminar e identifica cobertura incompleta de clima. As janelas agronômicas são estimativas exploratórias por macrorregião. A comparação do modelo é retrospectiva: usa clima observado durante o ciclo e área plantada daquele ano, não uma previsão disponível antes da colheita.
