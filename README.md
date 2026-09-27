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

Na barra lateral do dashboard, use **Baixar relatório desta página (PDF)** para exportar a página aberta com os filtros selecionados, gráficos e tabelas correspondentes aos dados exibidos. Os gráficos são desenhados em vetor no PDF. O relatório identifica o período, as safras e, quando aplicável, o estado, o indicador climático e a seleção da avaliação.

As observações brutas baixadas do INMET ficam em `data/raw/inmet/dadoshistoricos`. O painel apresenta 2025 como preliminar e identifica cobertura incompleta de clima. As janelas agronômicas são estimativas exploratórias por macrorregião. A comparação do modelo é retrospectiva: usa clima observado durante o ciclo e área plantada daquele ano, não uma previsão disponível antes da colheita.
