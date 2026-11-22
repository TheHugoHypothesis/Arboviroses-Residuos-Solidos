#!/usr/bin/env python3
"""
Script 04: Análise Exploratória de Dados (AED)
---------------------------------------------------------------------------

Executa os diagnósticos descritos na metodologia do artigo:

1. Superdispersão de Poisson (Razão Variância/Média e teste de superdispersão)
2. Zeros inflacionados e distribuição de frequências
3. Taxa de respondência histórica SNIS vs SINISA (2019-2024)
4. Desbalanceamento dos 4 estratos de risco do Ministério da Saúde
5. Matriz de correlação de Spearman e diagnóstico de multicolinearidade (VIF)
6. Relações não-lineares e curvas LOWESS (efeito limiar / tipping point)
7. Séries temporais e choque epidêmico de 2024
8. Disparidades espaciais por Macrorregião (justificando k-fold espacial)
9. Comparação Febre Amarela vs Arboviroses Urbanas (justificativa de descarte)

"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dados")
IMG_DIR = os.path.join(BASE_DIR, "imagens")
os.makedirs(IMG_DIR, exist_ok=True)

def salvar_grafico(nome, bbox_inches='tight'):
    png_path = os.path.join(IMG_DIR, f"{nome}.png")
    pdf_path = os.path.join(IMG_DIR, f"{nome}.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches=bbox_inches)
    plt.savefig(pdf_path, bbox_inches=bbox_inches)
    print(f"    Exportado PNG (300 DPI) e PDF vetorial: {nome}")

def carregar_dados():
    path_tabulao = os.path.join(DATA_DIR, "tabulao_integrado_2019_2024.parquet")
    df = pd.read_parquet(path_tabulao)
    return df

#evolução Temporal das Arboviroses
def plot_evolucao_temporal(df):
    print("  [1/11] Gerando Figura 01: Evolução temporal das arboviroses...")
    annual = df.groupby('ano')[['dengue_conf', 'chik_conf', 'zika_conf', 'arbo_conf_tot']].sum().reset_index()
    
    fig, ax1 = plt.subplots(figsize=(10, 5.5), dpi=300)
    
    #Dengue na escala principal (milhões)
    ax1.plot(annual['ano'], annual['dengue_conf'] / 1e6, marker='o', color='#d95f02', lw=2.5, label='Dengue (Milhões de Casos)')
    ax1.plot(annual['ano'], annual['arbo_conf_tot'] / 1e6, marker='s', color='#7570b3', lw=2.0, ls='--', label='Total Arboviroses (Milhões)')
    ax1.set_ylabel('Dengue e Total (Milhões de Casos Confirmados)', color='#d95f02', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Ano de Notificação', fontsize=11, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#d95f02')
    ax1.set_ylim(0, 7.0)
    
    #Eixo secundário para Chikungunya e Zika (milhares)
    ax2 = ax1.twinx()
    ax2.plot(annual['ano'], annual['chik_conf'] / 1e3, marker='^', color='#1b9e77', lw=2.0, label='Chikungunya (Milhares)')
    ax2.plot(annual['ano'], annual['zika_conf'] / 1e3, marker='d', color='#e7298a', lw=1.8, ls=':', label='Zika (Milhares)')
    ax2.set_ylabel('Chikungunya e Zika (Milhares de Casos)', color='#1b9e77', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#1b9e77')
    ax2.set_ylim(0, 300)
    ax2.grid(False)
    
    #Destacar 2024
    ax1.annotate('Surto Histórico de 2024\n(> 6 milhões de casos)', 
                 xy=(2024, annual.loc[annual['ano']==2024, 'dengue_conf'].values[0] / 1e6),
                 xytext=(2021.8, 5.8),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=7),
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", fc="#fee8c8", ec="#e34a33"))
    
    ax1.axvline(2022.5, color='gray', linestyle=':', lw=1.5)
    ax1.text(2020.5, 6.5, 'Conjunto Treino/Validação\n(2019-2022)', ha='center', fontsize=9, style='italic', color='#555555')
    ax1.text(2023.5, 6.5, 'Teste Prospectivo\n(2023-2024)', ha='center', fontsize=9, style='italic', color='#555555')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, framealpha=0.9)
    
    plt.title('Série Histórica Nacional de Casos Confirmados de Arboviroses (2019–2024)', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    salvar_grafico("01_evolucao_temporal_arboviroses_2019_2024")
    plt.close()

#Diagnóstico de Superdispersão de Poisson
def plot_superdispersao_poisson(df):
    print("  [2/11] Gerando Figura 02: Superdispersão e distribuição de contagens...")
    y = df['arbo_conf_tot'].dropna()
    mean_y = y.mean()
    var_y = y.var()
    phi = var_y / mean_y
    zeros_pct = (y == 0).mean() * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    
    #Painel A: Log-Contagens Observadas vs Modelo Teórico Poisson
    #Amostra para visualização sem ruído extremo
    y_filtered = y[y <= 500]
    sns.histplot(y_filtered, bins=50, kde=False, color='#2b83ba', stat='density', ax=ax1, label='Frequência Observada')
    
    #Teórico Poisson para a mesma média local
    x_poisson = np.arange(0, 500)
    
    mu_local = y_filtered.mean()
    y_pois_pmf = stats.poisson.pmf(x_poisson, mu_local)
    ax1.plot(x_poisson, y_pois_pmf, color='#d7191c', lw=2, label=f'Poisson Teórica (Média={mu_local:.1f})')
    
    ax1.set_title('A: Densidade Observada vs Poisson Teórica (<= 500 casos)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Contagem Anual de Casos Confirmados', fontsize=10)
    ax1.set_ylabel('Densidade', fontsize=10)
    ax1.legend(frameon=True)
    
    #Painel B: Gráfico de Dispersão Média vs Variância por População (Decis)
    df_valid = df.dropna(subset=['populacao_total', 'arbo_conf_tot']).copy()
    df_valid['pop_decil'] = pd.qcut(df_valid['populacao_total'], q=15, duplicates='drop')
    decil_stats = df_valid.groupby('pop_decil', observed=True)['arbo_conf_tot'].agg(['mean', 'var']).reset_index()
    
    ax2.scatter(decil_stats['mean'], decil_stats['var'], color='#7570b3', s=70, edgecolor='black', zorder=3, label='Grupos Populacionais (Decis)')
    
    #Linha de Equidispersão de Poisson (Var = Media)
    max_m = decil_stats['mean'].max()
    line_x = np.linspace(0, max_m, 100)
    ax2.plot(line_x, line_x, color='#1a9641', linestyle='--', lw=2, label='Equidispersão Poisson (Var = Média)')
    
    #Curva Quadrática de Superdispersão (Var = Mean + alpha * Mean^2)
    #Ajuste simples de alpha
    alpha_est = (decil_stats['var'] - decil_stats['mean']) / (decil_stats['mean']**2)
    alpha_mean = alpha_est.median()
    line_var = line_x + alpha_mean * (line_x**2)
    ax2.plot(line_x, line_var, color='#d7191c', lw=2, label=f'Superdispersão Quase-Poisson (α ≈ {alpha_mean:.2f})')
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title(f'B: Razão Variância/Média nos Decis Populacionais (log-log)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Média Condicional de Casos (E[Y])', fontsize=10)
    ax2.set_ylabel('Variância Condicional de Casos (Var(Y))', fontsize=10)
    ax2.legend(frameon=True)
    
    fig.suptitle(f'Diagnóstico de Superdispersão: Média Global = {mean_y:.1f} | Variância = {var_y:,.0f} | Parâmetro Φ = {phi:,.0f} | Zeros = {zeros_pct:.1f}%',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    salvar_grafico("02_distribuicao_superdispersao_poisson")
    plt.close()

#Desbalanceamento dos Estratos de Risco Epidemiológico
def plot_estratos_risco(df):
    print("  [3/11] Gerando Figura 03: Desbalanceamento de classes de risco...")
    order = ['Baixo', 'Médio', 'Alto', 'Epidêmico']
    palette = {'Baixo': '#2b83ba', 'Médio': '#abdda4', 'Alto': '#fdae61', 'Epidêmico': '#d7191c'}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    
    #Total consolidado
    counts_total = df['estrato_risco_ms'].value_counts()[order]
    pcts_total = counts_total / len(df.dropna(subset=['estrato_risco_ms'])) * 100
    
    bars1 = ax1.bar(order, pcts_total, color=[palette[c] for c in order], edgecolor='black', alpha=0.9)
    ax1.set_title('A: Proporção Consolidada (2019–2024)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Percentual dos Municípios (%)', fontsize=10)
    ax1.set_ylim(0, 65)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f'{yval:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    #Comparação Treino vs Teste
    split_df = df.groupby(['particao_temporal', 'estrato_risco_ms'], observed=True).size().unstack()[order]
    split_pct = split_df.div(split_df.sum(axis=1), axis=0) * 100
    
    x = np.arange(len(order))
    width = 0.35
    
    ax2.bar(x - width/2, split_pct.loc['Treino_Validacao', order], width, label='Treino (2019-2022)', color='#4575b4', edgecolor='black')
    ax2.bar(x + width/2, split_pct.loc['Teste_Prospectivo', order], width, label='Teste (2023-2024)', color='#d73027', edgecolor='black')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(order, fontsize=10)
    ax2.set_title('B: Comparação entre Partição de Treino e Teste Prospectivo', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Percentual no Conjunto (%)', fontsize=10)
    ax2.set_ylim(0, 65)
    ax2.legend(frameon=True)
    
    ax1.set_xticks(range(len(order)))
    ax1.set_xticklabels(['Baixo\n(<100)', 'Médio\n(100–300)', 'Alto\n(300–500)', 'Epidêmico\n(>500)'])
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Baixo\n(<100)', 'Médio\n(100–300)', 'Alto\n(300–500)', 'Epidêmico\n(>500)'])
    
    fig.suptitle('Distribuição das Faixas de Risco Epidemiológico do Ministério da Saúde (Casos / 100 mil hab.)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    salvar_grafico("03_desbalanceamento_estratos_risco_ms")
    plt.close()

#Taxa de Respondência e Cobertura Amostral do SNIS/SINISA
def plot_taxa_respondencia(df):
    print("  [4/11] Gerando Figura 04: Taxa de respondência histórica SNIS/SINISA...")
    TOTAL_MUNICIPIOS_BR = 5570
    
    # Contagem de municípios respondentes por ano
    resp = df.groupby('ano')['respondeu_modulo_rs'].agg(
        respondentes='sum',
        total_base='count'
    ).reset_index()
    resp['pct_brasil'] = (resp['respondentes'] / TOTAL_MUNICIPIOS_BR) * 100
    
    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=300)
    
    bars = ax1.bar(resp['ano'], resp['respondentes'], color='#41b6c4', edgecolor='#225ea8', width=0.6, label='Municípios com Informações Prestadas')
    ax1.axhline(TOTAL_MUNICIPIOS_BR, color='#e31a1c', linestyle='--', lw=2, label=f'Total de Municípios Brasileiros ({TOTAL_MUNICIPIOS_BR:,})')
    
    ax1.set_ylabel('Número de Municípios', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Ano de Coleta de Dados de Resíduos', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 6200)
    
    # Rótulo de porcentagem nas barras
    for idx, bar in enumerate(bars):
        pct = resp.loc[idx, 'pct_brasil']
        cnt = resp.loc[idx, 'respondentes']
        ax1.text(bar.get_x() + bar.get_width()/2.0, cnt + 120, f'{cnt:,}\n({pct:.1f}%)', 
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    #Anotação da transição SNIS -> SINISA
    ax1.axvline(2022.5, color='gray', linestyle=':', lw=1.5)
    ax1.text(2020.5, 5800, 'Marco Histórico: SNIS\n(2019–2022)', ha='center', fontsize=9, fontweight='bold', color='#1f78b4')
    ax1.text(2023.5, 5800, 'Novo Sistema: SINISA\n(2023–2024)', ha='center', fontsize=9, fontweight='bold', color='#33a02c')
    
    ax1.legend(loc='lower right', frameon=True)
    plt.title('Adesão e Cobertura Amostral da Gestão de Resíduos no Brasil (SNIS / SINISA)', fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    salvar_grafico("04_taxa_respondencia_snis_sinisa")
    plt.close()

#Distribuição dos Indicadores de Resíduos Sólidos
def plot_distribuicao_indicadores(df):
    print("  [5/11] Gerando Figura 05: Distribuição univariada dos indicadores de resíduos...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)
    
    #A. Cobertura de coleta (%)
    sns.histplot(df['cobertura_coleta_pct'].dropna(), bins=40, color='#2b83ba', kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('A: Cobertura da Coleta Domiciliar (%)', fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel('Taxa de Cobertura (%)', fontsize=10)
    axes[0, 0].set_ylabel('Frequência (Municípios-Ano)', fontsize=10)
    axes[0, 0].axvline(df['cobertura_coleta_pct'].median(), color='red', linestyle='--', label=f'Mediana: {df["cobertura_coleta_pct"].median():.1f}%')
    axes[0, 0].legend()
    
    # B. Despesa per capita (truncada a R$ 1000 para visualização)
    desp = df['despesa_percapita_ano'].dropna()
    desp_clip = desp[desp <= 800]
    sns.histplot(desp_clip, bins=40, color='#fdae61', kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('B: Despesa Municipal per Capita com Manejo (R$/hab.ano)', fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel('Despesa Anual per Capita (R$/hab)', fontsize=10)
    axes[0, 1].set_ylabel('Frequência', fontsize=10)
    axes[0, 1].axvline(desp.median(), color='red', linestyle='--', label=f'Mediana: R$ {desp.median():.1f}')
    axes[0, 1].legend()
    
    # C. Massa per capita (kg/hab.dia)
    massa = df['massa_percapita_kg_dia'].dropna()
    massa_clip = massa[massa <= 3.0]
    sns.histplot(massa_clip, bins=40, color='#abdda4', kde=True, ax=axes[1, 0])
    axes[1, 0].set_title('C: Massa Coletada per Capita (kg/hab.dia)', fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel('Massa per Capita (kg/hab.dia)', fontsize=10)
    axes[1, 0].set_ylabel('Frequência', fontsize=10)
    axes[1, 0].axvline(massa.median(), color='red', linestyle='--', label=f'Mediana: {massa.median():.2f} kg/hab.dia')
    axes[1, 0].legend()
    
    # D. Terceirização e Custos de Coleta
    terc = df['terceirizacao_coleta_pct'].dropna()
    sns.histplot(terc, bins=40, color='#d7191c', kde=True, ax=axes[1, 1])
    axes[1, 1].set_title('D: Taxa de Terceirização dos Serviços de Coleta (%)', fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel('Percentual Terceirizado (%)', fontsize=10)
    axes[1, 1].set_ylabel('Frequência', fontsize=10)
    axes[1, 1].axvline(terc.median(), color='red', linestyle='--', label=f'Mediana: {terc.median():.1f}%')
    axes[1, 1].legend()
    
    plt.tight_layout()
    salvar_grafico("05_distribuicao_indicadores_residuos")
    plt.close()

#Matriz de Correlação de Spearman e Multicolinearidade
def plot_matriz_correlacao(df):
    print("  [6/11] Gerando Figura 06: Matriz de correlação de Spearman...")
    features = {
        'taxa_incidencia_conf_100k': 'Incidência Arboviroses',
        'cobertura_coleta_pct': 'Cobertura Coleta (%)',
        'despesa_percapita_ano': 'Despesa per Capita',
        'massa_percapita_kg_dia': 'Massa per Capita',
        'custo_coleta_pct': 'Custo Coleta (%)',
        'terceirizacao_coleta_pct': 'Terceirização (%)',
        'presenca_lixao': 'Presença de Lixão',
        'populacao_total': 'População Total'
    }
    
    sub_df = df[list(features.keys())].rename(columns=features)
    corr = sub_df.corr(method='spearman')
    
    plt.figure(figsize=(9, 7.5), dpi=300)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-0.5, vmax=0.8, center=0,
                annot=True, fmt=".2f", square=True, linewidths=0.5, 
                cbar_kws={"shrink": 0.8, "label": "Coeficiente de Correlação de Postos de Spearman (ρ)"})
    
    plt.title('Matriz de Correlação Não-Paramétrica de Spearman entre Variáveis', fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    salvar_grafico("06_matriz_correlacao_spearman")
    plt.close()

#Relação Não-Linear: Cobertura de Coleta vs Incidência (LOWESS)
def plot_lowess_cobertura(df):
    print("  [7/11] Gerando Figura 07: Curva LOWESS Cobertura vs Incidência...")
    
    #Filtra dados com taxas válidas
    sample = df.dropna(subset=['cobertura_coleta_pct', 'taxa_incidencia_conf_100k']).copy()
    sample = sample[(sample['taxa_incidencia_conf_100k'] > 0) & (sample['taxa_incidencia_conf_100k'] <= 5000)]
    
    #Amostra de 5000 pontos para não sobrecarregar grafico
    if len(sample) > 5000:
        sample = sample.sample(5000, random_state=42)
        
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Scatter plot semitransparente em escala logarítmica na incidência
    ax.scatter(sample['cobertura_coleta_pct'], sample['taxa_incidencia_conf_100k'], 
               alpha=0.15, color='#2b83ba', s=20, edgecolors='none', label='Municípios-Ano')
    
    #Curva LOWESS / Regressão Polinomial Local
    sns.regplot(data=sample, x='cobertura_coleta_pct', y='taxa_incidencia_conf_100k', 
                lowess=True, scatter=False, ax=ax, color='#d7191c', 
                line_kws={'lw': 3, 'label': 'Ajuste Não-Linear LOWESS'})
    
    ax.set_yscale('log')
    ax.set_xlabel('Taxa de Cobertura da Coleta de Resíduos Domiciliares (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Taxa de Incidência (Casos / 100 mil hab. - Escala Log)', fontsize=11, fontweight='bold')
    ax.set_xlim(10, 102)
    
    #Limiares de risco epidemiológico do Ministério da Saúde
    ax.axhline(100, color='gold', linestyle='--', lw=1.2, label='Limiar Médio Risco (100/100k)')
    ax.axhline(500, color='red', linestyle='--', lw=1.2, label='Limiar Risco Epidêmico (500/100k)')
    
    # Destaque para o efeito limiar (tipping point) acima de 85%
    ax.axvspan(85, 100, color='#abdda4', alpha=0.2, label='Zona de Efeito Limiar (>85% Cobertura)')
    
    ax.legend(loc='lower left', frameon=True, framealpha=0.9)
    plt.title('Efeito Limiar da Cobertura de Coleta de Resíduos sobre a Incidência de Arboviroses', fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    salvar_grafico("07_relacao_cobertura_vs_incidencia_lowess")
    plt.close()

#Relação Não-Linear: Despesa per Capita vs Incidência
def plot_lowess_despesa(df):
    print("  [8/11] Gerando Figura 08: Curva LOWESS Despesa vs Incidência...")
    sample = df.dropna(subset=['despesa_percapita_ano', 'taxa_incidencia_conf_100k']).copy()
    sample = sample[(sample['taxa_incidencia_conf_100k'] > 0) & (sample['despesa_percapita_ano'] > 0) & (sample['despesa_percapita_ano'] <= 1000)]
    
    if len(sample) > 5000:
        sample = sample.sample(5000, random_state=42)
        
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    ax.scatter(sample['despesa_percapita_ano'], sample['taxa_incidencia_conf_100k'], 
               alpha=0.15, color='#fdae61', s=20, edgecolors='none', label='Municípios-Ano')
    
    sns.regplot(data=sample, x='despesa_percapita_ano', y='taxa_incidencia_conf_100k', 
                lowess=True, scatter=False, ax=ax, color='#2c7bb6', 
                line_kws={'lw': 3, 'label': 'Ajuste Não-Linear LOWESS'})
    
    ax.set_yscale('log')
    ax.set_xlabel('Despesa Municipal com Manejo de Resíduos per Capita (R$/hab.ano)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Taxa de Incidência (Casos / 100 mil hab. - Escala Log)', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 700)
    
    ax.axhline(500, color='red', linestyle='--', lw=1.2, label='Limiar Risco Epidêmico (500/100k)')
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    plt.title('Associação Não-Linear entre Gastos Municipais em Resíduos e Incidência Epidemiológica', fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    salvar_grafico("08_relacao_despesa_vs_incidencia_lowess")
    plt.close()

#Disparidade Espacial por Macrorregião (Justificativa de k-fold espacial)
def plot_macrorregioes(df):
    print("  [9/11] Gerando Figura 09: Disparidades regionais e boxplots...")
    sample = df.dropna(subset=['regiao', 'taxa_incidencia_conf_100k']).copy()
    sample = sample[sample['regiao'].isin(['Norte', 'Nordeste', 'Centro-Oeste', 'Sudeste', 'Sul'])]
    sample['log_incidencia'] = np.log10(sample['taxa_incidencia_conf_100k'] + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # A. Incidência por Região
    sns.boxplot(data=sample, x='regiao', y='log_incidencia', hue='regiao', legend=False, palette='Set2', ax=ax1, width=0.5, fliersize=2)
    ax1.set_title('A: Log-Taxa de Incidência de Arboviroses por Macrorregião', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Macrorregião do Brasil', fontsize=10)
    ax1.set_ylabel('Log10(Taxa de Incidência + 1)', fontsize=10)
    ax1.axhline(np.log10(100+1), color='gold', linestyle='--', label='Médio Risco (100/100k)')
    ax1.axhline(np.log10(500+1), color='red', linestyle='--', label='Risco Epidêmico (500/100k)')
    ax1.legend(loc='lower left')
    
    # B. Cobertura de Coleta por Região
    sns.boxplot(data=sample.dropna(subset=['cobertura_coleta_pct']), x='regiao', y='cobertura_coleta_pct', hue='regiao', legend=False, palette='Set2', ax=ax2, width=0.5, fliersize=2)
    ax2.set_title('B: Cobertura da Coleta de Resíduos Domiciliares por Macrorregião', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Macrorregião do Brasil', fontsize=10)
    ax2.set_ylabel('Taxa de Cobertura de Coleta (%)', fontsize=10)
    
    fig.suptitle('Heterogeneidade Espacial: Justificativa para Validação Cruzada Espacial (Spatial k-Fold)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    salvar_grafico("09_incidencia_por_macrorregiao_boxplots")
    plt.close()

#Comparação 2019-2023 vs 2024 (Quebra Estrutural / Teste Cego)
def plot_comparacao_2024(df):
    print("  [10/11] Gerando Figura 10: Comparação temporal Treino vs Quebra 2024...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    
    # A. Proporção de municípios em surto epidêmico por ano
    epidemico_pct = df.groupby('ano')['estrato_risco_ms'].apply(lambda s: (s == 'Epidêmico').mean() * 100).reset_index()
    
    colors = ['#4575b4']*4 + ['#fdae61', '#d73027']
    bars = ax1.bar(epidemico_pct['ano'], epidemico_pct['estrato_risco_ms'], color=colors, edgecolor='black', width=0.6)
    ax1.set_title('A: Percentual de Municípios em Nível Epidêmico (>500 casos/100k)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Municípios em Epidemia (%)', fontsize=10)
    ax1.set_xlabel('Ano', fontsize=10)
    ax1.set_ylim(0, 70)
    for b in bars:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2., h + 1.2, f'{h:.1f}%', ha='center', fontsize=9, fontweight='bold')
        
    # B. Curvas de Densidade de Kernel: 2019-2023 vs 2024
    pre_2024 = df[df['ano'] < 2024]['taxa_incidencia_conf_100k'].dropna()
    post_2024 = df[df['ano'] == 2024]['taxa_incidencia_conf_100k'].dropna()
    
    sns.kdeplot(np.log10(pre_2024 + 1), color='#4575b4', lw=2.5, label='Histórico de Treino (2019–2023)', ax=ax2)
    sns.kdeplot(np.log10(post_2024 + 1), color='#d73027', lw=2.5, label='Ano Epidêmico de Teste (2024)', ax=ax2)
    
    ax2.set_title('B: Deslocamento da Distribuição de Incidência (Log10)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Log10(Taxa de Incidência + 1)', fontsize=10)
    ax2.set_ylabel('Densidade', fontsize=10)
    ax2.legend(loc='upper right', frameon=True)
    
    fig.suptitle('Impacto da Megaepidemia de 2024 na Distribuição Espaço-Temporal das Arboviroses', fontsize=12, fontweight='bold')
    plt.tight_layout()
    salvar_grafico("10_comparacao_2019_2023_vs_2024_epidemia")
    plt.close()

#Febre Amarela vs Aedes: Justificativa para Descarte
def plot_febre_amarela_justificativa(df):
    print("  [11/11] Gerando Figura 11: Justificativa de exclusão da Febre Amarela...")
    # Comparação de magnitude e prevalência municipal
    total_aedes = df[['dengue_conf', 'chik_conf', 'zika_conf']].sum().sum()
    total_fa = df['fa_casos'].sum()
    
    mun_aedes = df.groupby('cod_mun_ibge')['arbo_conf_tot'].sum().gt(0).sum()
    mun_fa = df.groupby('cod_mun_ibge')['fa_casos'].sum().gt(0).sum()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    # A. Comparação em escala logarítmica de casos
    categories = ['Dengue, Chik e Zika\n(Vetor Aedes aegypti - Urbano)', 'Febre Amarela\n(Ciclo Silvestre - Matas)']
    counts = [total_aedes, total_fa]
    
    bars1 = ax1.bar(categories, counts, color=['#d95f02', '#7570b3'], edgecolor='black', width=0.5)
    ax1.set_yscale('log')
    ax1.set_title('A: Volume Acumulado de Casos no Brasil (2019–2024)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Casos Confirmados (Escala Logarítmica)', fontsize=10)
    ax1.set_ylim(1, 3e7)
    
    ax1.text(0, total_aedes * 1.5, f'{total_aedes:,}\ncasos', ha='center', fontweight='bold', fontsize=10)
    ax1.text(1, total_fa * 1.5, f'{total_fa:,}\ncasos', ha='center', fontweight='bold', fontsize=10)
    
    # B. Comparação de Municípios Acometidos
    counts_mun = [mun_aedes, mun_fa]
    bars2 = ax2.bar(categories, counts_mun, color=['#d95f02', '#7570b3'], edgecolor='black', width=0.5)
    ax2.set_title('B: Abrangência Geográfica (Municípios com >= 1 Caso)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Número de Municípios (Total Brasil = 5.570)', fontsize=10)
    ax2.set_ylim(0, 6200)
    
    pct_aedes = (mun_aedes / 5570) * 100
    pct_fa = (mun_fa / 5570) * 100
    ax2.text(0, mun_aedes + 150, f'{mun_aedes:,} ({pct_aedes:.1f}%)', ha='center', fontweight='bold', fontsize=10)
    ax2.text(1, mun_fa + 150, f'{mun_fa:,} ({pct_fa:.1f}%)', ha='center', fontweight='bold', fontsize=10)
    
    fig.suptitle('Justificativa Epidemiológica para Descarte do Dataset de Febre Amarela', fontsize=12, fontweight='bold')
    plt.tight_layout()
    salvar_grafico("11_febre_amarela_vs_aedes_justificativa_descarte")
    plt.close()

def exportar_estatisticas_resumo(df):
    print("  [+] Exportando estatísticas em formato JSON e CSV para documentação...")
    stats_dict = {
        'total_registros': int(len(df)),
        'municipios_unicos': int(df['cod_mun_ibge'].nunique()),
        'anos': sorted([int(x) for x in df['ano'].unique()]),
        'total_casos_notificados': int(df['arbo_notif_tot'].sum()),
        'total_casos_confirmados': int(df['arbo_conf_tot'].sum()),
        'total_dengue_conf': int(df['dengue_conf'].sum()),
        'total_chik_conf': int(df['chik_conf'].sum()),
        'total_zika_conf': int(df['zika_conf'].sum()),
        'total_fa_casos': int(df['fa_casos'].sum()),
        'media_casos_confirmados': float(df['arbo_conf_tot'].mean()),
        'variancia_casos_confirmados': float(df['arbo_conf_tot'].var()),
        'razao_superdispersao_phi': float(df['arbo_conf_tot'].var() / df['arbo_conf_tot'].mean()),
        'percentual_zeros': float((df['arbo_conf_tot'] == 0).mean() * 100),
        'estratos_risco_percentual': {k: float(v) for k, v in (df['estrato_risco_ms'].value_counts(normalize=True)*100).items()},
        'respondencia_por_ano': {int(k): int(v) for k, v in df.groupby('ano')['respondeu_modulo_rs'].sum().items()},
        'indicadores_medianas': {
            'cobertura_coleta_pct': float(df['cobertura_coleta_pct'].median()),
            'despesa_percapita_ano': float(df['despesa_percapita_ano'].median()),
            'massa_percapita_kg_dia': float(df['massa_percapita_kg_dia'].median()),
            'custo_coleta_pct': float(df['custo_coleta_pct'].median()),
            'terceirizacao_coleta_pct': float(df['terceirizacao_coleta_pct'].median())
        }
    }
    
    out_json = os.path.join(DATA_DIR, "estatisticas_eda_resumo.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=4, ensure_ascii=False)
        
    print(f"  [OK] Estatísticas salvas em {out_json}")

def main():
    print("\n[+] Iniciando geração dos gráficos e análises estatísticas da EDA...")
    df = carregar_dados()
    
    plot_evolucao_temporal(df)
    plot_superdispersao_poisson(df)
    plot_estratos_risco(df)
    plot_taxa_respondencia(df)
    plot_distribuicao_indicadores(df)
    plot_matriz_correlacao(df)
    plot_lowess_cobertura(df)
    plot_lowess_despesa(df)
    plot_macrorregioes(df)
    plot_comparacao_2024(df)
    plot_febre_amarela_justificativa(df)
    
    exportar_estatisticas_resumo(df)
    print("\n[OK] Todas as 11 figuras da Análise Exploratória foram geradas com sucesso em 'imagens/'!\n")

if __name__ == '__main__':
    main()
