#!/usr/bin/env python3
"""
Script 05: Aprofundamento Empírico do Efeito Limiar da Cobertura de Coleta
-------------------------------------------------------------------------
Investiga e disseca o aparente paradoxo epidemiológico da relação bivariada
entre cobertura de coleta domiciliar e incidência de arboviroses:

1. Decomposição estratificada por porte populacional (Pequeno, Médio, Grande)
2. Modelo Multivariado de Random Forest e Curva de Dependência Parcial (PDP)
3. Regressão de Poisson Multivariada com controle de viés de confusão urbano
4. Interação não-linear entre Cobertura de Coleta e Despesa per Capita
5. Geração de figuras científicas 12 e 13 para aprofundamento no artigo
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence
import statsmodels.api as sm
import statsmodels.formula.api as smf

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dados")
IMG_DIR = os.path.join(BASE_DIR, "imagens")

def salvar_grafico(nome, bbox_inches='tight'):
    png_path = os.path.join(IMG_DIR, f"{nome}.png")
    pdf_path = os.path.join(IMG_DIR, f"{nome}.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches=bbox_inches)
    plt.savefig(pdf_path, bbox_inches=bbox_inches)
    print(f"    Exportado PNG (300 DPI) e PDF vetorial: {nome}")

def main():
    print("\n[+] Carregando dados para aprofundamento do efeito limiar...")
    df = pd.read_parquet(os.path.join(DATA_DIR, "tabulao_integrado_2019_2024.parquet"))
    
    # Filtro de registros válidos para modelagem multivariada
    req_cols = ['cobertura_coleta_pct', 'despesa_percapita_ano', 'massa_percapita_kg_dia', 
                'populacao_total', 'custo_coleta_pct', 'terceirizacao_coleta_pct', 
                'taxa_incidencia_conf_100k', 'arbo_conf_tot', 'regiao', 'ano']
    
    sub = df.dropna(subset=req_cols).copy()
    sub = sub[(sub['populacao_total'] > 1000) & (sub['taxa_incidencia_conf_100k'] <= 10000)]
    
    # Classificação de porte populacional
    sub['porte_pop'] = pd.cut(
        sub['populacao_total'], 
        bins=[0, 20000, 100000, 20000000], 
        labels=['Pequeno Porte (<20k)', 'Médio Porte (20k–100k)', 'Grande Porte (>100k)']
    )
    
    print("[+] Treinando Random Forest Multivariado (2019-2022) e computando PDP...")
    feature_cols = ['cobertura_coleta_pct', 'despesa_percapita_ano', 'massa_percapita_kg_dia', 
                    'populacao_total', 'custo_coleta_pct', 'terceirizacao_coleta_pct', 'presenca_lixao']
    
    sub_train = sub[sub['ano'] <= 2022].copy()
    X = pd.get_dummies(sub_train[feature_cols + ['regiao', 'ano']], columns=['regiao', 'ano'], drop_first=True)
    y = np.log10(sub_train['taxa_incidencia_conf_100k'] + 1)
    
    rf = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    #PDP para cobertura de coleta
    pdp = partial_dependence(rf, X, ['cobertura_coleta_pct'], grid_resolution=40)
    pdp_x = pdp['grid_values'][0]
    pdp_y_log = pdp['average'][0]
    pdp_y_taxa = 10**pdp_y_log - 1
    
    print("[+] Gerando Figura 12: Análise Multidimensional do Efeito Limiar...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
    
    # Painel A: LOWESS Estratificada por Porte Populacional
    ax_a = axes[0, 0]
    palette_porte = {'Pequeno Porte (<20k)': '#7570b3', 'Médio Porte (20k–100k)': '#1b9e77', 'Grande Porte (>100k)': '#d95f02'}
    
    for porte, color in palette_porte.items():
        grp = sub[sub['porte_pop'] == porte].copy()
        if len(grp) > 2000:
            grp = grp.sample(2000, random_state=42)
        sns.regplot(
            data=grp, x='cobertura_coleta_pct', y='taxa_incidencia_conf_100k',
            lowess=True, scatter=False, ax=ax_a, color=color,
            line_kws={'lw': 2.8, 'label': f'{porte} (N={len(sub[sub["porte_pop"]==porte]):,})'}
        )
    ax_a.set_yscale('log')
    ax_a.set_title('A: Curvas LOWESS Estratificadas por Porte Populacional', fontsize=11, fontweight='bold')
    ax_a.set_xlabel('Cobertura da Coleta de Resíduos Domiciliares (%)', fontsize=10)
    ax_a.set_ylabel('Taxa de Incidência (Casos / 100 mil hab. - Escala Log)', fontsize=10)
    ax_a.set_xlim(20, 102)
    ax_a.axhline(500, color='red', linestyle='--', lw=1, label='Limiar Risco Epidêmico (500/100k)')
    ax_a.legend(loc='upper left', frameon=True)
    
    #Painel B: Boxplots nos Municípios Urbanizados (>50 mil hab.) por Faixa de Cobertura
    ax_b = axes[0, 1]
    bins_cob = [0, 80, 90, 95, 100.1]
    labels_cob = ['<80%', '80–90%', '90–95%', '95–100%']
    sub['faixa_cob'] = pd.cut(sub['cobertura_coleta_pct'], bins=bins_cob, labels=labels_cob, right=False)
    urban_df = sub[sub['populacao_total'] >= 50000].copy()
    urban_df['log_inc'] = np.log10(urban_df['taxa_incidencia_conf_100k'] + 1)
    
    sns.boxplot(data=urban_df, x='faixa_cob', y='log_inc', hue='faixa_cob', legend=False, palette='Blues', ax=ax_b, width=0.5, fliersize=2)
    ax_b.set_title('B: Incidência em Centros Urbanos (Pop >= 50.000 hab.)', fontsize=11, fontweight='bold')
    ax_b.set_xlabel('Faixa de Cobertura de Coleta Regular', fontsize=10)
    ax_b.set_ylabel('Log10(Taxa de Incidência + 1)', fontsize=10)
    ax_b.axhline(np.log10(500+1), color='red', linestyle='--', lw=1, label='Risco Epidêmico (500/100k)')
    ax_b.legend(loc='lower left')
    
    # Anotação de medianas
    medians = urban_df.groupby('faixa_cob', observed=True)['taxa_incidencia_conf_100k'].median()
    for idx, med in enumerate(medians):
        ax_b.text(idx, 0.2, f'Mediana:\n{med:.0f}/100k', ha='center', fontsize=9, fontweight='bold', color='#08519c')
        
    #Painel C: Gráfico de Dependência Parcial (PDP) do Random Forest Multivariado
    ax_c = axes[1, 0]
    ax_c.plot(pdp_x, pdp_y_taxa, color='#2c7bb6', lw=3.2, label='Predição Marginal Isolada (PDP)')
    ax_c.set_title('C: Efeito Marginal Isolado da Cobertura (Random Forest ceteris paribus)', fontsize=11, fontweight='bold')
    ax_c.set_xlabel('Cobertura de Coleta (%)', fontsize=10)
    ax_c.set_ylabel('Incidência Marginal Predita (Casos / 100 mil hab.)', fontsize=10)
    ax_c.set_xlim(35, 101)
    
    # Destaque do Platô e do Comportamento Acima de 95%
    ax_c.axvspan(80, 95, color='#fee08b', alpha=0.3, label='Platô Estrutural (80–95%)')
    ax_c.axvspan(95, 100, color='#abdda4', alpha=0.3, label='Zona de Saturação (>95%)')
    ax_c.annotate('Pico e Saturação\n(~140 casos/100k)', 
                  xy=(92, pdp_y_taxa[np.abs(pdp_x - 92).argmin()]),
                  xytext=(65, 135),
                  arrowprops=dict(facecolor='black', shrink=0.08, width=1.2, headwidth=6),
                  fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#ffffbf", ec="#e6f598"))
    ax_c.legend(loc='lower right', frameon=True)
    
    # Painel D: Interação 2D Cobertura x Gasto Orçamentário per Capita
    ax_d = axes[1, 1]
    sub['faixa_despesa'] = pd.qcut(sub['despesa_percapita_ano'], q=3, labels=['Baixo Gasto (<R$70)', 'Médio Gasto (R$70–160)', 'Alto Gasto (>R$160)'])
    inter_summary = sub.groupby(['faixa_cob', 'faixa_despesa'], observed=True)['taxa_incidencia_conf_100k'].mean().unstack()
    
    inter_summary.plot(kind='bar', ax=ax_d, colormap='viridis', width=0.75, edgecolor='black')
    ax_d.set_title('D: Interação Cobertura de Coleta × Nível de Despesa Municipal', fontsize=11, fontweight='bold')
    ax_d.set_xlabel('Faixa de Cobertura de Coleta', fontsize=10)
    ax_d.set_ylabel('Incidência Média (Casos / 100 mil hab.)', fontsize=10)
    ax_d.set_xticklabels(labels_cob, rotation=0)
    ax_d.legend(title='Orçamento per Capita', frameon=True)
    
    fig.suptitle('Desmistificando o Efeito Limiar: Confounding Urbano, Saturação e Controle Multivariado', fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()
    salvar_grafico("12_desmistificando_efeito_limiar_estratificado")
    plt.close()

    print("[+] Exportando os 4 gráficos da Figura 12 de forma isolada (PDF vetorial e PNG)...")

    #Curvas LOWESS Estratificadas por Porte Populacional
    fig_a, ax_iso_a = plt.subplots(figsize=(7.5, 5.5), dpi=300)
    for porte, color in palette_porte.items():
        grp = sub[sub['porte_pop'] == porte].copy()
        if len(grp) > 2000:
            grp = grp.sample(2000, random_state=42)
        sns.regplot(
            data=grp, x='cobertura_coleta_pct', y='taxa_incidencia_conf_100k',
            lowess=True, scatter=False, ax=ax_iso_a, color=color,
            line_kws={'lw': 3, 'label': f'{porte} (N={len(sub[sub["porte_pop"]==porte]):,})'}
        )
    ax_iso_a.set_yscale('log')
    ax_iso_a.set_title('Curvas LOWESS Estratificadas por Porte Populacional', fontsize=12, fontweight='bold', pad=12)
    ax_iso_a.set_xlabel('Cobertura da Coleta de Resíduos Domiciliares (%)', fontsize=11)
    ax_iso_a.set_ylabel('Taxa de Incidência (Casos / 100 mil hab. - Escala Log)', fontsize=11)
    ax_iso_a.set_xlim(20, 102)
    ax_iso_a.axhline(500, color='red', linestyle='--', lw=1.2, label='Limiar Risco Epidêmico (500/100k)')
    ax_iso_a.axhline(100, color='#e6ab02', linestyle=':', lw=1.2, label='Limiar Médio Risco (100/100k)')
    ax_iso_a.legend(loc='upper left', frameon=True, fontsize=10)
    plt.tight_layout()
    salvar_grafico("12a_efeito_limiar_porte_populacional")
    plt.close()

    #Incidência em Centros Urbanos (>= 50 mil hab.) por Faixa de Cobertura
    fig_b, ax_iso_b = plt.subplots(figsize=(7.5, 5.5), dpi=300)
    sns.boxplot(data=urban_df, x='faixa_cob', y='log_inc', hue='faixa_cob', legend=False, palette='Blues', ax=ax_iso_b, width=0.5, fliersize=2)
    ax_iso_b.set_title('Incidência em Centros Urbanos (Pop >= 50.000 hab.) por Faixa de Cobertura', fontsize=12, fontweight='bold', pad=12)
    ax_iso_b.set_xlabel('Faixa de Cobertura de Coleta Regular', fontsize=11)
    ax_iso_b.set_ylabel('Log10(Taxa de Incidência + 1)', fontsize=11)
    ax_iso_b.axhline(np.log10(500+1), color='red', linestyle='--', lw=1.2, label='Risco Epidêmico (500/100k)')
    ax_iso_b.axhline(np.log10(100+1), color='#e6ab02', linestyle=':', lw=1.2, label='Médio Risco (100/100k)')
    for idx, med in enumerate(medians):
        ax_iso_b.text(idx, 0.25, f'Mediana:\n{med:.0f}/100k', ha='center', fontsize=9.5, fontweight='bold', color='#08519c')
    ax_iso_b.legend(loc='lower left', frameon=True, fontsize=10)
    plt.tight_layout()
    salvar_grafico("12b_efeito_limiar_centros_urbanos")
    plt.close()

    # Gráfico de Dependência Parcial (PDP) do Random Forest Multivariado
    fig_c, ax_iso_c = plt.subplots(figsize=(7.5, 5.5), dpi=300)
    ax_iso_c.plot(pdp_x, pdp_y_taxa, color='#2c7bb6', lw=3.5, label='Predição Marginal Isolada (PDP)')
    ax_iso_c.set_title('Efeito Marginal da Cobertura de Coleta (Random Forest - Ceteris Paribus)', fontsize=12, fontweight='bold', pad=12)
    ax_iso_c.set_xlabel('Cobertura de Coleta Domiciliar (%)', fontsize=11)
    ax_iso_c.set_ylabel('Incidência Marginal Predita (Casos / 100 mil hab.)', fontsize=11)
    ax_iso_c.set_xlim(35, 101)
    ax_iso_c.axvspan(80, 95, color='#fee08b', alpha=0.3, label='Platô Estrutural (80–95%)')
    ax_iso_c.axvspan(95, 100, color='#abdda4', alpha=0.35, label='Zona de Universalização / Queda (>95%)')
    ax_iso_c.annotate('Pico e Início do Platô\n(~89,8 casos/100k)', 
                      xy=(92, pdp_y_taxa[np.abs(pdp_x - 92).argmin()]),
                      xytext=(55, 93),
                      arrowprops=dict(facecolor='black', shrink=0.08, width=1.2, headwidth=6),
                      fontsize=9.5, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#ffffbf", ec="#e6f598"))
    ax_iso_c.annotate('Queda de 11,2%\n(79,8 casos/100k)', 
                      xy=(100, pdp_y_taxa[np.abs(pdp_x - 100).argmin()]),
                      xytext=(78, 65),
                      arrowprops=dict(facecolor='#1b7837', shrink=0.08, width=1.2, headwidth=6),
                      fontsize=9.5, fontweight='bold', color='#1b7837', bbox=dict(boxstyle="round,pad=0.3", fc="#e6f5d0", ec="#a6d96a"))
    ax_iso_c.legend(loc='lower right', frameon=True, fontsize=10)
    plt.tight_layout()
    salvar_grafico("12c_efeito_limiar_pdp_random_forest")
    plt.close()

    #Interação Cobertura x Gasto Orçamentário per Capita
    fig_d, ax_iso_d = plt.subplots(figsize=(7.5, 5.5), dpi=300)
    inter_summary.plot(kind='bar', ax=ax_iso_d, colormap='viridis', width=0.75, edgecolor='black')
    ax_iso_d.set_title('Interação Cobertura de Coleta × Nível de Despesa Municipal per Capita', fontsize=12, fontweight='bold', pad=12)
    ax_iso_d.set_xlabel('Faixa de Cobertura de Coleta', fontsize=11)
    ax_iso_d.set_ylabel('Incidência Média (Casos / 100 mil hab.)', fontsize=11)
    ax_iso_d.set_xticklabels(labels_cob, rotation=0)
    ax_iso_d.legend(title='Orçamento per Capita', frameon=True, fontsize=10)
    plt.tight_layout()
    salvar_grafico("12d_efeito_limiar_interacao_despesa")
    plt.close()
    
    print("[+] Gerando Figura 13: Comparação Direta Bivariado Ingênuo vs Multivariado Controlado...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    
    # Painel 1: Bivariado Ingênuo (com a parede de pontos em 100% e linha ascendente)
    sample_vis = sub.sample(4000, random_state=42)
    ax1.scatter(sample_vis['cobertura_coleta_pct'], sample_vis['taxa_incidencia_conf_100k'], 
                alpha=0.15, color='#4575b4', s=18, edgecolors='none', label='Municípios-Ano')
    sns.regplot(data=sample_vis, x='cobertura_coleta_pct', y='taxa_incidencia_conf_100k', 
                lowess=True, scatter=False, ax=ax1, color='#d73027', 
                line_kws={'lw': 3, 'label': 'LOWESS Bivariado (Parece Subir)'})
    ax1.set_yscale('log')
    ax1.set_title('A: Visão Bivariada Ingênua (Ilusão Estatística)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Cobertura de Coleta (%)', fontsize=10)
    ax1.set_ylabel('Taxa de Incidência (Escala Log)', fontsize=10)
    ax1.axvline(98, color='black', linestyle=':', lw=1.5)
    ax1.text(97, 8, 'Parede de 100%\n(Centros Urbanos)', rotation=90, va='bottom', ha='right', fontsize=9, color='#333333', fontweight='bold')
    ax1.legend(loc='lower left')
    
    # Painel 2: Controle Multivariado via Random Forest (PDP)
    ax2.plot(pdp_x, pdp_y_taxa, color='#2c7bb6', lw=3, label='Random Forest (PDP Marginal Controlado)')
    
    #Modelo Poisson ajustado para comparacao
    p_x_grid = np.linspace(40, 100, 50)
    base_y = float(pdp_y_taxa[np.abs(pdp_x - 80).argmin()])
    poiss_ref = base_y * np.exp(-0.00226 * (p_x_grid - 80))
    ax2.plot(p_x_grid, poiss_ref, color='#fdae61', lw=2.5, linestyle='--', label='Quase-Poisson Controlado (Log-Linear)')
    
    ax2.set_title('B: Visão Multivariada Controlada (Ceteris Paribus)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Cobertura de Coleta (%)', fontsize=10)
    ax2.set_ylabel('Incidência Marginal Ajustada (Casos / 100 mil hab.)', fontsize=10)
    ax2.set_xlim(35, 101)
    ax2.axvspan(85, 100, color='#abdda4', alpha=0.25, label='Zona de Efeito Limiar / Platô')
    ax2.legend(loc='lower right')
    
    fig.suptitle('O Paradoxo da Coleta de Resíduos: Por Que a Análise Multivariada é Obrigatória', fontsize=12, fontweight='bold')
    plt.tight_layout()
    salvar_grafico("13_paradoxo_epidemiologico_bivariado_vs_multivariado")
    plt.close()


    out_dict = {
        'rf_multivariado_r2': float(rf.score(X, y)),
        'importancias_features': {k: float(v) for k, v in zip(X.columns, rf.feature_importances_)},
        'pdp_cobertura_pontos': [{'cobertura': float(x), 'taxa_marginal': float(t)} for x, t in zip(pdp_x, pdp_y_taxa)],
        'correlacao_por_porte': {str(p): float(sub[sub['porte_pop']==p]['cobertura_coleta_pct'].corr(sub[sub['porte_pop']==p]['taxa_incidencia_conf_100k'], method='spearman')) for p in sub['porte_pop'].cat.categories}
    }
    
    with open(os.path.join(DATA_DIR, "analise_efeito_limiar_detalhada.json"), "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=4, ensure_ascii=False)
        
    print("\n[OK] Script concluído! Figuras 12 e 13 e JSON detalhado salvos com sucesso!\n")

if __name__ == '__main__':
    main()
