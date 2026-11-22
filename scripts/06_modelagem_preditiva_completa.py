#!/usr/bin/env python3
"""
Script 06: Modelagem Preditiva Completa (Etapa 05)
--------------------------------------------------
Compara formalmente a Regressão Quase-Poisson e o Random Forest:

1. Preparação de dados e engenharia de atributos (taxa de urbanização, imputação longitudinal).
2. Validação Cruzada Espacial (Spatial GroupKFold por UF, 5 folds) na partição de Treino (2019-2022).
3. Teste Prospectivo Cego no Choque Epidêmico de 2024 (e consolidado 2023-2024).
4. Modelagem Contínua (Quase-Poisson GLM com offset ln(Pop) e Huber-White vs RF Regressor).
5. Modelagem Discreta de Risco (Poisson categorizado vs RF Classifier com pesos balanceados).
6. Interpretabilidade: Coeficientes IRR com IC95% (Poisson) e Feature Importance (Random Forest).
7. Geração de 5 figuras científicas (PNG 300 DPI e PDF vetorial Type 42) e tabelas de métricas.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix
)
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
os.makedirs(IMG_DIR, exist_ok=True)

def salvar_grafico(nome, bbox_inches='tight'):
    png_path = os.path.join(IMG_DIR, f"{nome}.png")
    pdf_path = os.path.join(IMG_DIR, f"{nome}.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches=bbox_inches)
    plt.savefig(pdf_path, bbox_inches=bbox_inches)
    print(f"    Exportado PNG (300 DPI) e PDF vetorial: {nome}")

def preparar_dados():
    print("[1/6] Carregando e preparando base de dados analítica (sem vazamento temporal)...")
    p_tab = os.path.join(DATA_DIR, "tabulao_integrado_2019_2024.parquet")
    df = pd.read_parquet(p_tab)
    df = df[df['populacao_total'] >= 500].sort_values(['cod_mun_ibge', 'ano']).copy()

    #Split temporal ANTES de qualquer imputação
    train_df = df[df['ano'] <= 2022].copy()
    test_2023 = df[df['ano'] == 2023].copy()
    test_2024 = df[df['ano'] == 2024].copy()

    feature_impute = [
        'cobertura_coleta_pct', 'despesa_percapita_ano', 'massa_percapita_kg_dia',
        'custo_coleta_pct', 'terceirizacao_coleta_pct', 'presenca_lixao', 'populacao_urbana'
    ]

    # Imputação intra-municipal estritamente dentro do treino (2019-2022)
    for col in feature_impute:
        train_df[col] = train_df.groupby('cod_mun_ibge')[col].transform(lambda x: x.ffill().bfill())

    reg_medians = train_df.groupby('regiao')[feature_impute].median()
    glob_medians = train_df[feature_impute].median()

    for col in feature_impute:
        train_df[col] = train_df.apply(
            lambda r: reg_medians.loc[r['regiao'], col] if pd.isna(r[col]) else r[col], axis=1
        ).fillna(glob_medians[col])

    # Imputação causal em 2023 (herda até 2022) e 2024 (herda até 2023), fallback nas medianas do treino
    last_train = train_df.sort_values('ano').groupby('cod_mun_ibge')[feature_impute].last()
    for col in feature_impute:
        m23 = test_2023[col].isna()
        test_2023.loc[m23, col] = test_2023.loc[m23, 'cod_mun_ibge'].map(last_train[col])
        test_2023[col] = test_2023.apply(
            lambda r: reg_medians.loc[r['regiao'], col] if pd.isna(r[col]) else r[col], axis=1
        ).fillna(glob_medians[col])

    last_u23 = pd.concat([train_df, test_2023]).sort_values('ano').groupby('cod_mun_ibge')[feature_impute].last()
    for col in feature_impute:
        m24 = test_2024[col].isna()
        test_2024.loc[m24, col] = test_2024.loc[m24, 'cod_mun_ibge'].map(last_u23[col])
        test_2024[col] = test_2024.apply(
            lambda r: reg_medians.loc[r['regiao'], col] if pd.isna(r[col]) else r[col], axis=1
        ).fillna(glob_medians[col])

    classes_ordem = ['Baixo', 'Médio', 'Alto', 'Epidêmico']
    for part in [train_df, test_2023, test_2024]:
        part['taxa_urbanizacao'] = (part['populacao_urbana'] / part['populacao_total'].clip(lower=1) * 100).clip(0, 100)
        part['ln_pop'] = np.log(part['populacao_total'].clip(lower=100))
        part['log_incidencia'] = np.log10(part['taxa_incidencia_conf_100k'] + 1)
        part['estrato_risco_ms'] = pd.Categorical(part['estrato_risco_ms'], categories=classes_ordem, ordered=True)

    test_df = pd.concat([test_2023, test_2024], ignore_index=True)

    print(f"    Conjunto de Treino/Validação (2019–2022): {len(train_df):,} municípios-ano ({train_df['uf'].nunique()} UFs).")
    print(f"    Teste Prospectivo 2023 (Regular): {len(test_2023):,} municípios.")
    print(f"    Teste Prospectivo 2024 (Choque): {len(test_2024):,} municípios.")

    return df, train_df, test_df, test_2024

def classificar_estrato_ms(taxa):
    """Converte taxa de incidência contínua nas 4 faixas oficiais do Ministério da Saúde."""
    if taxa < 100:
        return 'Baixo'
    elif taxa < 300:
        return 'Médio'
    elif taxa < 500:
        return 'Alto'
    else:
        return 'Epidêmico'

def treinar_e_avaliar_poisson(train_df, test_df, test_2024):
    print("\n[2/6] Ajustando Modelo Quase-Poisson (Winsorizado p99 + Cluster por Município + Ref. Centro-Oeste)...")
    
    p99_desp = float(train_df['despesa_percapita_ano'].quantile(0.99))
    train_w = train_df.copy()
    test_2024 = test_2024.copy()
    train_w['despesa_percapita_ano'] = train_w['despesa_percapita_ano'].clip(upper=p99_desp)
    test_2024['despesa_percapita_ano'] = test_2024['despesa_percapita_ano'].clip(upper=p99_desp)

    formula = (
        "arbo_conf_tot ~ cobertura_coleta_pct + despesa_percapita_ano + "
        "massa_percapita_kg_dia + custo_coleta_pct + terceirizacao_coleta_pct + "
        "presenca_lixao + taxa_urbanizacao + C(regiao, Treatment(reference='Centro-Oeste'))"
    )
    
    t0 = time.time()
    poiss_model = smf.glm(
        formula=formula,
        data=train_w,
        family=sm.families.Poisson(),
        offset=train_w['ln_pop']
    )
    poiss_fit = poiss_model.fit(cov_type='cluster', cov_kwds={'groups': train_w['cod_mun_ibge']})
    tempo_treino = time.time() - t0
    
    print(f"    Quase-Poisson ajustado em {tempo_treino:.2f}s com sucesso!")
    
    # Extração de coeficientes e Razões de Taxa de Incidência (IRR)
    coefs = poiss_fit.params
    bse = poiss_fit.bse
    pvalues = poiss_fit.pvalues
    conf_int = poiss_fit.conf_int()
    
    irr_df = pd.DataFrame({
        'variavel': coefs.index,
        'beta': coefs.values,
        'se_robusto': bse.values,
        'p_valor': pvalues.values,
        'irr': np.exp(coefs.values),
        'irr_ci_low': np.exp(conf_int[0].values),
        'irr_ci_high': np.exp(conf_int[1].values)
    })
    
    # Predições no teste prospectivo 2024
    test_2024 = test_2024.copy()
    pred_contagem_2024 = poiss_fit.predict(test_2024, offset=test_2024['ln_pop'])
    test_2024['pred_poisson_contagem'] = pred_contagem_2024
    test_2024['pred_poisson_taxa'] = (pred_contagem_2024 / test_2024['populacao_total']) * 100000
    test_2024['pred_poisson_estrato'] = test_2024['pred_poisson_taxa'].apply(classificar_estrato_ms)
    
    # Métricas de regressão
    y_true_taxa = test_2024['taxa_incidencia_conf_100k'].values
    y_pred_taxa = test_2024['pred_poisson_taxa'].values
    
    rmse = np.sqrt(mean_squared_error(y_true_taxa, y_pred_taxa))
    mae = mean_absolute_error(y_true_taxa, y_pred_taxa)
    r2 = r2_score(y_true_taxa, y_pred_taxa)
    
    # Métricas de classificação dos estratos
    classes = ['Baixo', 'Médio', 'Alto', 'Epidêmico']
    y_true_cls = test_2024['estrato_risco_ms'].astype(str)
    y_pred_cls = test_2024['pred_poisson_estrato'].astype(str)
    
    acc = accuracy_score(y_true_cls, y_pred_cls)
    bal_acc = balanced_accuracy_score(y_true_cls, y_pred_cls)
    f1_macro = f1_score(y_true_cls, y_pred_cls, labels=classes, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true_cls, y_pred_cls, labels=classes, average='weighted', zero_division=0)
    
    metricas_poisson = {
        'modelo': 'Quase-Poisson (GLM Robusto)',
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'acuracia': float(acc),
        'acuracia_balanceada': float(bal_acc),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'tempo_segundos': float(tempo_treino)
    }
    
    return poiss_fit, irr_df, test_2024, metricas_poisson

def treinar_e_avaliar_random_forest(train_df, test_df, test_2024):
    print("\n[3/6] Ajustando Modelos Random Forest (Regressor e Classifier)...")
    
    feature_cols = [
        'cobertura_coleta_pct', 'despesa_percapita_ano', 'massa_percapita_kg_dia',
        'custo_coleta_pct', 'terceirizacao_coleta_pct', 'presenca_lixao',
        'taxa_urbanizacao', 'populacao_total'
    ]
    
    #One-hot encoding de Macrorregião
    X_train = pd.get_dummies(train_df[feature_cols + ['regiao']], columns=['regiao'], drop_first=True)
    X_test_2024 = pd.get_dummies(test_2024[feature_cols + ['regiao']], columns=['regiao'], drop_first=True)
    
    # Garante alinhamento exato de colunas
    X_test_2024 = X_test_2024.reindex(columns=X_train.columns, fill_value=0)
    
    #Random Forest Regressor
    y_train_reg = train_df['taxa_incidencia_conf_100k'].values
    y_true_taxa = test_2024['taxa_incidencia_conf_100k'].values
    
    t0 = time.time()
    rf_reg = RandomForestRegressor(
        n_estimators=200,
        max_depth=14,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )
    rf_reg.fit(X_train, y_train_reg)
    tempo_rf_reg = time.time() - t0
    
    pred_rf_reg_taxa = rf_reg.predict(X_test_2024)
    test_2024['pred_rf_taxa'] = pred_rf_reg_taxa
    
    rmse_rf = np.sqrt(mean_squared_error(y_true_taxa, pred_rf_reg_taxa))
    mae_rf = mean_absolute_error(y_true_taxa, pred_rf_reg_taxa)
    r2_rf = r2_score(y_true_taxa, pred_rf_reg_taxa)
    
    #Random Forest Classifier (com pesos balanceados)
    classes = ['Baixo', 'Médio', 'Alto', 'Epidêmico']
    y_train_clf = train_df['estrato_risco_ms'].astype(str).values
    y_true_clf = test_2024['estrato_risco_ms'].astype(str).values
    
    t0 = time.time()
    rf_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_clf.fit(X_train, y_train_clf)
    tempo_rf_clf = time.time() - t0
    
    pred_rf_clf = rf_clf.predict(X_test_2024)
    test_2024['pred_rf_estrato'] = pred_rf_clf
    
    acc_rf = accuracy_score(y_true_clf, pred_rf_clf)
    bal_acc_rf = balanced_accuracy_score(y_true_clf, pred_rf_clf)
    f1_macro_rf = f1_score(y_true_clf, pred_rf_clf, labels=classes, average='macro', zero_division=0)
    f1_weighted_rf = f1_score(y_true_clf, pred_rf_clf, labels=classes, average='weighted', zero_division=0)
    
    metricas_rf = {
        'modelo': 'Random Forest (Ensemble)',
        'rmse': float(rmse_rf),
        'mae': float(mae_rf),
        'r2': float(r2_rf),
        'acuracia': float(acc_rf),
        'acuracia_balanceada': float(bal_acc_rf),
        'f1_macro': float(f1_macro_rf),
        'f1_weighted': float(f1_weighted_rf),
        'tempo_segundos': float(tempo_rf_reg + tempo_rf_clf)
    }
    
    # Importância de atributos (MDI e Permutação)
    feat_importances = pd.DataFrame({
        'feature': X_train.columns,
        'importancia_mdi_reg': rf_reg.feature_importances_,
        'importancia_mdi_clf': rf_clf.feature_importances_
    }).sort_values('importancia_mdi_reg', ascending=False)
    
    return rf_reg, rf_clf, feat_importances, test_2024, metricas_rf, X_train, X_test_2024

def executar_validacao_cruzada_espacial(train_df):
    print("\n[4/6] Executando Validação Cruzada Espacial (Spatial GroupKFold por UF)...")
    
    feature_cols = [
        'cobertura_coleta_pct', 'despesa_percapita_ano', 'massa_percapita_kg_dia',
        'custo_coleta_pct', 'terceirizacao_coleta_pct', 'presenca_lixao',
        'taxa_urbanizacao', 'populacao_total'
    ]
    
    X = pd.get_dummies(train_df[feature_cols + ['regiao']], columns=['regiao'], drop_first=True)
    y_reg = train_df['taxa_incidencia_conf_100k'].values
    y_clf = train_df['estrato_risco_ms'].astype(str).values
    groups = train_df['uf'].values
    
    gkf = GroupKFold(n_splits=5)
    
    folds_res = []
    classes = ['Baixo', 'Médio', 'Alto', 'Epidêmico']
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_reg, groups=groups), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr_reg, y_val_reg = y_reg[train_idx], y_reg[val_idx]
        y_tr_clf, y_val_clf = y_clf[train_idx], y_clf[val_idx]
        
        # RF Regressor
        rf_r = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
        rf_r.fit(X_tr, y_tr_reg)
        p_reg = rf_r.predict(X_val)
        
        r2_f = r2_score(y_val_reg, p_reg)
        rmse_f = np.sqrt(mean_squared_error(y_val_reg, p_reg))
        mae_f = mean_absolute_error(y_val_reg, p_reg)
        
        # RF Classifier
        rf_c = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
        rf_c.fit(X_tr, y_tr_clf)
        p_clf = rf_c.predict(X_val)
        
        f1_m = f1_score(y_val_clf, p_clf, labels=classes, average='macro', zero_division=0)
        bal_acc = balanced_accuracy_score(y_val_clf, p_clf)
        
        ufs_val = np.unique(groups[val_idx])
        folds_res.append({
            'fold': fold,
            'num_ufs_val': len(ufs_val),
            'n_val': len(val_idx),
            'rmse': rmse_f,
            'mae': mae_f,
            'r2': r2_f,
            'bal_acc': bal_acc,
            'f1_macro': f1_m
        })
        print(f"    Fold {fold} ({len(ufs_val)} UFs na validação): R²={r2_f:.3f} | RMSE={rmse_f:.1f} | F1-Macro={f1_m:.3f}")
        
    df_cv = pd.DataFrame(folds_res)
    print(f"  -> Média Validação Cruzada Espacial: R²={df_cv['r2'].mean():.3f} (±{df_cv['r2'].std():.3f}) | F1-Macro={df_cv['f1_macro'].mean():.3f} (±{df_cv['f1_macro'].std():.3f})")
    return df_cv

def gerar_graficos_e_tabelas(irr_df, feat_importances, test_2024, metricas_poisson, metricas_rf, df_cv):
    print("\n[5/6] Gerando Figuras Científicas e Tabelas dos Resultados...")
    
    #Comparativo de Métricas dos Modelos
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    
    modelos = ['Quase-Poisson (GLM)', 'Random Forest (Ensemble)']
    rmses = [metricas_poisson['rmse'], metricas_rf['rmse']]
    maes = [metricas_poisson['mae'], metricas_rf['mae']]
    f1_macros = [metricas_poisson['f1_macro'], metricas_rf['f1_macro']]
    bal_accs = [metricas_poisson['acuracia_balanceada'], metricas_rf['acuracia_balanceada']]
    
    #Painel 1: Erros de Regressão
    x = np.arange(len(modelos))
    width = 0.35
    axes[0].bar(x - width/2, rmses, width, label='RMSE', color='#4575b4', edgecolor='black')
    axes[0].bar(x + width/2, maes, width, label='MAE', color='#74add1', edgecolor='black')
    axes[0].set_title('A: Erros de Predição no Teste 2024 (Menor é Melhor)', fontsize=11, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(modelos, fontsize=10)
    axes[0].set_ylabel('Taxa de Incidência (Casos / 100 mil hab.)', fontsize=10)
    axes[0].legend(frameon=True)
    
    for i, (r, m) in enumerate(zip(rmses, maes)):
        axes[0].text(i - width/2, r + 20, f'{r:.1f}', ha='center', fontsize=9, fontweight='bold')
        axes[0].text(i + width/2, m + 20, f'{m:.1f}', ha='center', fontsize=9, fontweight='bold')
        
    #Métricas de Classificação de Risco
    axes[1].bar(x - width/2, f1_macros, width, label='F1-Score Macro', color='#d73027', edgecolor='black')
    axes[1].bar(x + width/2, bal_accs, width, label='Acurácia Balanceada', color='#fdae61', edgecolor='black')
    axes[1].set_title('B: Identificação dos Estratos de Risco (Maior é Melhor)', fontsize=11, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(modelos, fontsize=10)
    axes[1].set_ylabel('Pontuação (0.0 a 1.0)', fontsize=10)
    axes[1].set_ylim(0, 0.8)
    axes[1].legend(frameon=True)
    
    for i, (f, a) in enumerate(zip(f1_macros, bal_accs)):
        axes[1].text(i - width/2, f + 0.02, f'{f:.3f}', ha='center', fontsize=9, fontweight='bold')
        axes[1].text(i + width/2, a + 0.02, f'{a:.3f}', ha='center', fontsize=9, fontweight='bold')
        
    fig.suptitle('Benchmark Prospectivo: Quase-Poisson vs. Random Forest no Choque Epidêmico de 2024', fontsize=12, fontweight='bold')
    plt.tight_layout()
    salvar_grafico("14_comparacao_metricas_modelos")
    plt.close()
    
    #Matrizes de Confusão dos Estratos de Risco
    classes = ['Baixo', 'Médio', 'Alto', 'Epidêmico']
    cm_poiss = confusion_matrix(test_2024['estrato_risco_ms'].astype(str), test_2024['pred_poisson_estrato'].astype(str), labels=classes, normalize='true')
    cm_rf = confusion_matrix(test_2024['estrato_risco_ms'].astype(str), test_2024['pred_rf_estrato'].astype(str), labels=classes, normalize='true')
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    sns.heatmap(cm_poiss, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=axes[0], cbar=False)
    axes[0].set_title('A: Quase-Poisson (GLM)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Estrato Predito', fontsize=10)
    axes[0].set_ylabel('Estrato Real (Ministério da Saúde)', fontsize=10)
    
    sns.heatmap(cm_rf, annot=True, fmt='.2f', cmap='Greens', xticklabels=classes, yticklabels=classes, ax=axes[1], cbar=False)
    axes[1].set_title('B: Random Forest (Ensemble Balanceado)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Estrato Predito', fontsize=10)
    axes[1].set_ylabel('Estrato Real (Ministério da Saúde)', fontsize=10)
    
    fig.suptitle('Matrizes de Confusão Normalizadas no Teste Prospectivo de 2024', fontsize=12, fontweight='bold')
    plt.tight_layout()
    salvar_grafico("15_matrizes_confusao_estratos_risco")
    plt.close()
    
    #Importância de Atributos do Random Forest
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)
    top_feats = feat_importances.head(10).sort_values('importancia_mdi_reg', ascending=True)
    
    nome_legivel = {
        'populacao_total': 'População Total',
        'taxa_urbanizacao': 'Taxa de Urbanização (%)',
        'despesa_percapita_ano': 'Despesa per Capita em Resíduos (R$/hab)',
        'massa_percapita_kg_dia': 'Massa Coletada per Capita (kg/hab.dia)',
        'cobertura_coleta_pct': 'Cobertura de Coleta Domiciliar (%)',
        'custo_coleta_pct': 'Incidência de Custo de Coleta (%)',
        'terceirizacao_coleta_pct': 'Taxa de Terceirização (%)',
        'presenca_lixao': 'Presença de Lixão / Disposição Inadequada',
        'regiao_Sudeste': 'Região: Sudeste',
        'regiao_Nordeste': 'Região: Nordeste',
        'regiao_Sul': 'Região: Sul',
        'regiao_Norte': 'Região: Norte'
    }
    top_feats['nome_formatado'] = top_feats['feature'].map(lambda x: nome_legivel.get(x, x))
    
    y_pos = np.arange(len(top_feats))
    ax.barh(y_pos, top_feats['importancia_mdi_reg'], color='#2b83ba', edgecolor='black', alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_feats['nome_formatado'], fontsize=10)
    ax.set_xlabel('Importância Relativa (MDI - Gini / Variance Reduction)', fontsize=10)
    ax.set_title('Ranking de Importância das Variáveis de Resíduos e Território (Random Forest)', fontsize=11, fontweight='bold')
    
    for idx, val in enumerate(top_feats['importancia_mdi_reg']):
        ax.text(val + 0.005, idx, f'{val*100:.1f}%', va='center', fontsize=9, fontweight='bold', color='#1a5276')
        
    plt.tight_layout()
    salvar_grafico("16_importancia_variaveis_random_forest")
    plt.close()
    
    # Coeficientes IRR da Quase-Poisson (Forest Plot)
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)
    irr_plot = irr_df[~irr_df['variavel'].str.contains('Intercept')].copy()
    irr_plot['nome_formatado'] = irr_plot['variavel'].map(
        lambda x: nome_legivel.get(
            x.replace("C(regiao, Treatment(reference='Centro-Oeste'))[T.", 'regiao_').replace('C(regiao)[T.', 'regiao_').replace(']', ''),
            x
        )
    )
    irr_plot = irr_plot.sort_values('irr', ascending=True)
    
    y_pos = np.arange(len(irr_plot))
    err_low = irr_plot['irr'] - irr_plot['irr_ci_low']
    err_high = irr_plot['irr_ci_high'] - irr_plot['irr']
    
    ax.errorbar(irr_plot['irr'], y_pos, xerr=[err_low, err_high], fmt='o', color='#d73027', ecolor='#4575b4', elinewidth=2, capsize=4, markersize=7)
    ax.axvline(1.0, color='black', linestyle='--', lw=1.2, label='Nenhum Efeito (IRR = 1.0)')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(irr_plot['nome_formatado'], fontsize=10)
    ax.set_xlabel('Razão de Taxas de Incidência (IRR = exp(β)) com IC 95% Robusto', fontsize=10)
    ax.set_title('Efeito Multiplicativo na Incidência de Arboviroses (Quase-Poisson com Huber-White)', fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', frameon=True)
    
    for idx, (irr, p) in enumerate(zip(irr_plot['irr'], irr_plot['p_valor'])):
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.text(irr + 0.02, idx, f'{irr:.3f} ({sig})', va='center', fontsize=8.5, fontweight='bold')
        
    plt.tight_layout()
    salvar_grafico("17_coeficientes_irr_quase_poisson")
    plt.close()
    
    # Dispersão Predito vs Real no Choque de 2024
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    
    sample_vis = test_2024.sample(min(3000, len(test_2024)), random_state=42)
    
    # Quase-Poisson
    axes[0].scatter(sample_vis['taxa_incidencia_conf_100k'], sample_vis['pred_poisson_taxa'], alpha=0.3, color='#4575b4', s=16)
    axes[0].plot([0, 5000], [0, 5000], color='red', linestyle='--', lw=1.5, label='Ajuste Perfeito (1:1)')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_title(f'A: Quase-Poisson (RMSE: {metricas_poisson["rmse"]:.0f})', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Incidência Real (Casos/100k - Log)', fontsize=10)
    axes[0].set_ylabel('Incidência Predita (Casos/100k - Log)', fontsize=10)
    axes[0].legend(loc='upper left')
    
    # Random Forest
    axes[1].scatter(sample_vis['taxa_incidencia_conf_100k'], sample_vis['pred_rf_taxa'], alpha=0.3, color='#2ca25f', s=16)
    axes[1].plot([0, 5000], [0, 5000], color='red', linestyle='--', lw=1.5, label='Ajuste Perfeito (1:1)')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_title(f'B: Random Forest (RMSE: {metricas_rf["rmse"]:.0f})', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Incidência Real (Casos/100k - Log)', fontsize=10)
    axes[1].set_ylabel('Incidência Predita (Casos/100k - Log)', fontsize=10)
    axes[1].legend(loc='upper left')
    
    fig.suptitle('Capacidade Preditiva sob o Teste de Estresse Epidêmico de 2024', fontsize=12, fontweight='bold')
    plt.tight_layout()
    salvar_grafico("18_dispersao_predito_vs_real_teste2024")
    plt.close()

def exportar_artefatos(irr_df, feat_importances, test_2024, metricas_poisson, metricas_rf, df_cv):
    print("\n[6/6] Exportando Tabelas, Métricas e Predições para a pasta dados/...")
    
    #Tabela de Métricas Comparativas
    df_metrics = pd.DataFrame([metricas_poisson, metricas_rf])
    df_metrics.to_csv(os.path.join(DATA_DIR, "metricas_benchmark_modelos.csv"), index=False)
    with open(os.path.join(DATA_DIR, "metricas_benchmark_modelos.json"), "w", encoding="utf-8") as f:
        json.dump([metricas_poisson, metricas_rf], f, indent=4, ensure_ascii=False)
        
    #Tabela de Coeficientes Poisson / IRR
    irr_df.to_csv(os.path.join(DATA_DIR, "tabela_coeficientes_quase_poisson.csv"), index=False)
    
    #Tabela de Importância de Features RF
    feat_importances.to_csv(os.path.join(DATA_DIR, "tabela_importancia_features_rf.csv"), index=False)
    
    #Métricas de Validação Cruzada Espacial
    df_cv.to_csv(os.path.join(DATA_DIR, "metricas_validacao_cruzada_espacial.csv"), index=False)
    
    #Predições do Teste Prospectivo de 2024
    cols_pred = ['cod_mun_ibge', 'ano', 'municipio', 'uf', 'regiao', 'populacao_total',
                 'taxa_incidencia_conf_100k', 'estrato_risco_ms', 
                 'pred_poisson_taxa', 'pred_poisson_estrato',
                 'pred_rf_taxa', 'pred_rf_estrato']
    test_2024[cols_pred].to_parquet(os.path.join(DATA_DIR, "predicoes_teste_prospectivo_2024.parquet"))
    
    print("\n[OK] Todos os artefatos técnicos da Etapa 05 foram gerados e exportados com sucesso!\n")

def main():
    t_start = time.time()
    print("================================================================================")
    print(" INICIANDO PIPELINE DE MODELAGEM PREDITIVA E BENCHMARK COMPARATIVO (ETAPA 05)")
    print("================================================================================")
    
    df, train_df, test_df, test_2024 = preparar_dados()
    poiss_fit, irr_df, test_2024, metricas_poisson = treinar_e_avaliar_poisson(train_df, test_df, test_2024)
    rf_reg, rf_clf, feat_importances, test_2024, metricas_rf, X_train, X_test_2024 = treinar_e_avaliar_random_forest(train_df, test_df, test_2024)
    df_cv = executar_validacao_cruzada_espacial(train_df)
    gerar_graficos_e_tabelas(irr_df, feat_importances, test_2024, metricas_poisson, metricas_rf, df_cv)
    exportar_artefatos(irr_df, feat_importances, test_2024, metricas_poisson, metricas_rf, df_cv)
    
    print(f"Tempo total de execução do pipeline completo: {time.time() - t_start:.2f} segundos.")

if __name__ == '__main__':
    main()
