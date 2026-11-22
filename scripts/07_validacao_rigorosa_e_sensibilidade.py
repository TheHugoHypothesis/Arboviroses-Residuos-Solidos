#!/usr/bin/env python3
"""
Script 07: Validação e Análise de Sensibilidade
---------------------------------------------------------------------------------
Executa as verificações científicas e correções metodológicas mapeadas na auditoria:

1. Split temporal ANTES da imputação (zero vazamento de 2023-2024 para 2019-2022).
2. Referência regional (Centro-Oeste) e 27 UFs reais no Spatial GroupKFold.
3. Sensibilidade da Quase-Poisson: Bruto HC1 vs Winsorizado p99 HC1 vs Cluster por Município.
4. Avaliação separada para 2023 (ano regular) e 2024 (choque), com Baseline histórico.
5. Permutation Importance no RF (além do MDI) em 2023 e 2024.
6. PDP da cobertura de coleta treinado exclusivamente em 2019-2022.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, balanced_accuracy_score, f1_score,
    recall_score, precision_score, confusion_matrix
)
import statsmodels.api as sm
import statsmodels.formula.api as smf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dados")

FEATURE_IMPUTE = [
    'cobertura_coleta_pct', 'despesa_percapita_ano', 'massa_percapita_kg_dia',
    'custo_coleta_pct', 'terceirizacao_coleta_pct', 'presenca_lixao', 'populacao_urbana'
]

FEATURE_COLS = [
    'cobertura_coleta_pct', 'despesa_percapita_ano', 'massa_percapita_kg_dia',
    'custo_coleta_pct', 'terceirizacao_coleta_pct', 'presenca_lixao',
    'taxa_urbanizacao', 'populacao_total'
]

CLASSES = ['Baixo', 'Médio', 'Alto', 'Epidêmico']


def classificar_estrato_ms(taxa):
    if taxa < 100.0:
        return 'Baixo'
    elif taxa <= 300.0:
        return 'Médio'
    elif taxa <= 500.0:
        return 'Alto'
    else:
        return 'Epidêmico'


def preparar_dados_sem_vazamento():
    df = pd.read_parquet(os.path.join(DATA_DIR, "tabulao_integrado_2019_2024.parquet"))
    df = df[df['populacao_total'] >= 500].sort_values(['cod_mun_ibge', 'ano']).copy()

    #split temporal ANTES de qualquer imputacao
    train = df[df['ano'] <= 2022].copy()
    test_23 = df[df['ano'] == 2023].copy()
    test_24 = df[df['ano'] == 2024].copy()

    #imputação intra-municipal dentro do treino (2019-2022)
    for col in FEATURE_IMPUTE:
        train[col] = train.groupby('cod_mun_ibge')[col].transform(lambda x: x.ffill().bfill())

    reg_medians = train.groupby('regiao')[FEATURE_IMPUTE].median()
    glob_medians = train[FEATURE_IMPUTE].median()

    for col in FEATURE_IMPUTE:
        train[col] = train.apply(
            lambda r: reg_medians.loc[r['regiao'], col] if pd.isna(r[col]) else r[col], axis=1
        ).fillna(glob_medians[col])

    #Imputação causal para 2023 (ffill a partir de 2019-2022) e 2024 (ffill a partir de <=2023)
    last_train = train.sort_values('ano').groupby('cod_mun_ibge')[FEATURE_IMPUTE].last()
    for col in FEATURE_IMPUTE:
        # 2023 herda último valor conhecido do treino se ausente
        mask_23 = test_23[col].isna()
        test_23.loc[mask_23, col] = test_23.loc[mask_23, 'cod_mun_ibge'].map(last_train[col])
        test_23[col] = test_23.apply(
            lambda r: reg_medians.loc[r['regiao'], col] if pd.isna(r[col]) else r[col], axis=1
        ).fillna(glob_medians[col])

    last_until_23 = pd.concat([train, test_23]).sort_values('ano').groupby('cod_mun_ibge')[FEATURE_IMPUTE].last()
    for col in FEATURE_IMPUTE:
        mask_24 = test_24[col].isna()
        test_24.loc[mask_24, col] = test_24.loc[mask_24, 'cod_mun_ibge'].map(last_until_23[col])
        test_24[col] = test_24.apply(
            lambda r: reg_medians.loc[r['regiao'], col] if pd.isna(r[col]) else r[col], axis=1
        ).fillna(glob_medians[col])

    for part in [train, test_23, test_24]:
        part['taxa_urbanizacao'] = (part['populacao_urbana'] / part['populacao_total'].clip(lower=1) * 100).clip(0, 100)
        part['ln_pop'] = np.log(part['populacao_total'].clip(lower=100))
        part['estrato_risco_ms'] = pd.Categorical(part['estrato_risco_ms'], categories=CLASSES, ordered=True)

    return train, test_23, test_24


def avaliar_binaria(y_true_taxa, y_pred_taxa=None, y_pred_estrato=None):
    y_true_bin = (y_true_taxa >= 500.0).astype(int)
    if y_pred_taxa is not None:
        y_pred_bin = (y_pred_taxa >= 500.0).astype(int)
    else:
        y_pred_bin = (pd.Series(y_pred_estrato) == 'Epidêmico').astype(int).values
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
    return {
        'acc_bin': float(accuracy_score(y_true_bin, y_pred_bin)),
        'sensibilidade': float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
        'especificidade': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'precisao': float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
        'f1_epidemico': float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
    }


def avaliar_regressao_e_4classes(y_true_taxa, y_true_cls, y_pred_taxa, y_pred_cls):
    return {
        'rmse': float(np.sqrt(mean_squared_error(y_true_taxa, y_pred_taxa))),
        'mae': float(mean_absolute_error(y_true_taxa, y_pred_taxa)),
        'r2': float(r2_score(y_true_taxa, y_pred_taxa)),
        'acc_4c': float(accuracy_score(y_true_cls, y_pred_cls)),
        'bal_acc_4c': float(balanced_accuracy_score(y_true_cls, y_pred_cls)),
        'f1_macro_4c': float(f1_score(y_true_cls, y_pred_cls, labels=CLASSES, average='macro', zero_division=0))
    }


def rodar_quase_poisson_sensibilidade(train, test_23, test_24):
    formula = (
        "arbo_conf_tot ~ cobertura_coleta_pct + despesa_percapita_ano + "
        "massa_percapita_kg_dia + custo_coleta_pct + terceirizacao_coleta_pct + "
        "presenca_lixao + taxa_urbanizacao + C(regiao, Treatment(reference='Centro-Oeste'))"
    )
    p99_desp = float(train['despesa_percapita_ano'].quantile(0.99))

    # A: Bruto HC1
    fit_hc1 = smf.glm(formula=formula, data=train, family=sm.families.Poisson(), offset=train['ln_pop']).fit(cov_type='HC1')

    # B: Winsorizado p99 (treino e teste truncados pelo p99 do treino) + HC1
    train_w = train.copy()
    test23_w = test_23.copy()
    test24_w = test_24.copy()
    for d in [train_w, test23_w, test24_w]:
        d['despesa_percapita_ano'] = d['despesa_percapita_ano'].clip(upper=p99_desp)

    fit_w_hc1 = smf.glm(formula=formula, data=train_w, family=sm.families.Poisson(), offset=train_w['ln_pop']).fit(cov_type='HC1')

    # C: Winsorizado p99 + Clusterizado por Município
    fit_w_cl = smf.glm(formula=formula, data=train_w, family=sm.families.Poisson(), offset=train_w['ln_pop']).fit(
        cov_type='cluster', cov_kwds={'groups': train_w['cod_mun_ibge']}
    )

    def extrair_tabela(fit_obj, nome):
        ci = fit_obj.conf_int()
        return pd.DataFrame({
            'modelo': nome,
            'variavel': fit_obj.params.index,
            'beta': fit_obj.params.values,
            'se': fit_obj.bse.values,
            'p_valor': fit_obj.pvalues.values,
            'irr': np.exp(fit_obj.params.values),
            'irr_ci_low': np.exp(ci[0].values),
            'irr_ci_high': np.exp(ci[1].values)
        })

    coefs_all = pd.concat([
        extrair_tabela(fit_hc1, 'Bruto_HC1'),
        extrair_tabela(fit_w_hc1, 'Winsor_p99_HC1'),
        extrair_tabela(fit_w_cl, 'Winsor_p99_Cluster_Mun')
    ], ignore_index=True)

    # Predições 2023 e 2024 para Bruto e Winsorizado
    preds = {}
    for nome, fit_obj, t23, t24 in [
        ('QuasePoisson_Bruto', fit_hc1, test_23, test_24),
        ('QuasePoisson_Winsor_p99', fit_w_hc1, test23_w, test24_w)
    ]:
        for ano_lbl, t_df in [('2023', t23), ('2024', t24)]:
            cnt = fit_obj.predict(t_df, offset=t_df['ln_pop'])
            taxa = (cnt / t_df['populacao_total']) * 100000.0
            cls_p = taxa.apply(classificar_estrato_ms).astype(str).values
            m = avaliar_regressao_e_4classes(
                t_df['taxa_incidencia_conf_100k'].values,
                t_df['estrato_risco_ms'].astype(str).values,
                taxa.values, cls_p
            )
            m.update(avaliar_binaria(t_df['taxa_incidencia_conf_100k'].values, y_pred_taxa=taxa.values))
            preds[f"{nome}_{ano_lbl}"] = m

    return p99_desp, coefs_all, preds


def rodar_rf_e_cv_espacial(train, test_23, test_24):
    X_train = pd.get_dummies(train[FEATURE_COLS + ['regiao']], columns=['regiao'], drop_first=True)
    X_23 = pd.get_dummies(test_23[FEATURE_COLS + ['regiao']], columns=['regiao'], drop_first=True).reindex(columns=X_train.columns, fill_value=0)
    X_24 = pd.get_dummies(test_24[FEATURE_COLS + ['regiao']], columns=['regiao'], drop_first=True).reindex(columns=X_train.columns, fill_value=0)

    y_tr_reg = train['taxa_incidencia_conf_100k'].values
    y_tr_clf = train['estrato_risco_ms'].astype(str).values

    # Spatial GroupKFold com 27 UFs reais
    gkf = GroupKFold(n_splits=5)
    groups = train['uf'].values
    cv_folds = []
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_tr_reg, groups=groups), 1):
        rf_r = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
        rf_r.fit(X_train.iloc[tr_idx], y_tr_reg[tr_idx])
        p_r = rf_r.predict(X_train.iloc[val_idx])

        rf_c = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
        rf_c.fit(X_train.iloc[tr_idx], y_tr_clf[tr_idx])
        p_c = rf_c.predict(X_train.iloc[val_idx])

        cv_folds.append({
            'fold': fold,
            'num_ufs_val': int(pd.Series(groups[val_idx]).nunique()),
            'n_val': int(len(val_idx)),
            'rmse': float(np.sqrt(mean_squared_error(y_tr_reg[val_idx], p_r))),
            'mae': float(mean_absolute_error(y_tr_reg[val_idx], p_r)),
            'r2': float(r2_score(y_tr_reg[val_idx], p_r)),
            'bal_acc': float(balanced_accuracy_score(y_tr_clf[val_idx], p_c)),
            'f1_macro': float(f1_score(y_tr_clf[val_idx], p_c, labels=CLASSES, average='macro', zero_division=0))
        })

    # Modelos finais no treino completo (2019-2022)
    rf_reg = RandomForestRegressor(n_estimators=200, max_depth=14, min_samples_split=10, min_samples_leaf=4, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train, y_tr_reg)

    rf_clf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=10, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_clf.fit(X_train, y_tr_clf)

    rf_preds = {}
    perm_imp = {}
    for ano_lbl, X_te, t_df in [('2023', X_23, test_23), ('2024', X_24, test_24)]:
        y_te_taxa = t_df['taxa_incidencia_conf_100k'].values
        y_te_cls = t_df['estrato_risco_ms'].astype(str).values
        p_reg = rf_reg.predict(X_te)
        p_clf = rf_clf.predict(X_te)

        m = avaliar_regressao_e_4classes(y_te_taxa, y_te_cls, p_reg, p_clf)
        m_bin_reg = avaliar_binaria(y_te_taxa, y_pred_taxa=p_reg)
        m_bin_clf = avaliar_binaria(y_te_taxa, y_pred_estrato=p_clf)
        rf_preds[f"RF_{ano_lbl}"] = {'geral': m, 'bin_regressor': m_bin_reg, 'bin_classificador': m_bin_clf}

        pi = permutation_importance(rf_reg, X_te, y_te_taxa, n_repeats=10, random_state=42, scoring='neg_mean_absolute_error', n_jobs=-1)
        perm_imp[ano_lbl] = {col: float(val) for col, val in sorted(zip(X_train.columns, pi.importances_mean), key=lambda x: -x[1])}

    mdi_imp = {col: float(val) for col, val in sorted(zip(X_train.columns, rf_reg.feature_importances_), key=lambda x: -x[1])}

    # PDP da cobertura treinado APENAS no treino (2019-2022) em log10(taxa+1)
    train_pdp = train[(train['populacao_total'] > 1000) & (train['taxa_incidencia_conf_100k'] <= 10000)].copy()
    X_pdp = pd.get_dummies(train_pdp[FEATURE_COLS + ['regiao', 'ano']], columns=['regiao', 'ano'], drop_first=True)
    y_pdp = np.log10(train_pdp['taxa_incidencia_conf_100k'] + 1.0)
    rf_pdp = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    rf_pdp.fit(X_pdp, y_pdp)
    pdp = partial_dependence(rf_pdp, X_pdp, ['cobertura_coleta_pct'], grid_resolution=25)
    pdp_pts = [{'cobertura': float(x), 'taxa_marginal': float(10**v - 1)} for x, v in zip(pdp['grid_values'][0], pdp['average'][0])]

    return cv_folds, rf_preds, mdi_imp, perm_imp, pdp_pts


def rodar_baseline_historico(train, test_23, test_24):
    hist_mean = train.groupby('cod_mun_ibge')['taxa_incidencia_conf_100k'].mean()
    glob_mean = float(train['taxa_incidencia_conf_100k'].mean())
    res = {}
    for ano_lbl, t_df in [('2023', test_23), ('2024', test_24)]:
        pred = t_df['cod_mun_ibge'].map(hist_mean).fillna(glob_mean).values
        pred_cls = pd.Series(pred).apply(classificar_estrato_ms).values
        m = avaliar_regressao_e_4classes(
            t_df['taxa_incidencia_conf_100k'].values,
            t_df['estrato_risco_ms'].astype(str).values,
            pred, pred_cls
        )
        m.update(avaliar_binaria(t_df['taxa_incidencia_conf_100k'].values, y_pred_taxa=pred))
        res[f"Baseline_Hist_{ano_lbl}"] = m
    return res


def main():
    print("[1/5] Preparando dados sem vazamento temporal...")
    train, test_23, test_24 = preparar_dados_sem_vazamento()
    print(f"      Treino: {len(train):,} | Teste 2023: {len(test_23):,} | Teste 2024: {len(test_24):,}")
    print(f"      UFs únicas no treino: {train['uf'].nunique()}")

    print("[2/5] Rodando sensibilidade Quase-Poisson (Bruto, Winsor p99, Cluster Municipal)...")
    p99_desp, coefs_all, qp_preds = rodar_quase_poisson_sensibilidade(train, test_23, test_24)
    coefs_all.to_csv(os.path.join(DATA_DIR, "validacao_sensibilidade_quase_poisson.csv"), index=False)

    print("[3/5] Rodando Spatial GroupKFold (27 UFs), Random Forest, Permutation Importance e PDP Treino...")
    cv_folds, rf_preds, mdi_imp, perm_imp, pdp_pts = rodar_rf_e_cv_espacial(train, test_23, test_24)

    print("[4/5] Rodando Baseline histórico (média municipal 2019-2022)...")
    base_preds = rodar_baseline_historico(train, test_23, test_24)

    out = {
        'p99_despesa_treino': p99_desp,
        'quase_poisson_preds': qp_preds,
        'spatial_cv_27_ufs': cv_folds,
        'rf_preds': rf_preds,
        'rf_mdi_importance': mdi_imp,
        'rf_permutation_importance_mae': perm_imp,
        'baseline_historico': base_preds,
        'pdp_apenas_treino_2019_2022': pdp_pts
    }
    out_path = os.path.join(DATA_DIR, "validacao_sensibilidade.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"[5/5] Validação concluída! Resultados salvos em {out_path}")


if __name__ == '__main__':
    main()
