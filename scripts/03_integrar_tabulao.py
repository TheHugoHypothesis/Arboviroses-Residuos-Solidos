#!/usr/bin/env python3
"""
Script 03: Integração do Tabulão Analítico (Saúde + Resíduos)
------------------------------------------------------------

Realiza a junção entre o painel de saúde e o painel de saneamento/resíduos,
calcula as taxas de incidência por 100 mil habitantes, define os 4 estratos
de risco epidemiológico segundo as diretrizes do Ministério da Saúde e gera
as variáveis analíticas para modelagem preditiva (Poisson e Random Forest).
"""

import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dados")

def classificar_estrato_risco(taxa):
    if pd.isna(taxa):
        return np.nan
    if taxa < 100.0:
        return 'Baixo'
    elif taxa <= 300.0:
        return 'Médio'
    elif taxa <= 500.0:
        return 'Alto'
    else:
        return 'Epidêmico'

def main():
    print("[+] Lendo painéis de saúde e resíduos...")
    path_saude = os.path.join(DATA_DIR, "painel_saude_2019_2024.parquet")
    path_residuos = os.path.join(DATA_DIR, "painel_residuos_2019_2024.parquet")
    
    df_saude = pd.read_parquet(path_saude)
    df_residuos = pd.read_parquet(path_residuos)
    
    #Padronização de tipos de chaves
    if 'cod_mun' in df_saude.columns and 'cod_mun_ibge' not in df_saude.columns:
        df_saude.rename(columns={'cod_mun': 'cod_mun_ibge'}, inplace=True)
        
    df_saude['cod_mun_ibge'] = df_saude['cod_mun_ibge'].astype(str).str.zfill(6)
    df_saude['ano'] = df_saude['ano'].astype(int)
    
    df_residuos['cod_mun_ibge'] = df_residuos['cod_mun_ibge'].astype(str).str.zfill(6)
    df_residuos['ano'] = df_residuos['ano'].astype(int)
    
    print(f"  Saúde: {len(df_saude):,} registros | Resíduos: {len(df_residuos):,} registros")
    
    # Inner / Left join: a base de resíduos fornece os municípios oficiais e população
    # Municípios presentes no painel de resíduos sem notificação de saúde recebem 0 casos
    tabulao = df_residuos.merge(df_saude, on=['cod_mun_ibge', 'ano'], how='left')
    
    # Preencher contagens de saúde com zero se ausente na base do SINAN (ausência de notificação = 0 casos)
    health_cols = ['dengue_notif', 'dengue_conf', 'chik_notif', 'chik_conf', 
                   'zika_notif', 'zika_conf', 'fa_casos', 'arbo_notif_tot', 'arbo_conf_tot']
    for c in health_cols:
        tabulao[c] = tabulao[c].fillna(0).astype(int)
        
    print(f"  Tabulão Integrado inicial: {len(tabulao):,} linhas.")
    
    # Cálculo das Taxas de Incidência por 100 mil habitantes
    # Exclui populações nulas ou inconsistentes (< 50 habitantes)
    valid_pop = (tabulao['populacao_total'] > 50)
    
    tabulao['taxa_incidencia_notif_100k'] = np.where(
        valid_pop,
        (tabulao['arbo_notif_tot'] / tabulao['populacao_total']) * 100000.0,
        np.nan
    )
    
    tabulao['taxa_incidencia_conf_100k'] = np.where(
        valid_pop,
        (tabulao['arbo_conf_tot'] / tabulao['populacao_total']) * 100000.0,
        np.nan
    )
    
    tabulao['dengue_incidencia_conf_100k'] = np.where(
        valid_pop,
        (tabulao['dengue_conf'] / tabulao['populacao_total']) * 100000.0,
        np.nan
    )
    
    # Estratos de risco epidemiológico do Ministério da Saúde
    tabulao['estrato_risco_ms'] = tabulao['taxa_incidencia_conf_100k'].apply(classificar_estrato_risco)
    tabulao['estrato_risco_notif_ms'] = tabulao['taxa_incidencia_notif_100k'].apply(classificar_estrato_risco)
    
    #Termo de compensação (offset) logarítmico para regressão de Poisson
    tabulao['ln_pop'] = np.where(valid_pop, np.log(tabulao['populacao_total']), np.nan)
    
    #Particionamento temporal descrito no artigo (Treino 2019-2022 vs Teste Prospectivo 2023-2024)
    tabulao['particao_temporal'] = np.where(tabulao['ano'] <= 2022, 'Treino_Validacao', 'Teste_Prospectivo')
    
    # Ordenação final
    tabulao.sort_values(by=['ano', 'cod_mun_ibge'], inplace=True)
    
    # Exportação
    out_parquet = os.path.join(DATA_DIR, "tabulao_integrado_2019_2024.parquet")
    out_csv = os.path.join(DATA_DIR, "tabulao_integrado_2019_2024.csv")
    tabulao.to_parquet(out_parquet, index=False)
    tabulao.to_csv(out_csv, index=False)
    
    print("\n[OK] Tabulão Analítico Integrado exportado com sucesso!")
    print(f"  Total de registros: {len(tabulao):,}")
    print(f"  Municípios únicos: {tabulao['cod_mun_ibge'].nunique():,}")
    print(f"  Anos: {sorted(tabulao['ano'].unique())}")
    print(f"  Distribuição por partição temporal:\n{tabulao['particao_temporal'].value_counts()}")
    print(f"\n  Distribuição dos Estratos de Risco (Casos Confirmados):\n{tabulao['estrato_risco_ms'].value_counts(dropna=False)}")
    print(f"\n  Arquivos salvos:")
    print(f"    - {out_parquet}")
    print(f"    - {out_csv}\n")

if __name__ == '__main__':
    main()
