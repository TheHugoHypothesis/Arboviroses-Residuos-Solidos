#!/usr/bin/env python3
"""
Script 01: Processamento e Agregação dos Microdados de Saúde (SINAN)
-------------------------------------------------------------------
Agrega as notificações individuais de Dengue, Chikungunya e Zika (2019-2024)
em contagens anuais por município de residência (ID_MN_RESI, código IBGE de 6 dígitos).
Gera o painel municipal com notificações totais e casos confirmados.
"""

import os
import glob
import time
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_HEALTH_DIR = os.path.join(PROJECT_ROOT, "Base de Dados", "Saúde")
OUTPUT_DIR = os.path.join(BASE_DIR, "dados")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_sinan_disease(disease_name, folder_name, file_prefix, confirmed_values):
    print(f"\n[+] Processando dados de {disease_name.upper()}...")
    csv_dir = os.path.join(DATA_HEALTH_DIR, folder_name, "csv")
    csv_files = sorted(glob.glob(os.path.join(csv_dir, f"{file_prefix}*.csv")))
    
    records = []
    
    for fpath in csv_files:
        fname = os.path.basename(fpath)
        #Extrai ano do nome do arquivo (ex: DENGBR19 -> 2019)
        yr_short = fname.replace(file_prefix, "").replace(".csv", "")
        year = int("20" + yr_short)
        print(f"  -> Lendo {fname} (Ano {year})...", end=" ", flush=True)
        t0 = time.time()
        
        # Leitura por blocos para otimização de memória
        chunk_aggs = []
        for chunk in pd.read_csv(fpath, usecols=['ID_MN_RESI', 'CLASSI_FIN'], chunksize=500000, low_memory=False):
            # Remover registros sem município de residência
            chunk = chunk.dropna(subset=['ID_MN_RESI']).copy()
            # Tratar código de município para numérico inteiro
            chunk['cod_mun'] = pd.to_numeric(chunk['ID_MN_RESI'], errors='coerce')
            chunk = chunk.dropna(subset=['cod_mun'])
            chunk['cod_mun'] = chunk['cod_mun'].astype(int).astype(str).str.zfill(6)
            
            # Filtro de casos confirmados
            chunk['is_conf'] = chunk['CLASSI_FIN'].isin(confirmed_values).astype(int)
            
            agg = chunk.groupby('cod_mun').agg(
                notif=('CLASSI_FIN', 'count'),
                conf=('is_conf', 'sum')
            ).reset_index()
            chunk_aggs.append(agg)
        
        if chunk_aggs:
            df_year = pd.concat(chunk_aggs).groupby('cod_mun').sum().reset_index()
            df_year['ano'] = year
            df_year.rename(columns={
                'notif': f'{disease_name}_notif',
                'conf': f'{disease_name}_conf'
            }, inplace=True)
            records.append(df_year)
            dt = time.time() - t0
            print(f"OK! {len(df_year):,} municípios processados em {dt:.1f}s.")
        else:
            print("Vazio ou sem registros válidos.")
            
    if records:
        df_all = pd.concat(records, ignore_index=True)
        return df_all
    return pd.DataFrame()

def process_febre_amarela():
    print("\n[+] Processando dados de Febre Amarela para análise exploratória...")
    fpath = os.path.join(DATA_HEALTH_DIR, "Febre Amarela", "fa_casoshumanos_1994-2026.csv")
    if not os.path.exists(fpath):
        print("  Arquivo de febre amarela não encontrado.")
        return pd.DataFrame()
        
    df_fa = pd.read_csv(fpath, sep=';', encoding='latin1')
    df_fa = df_fa.dropna(subset=['ANO_IS', 'COD_MUN_LPI'])
    df_fa['ano'] = df_fa['ANO_IS'].astype(int)
    df_fa = df_fa[df_fa['ano'].between(2019, 2024)]
    df_fa['cod_mun'] = df_fa['COD_MUN_LPI'].astype(int).astype(str).str.zfill(6)
    
    agg = df_fa.groupby(['cod_mun', 'ano']).size().reset_index(name='fa_casos')
    print(f"  Total de casos humanos de Febre Amarela (2019-2024): {agg['fa_casos'].sum()} casos em {len(agg)} tuplas município-ano.")
    return agg

def main():
    start_total = time.time()
    
    #Dengue: 10 = Dengue, 11 = Dengue com sinais de alarme, 12 = Dengue grave
    dengue_conf_codes = [10, 11, 12, 10.0, 11.0, 12.0]
    df_dengue = process_sinan_disease("dengue", "Dengue", "DENGBR", dengue_conf_codes)
    
    #Chikungunya: 13 = Chikungunya confirmada
    chik_conf_codes = [13, 13.0]
    df_chik = process_sinan_disease("chik", "Chikungunha", "CHIKBR", chik_conf_codes)
    
    #Zika: 1 = Caso confirmado
    zika_conf_codes = [1, 1.0]
    df_zika = process_sinan_disease("zika", "Zika", "ZIKABR", zika_conf_codes)
    
    #Febre Amarela
    df_fa = process_febre_amarela()
    
    # Unificação do painel de arboviroses urbanas (Dengue, Chik, Zika)
    print("\n[+] Integrando painel municipal de arboviroses (2019-2024)...")
    # Base completa de (cod_mun, ano)
    all_keys = pd.concat([
        df_dengue[['cod_mun', 'ano']],
        df_chik[['cod_mun', 'ano']],
        df_zika[['cod_mun', 'ano']]
    ]).drop_duplicates()
    
    panel = all_keys.merge(df_dengue, on=['cod_mun', 'ano'], how='left')
    panel = panel.merge(df_chik, on=['cod_mun', 'ano'], how='left')
    panel = panel.merge(df_zika, on=['cod_mun', 'ano'], how='left')
    if not df_fa.empty:
        panel = panel.merge(df_fa, on=['cod_mun', 'ano'], how='left')
    else:
        panel['fa_casos'] = 0
        
    #Preencher NaNs com 0 para contagens
    count_cols = ['dengue_notif', 'dengue_conf', 'chik_notif', 'chik_conf', 
                  'zika_notif', 'zika_conf', 'fa_casos']
    for c in count_cols:
        if c in panel.columns:
            panel[c] = panel[c].fillna(0).astype(int)
            
    # Agregados totais de arboviroses do vetor Aedes aegypti
    panel['arbo_notif_tot'] = panel['dengue_notif'] + panel['chik_notif'] + panel['zika_notif']
    panel['arbo_conf_tot'] = panel['dengue_conf'] + panel['chik_conf'] + panel['zika_conf']
    
    panel.sort_values(by=['ano', 'cod_mun'], inplace=True)
    
    # Exportação
    out_parquet = os.path.join(OUTPUT_DIR, "painel_saude_2019_2024.parquet")
    out_csv = os.path.join(OUTPUT_DIR, "painel_saude_2019_2024.csv")
    panel.to_parquet(out_parquet, index=False)
    panel.to_csv(out_csv, index=False)
    
    print(f"\n[OK] Painel de Saúde gerado com sucesso!")
    print(f"  Linhas totais: {len(panel):,}")
    print(f"  Casos Notificados Totais (2019-2024): {panel['arbo_notif_tot'].sum():,}")
    print(f"  Casos Confirmados Totais (2019-2024): {panel['arbo_conf_tot'].sum():,}")
    print(f"  Arquivos salvos:")
    print(f"    - {out_parquet}")
    print(f"    - {out_csv}")
    print(f"Tempo total de execução: {time.time() - start_total:.1f} segundos.\n")

if __name__ == '__main__':
    main()
