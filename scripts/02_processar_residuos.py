#!/usr/bin/env python3
"""
Script 02: Processamento e Harmonização dos Dados de Resíduos Sólidos (SNIS e SINISA)
-----------------------------------------------------------------------------------
Harmoniza as bases do SNIS (2019-2022) e do SINISA (2023-2024), extraindo os indicadores
operacionais, financeiros, de cobertura e de destinação final, além dos denominadores
populacionais oficiais. Converte todos os códigos municipais para o padrão IBGE 6 dígitos.
"""

import os
import glob
import time
import openpyxl
import xlrd
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_WASTE_DIR = os.path.join(PROJECT_ROOT, "Base de Dados", "Resíduos")
OUTPUT_DIR = os.path.join(BASE_DIR, "dados")
os.makedirs(OUTPUT_DIR, exist_ok=True)

UF_TO_REGION = {
    'AC': 'Norte', 'AP': 'Norte', 'AM': 'Norte', 'PA': 'Norte', 'RO': 'Norte', 'RR': 'Norte', 'TO': 'Norte',
    'AL': 'Nordeste', 'BA': 'Nordeste', 'CE': 'Nordeste', 'MA': 'Nordeste', 'PB': 'Nordeste', 
    'PE': 'Nordeste', 'PI': 'Nordeste', 'RN': 'Nordeste', 'SE': 'Nordeste',
    'DF': 'Centro-Oeste', 'GO': 'Centro-Oeste', 'MT': 'Centro-Oeste', 'MS': 'Centro-Oeste',
    'ES': 'Sudeste', 'MG': 'Sudeste', 'RJ': 'Sudeste', 'SP': 'Sudeste',
    'PR': 'Sul', 'RS': 'Sul', 'SC': 'Sul'
}

def to_num(val):
    if val is None:
        return np.nan
    s = str(val).strip().replace('.', '').replace(',', '.') if isinstance(val, str) else val
    try:
        f = float(s)
        return np.nan if np.isnan(f) else f
    except (ValueError, TypeError):
        return np.nan

def clean_ibge6(code):
    if code is None:
        return None
    s = str(code).strip().split('.')[0]
    if not s or s in ['-', 'None', 'nan']:
        return None
    # Trunca se tiver 7 dígitos, adiciona zeros à esquerda se tiver menos de 6
    if len(s) >= 6:
        return s[:6].zfill(6)
    return s.zfill(6)

def process_snis_lixoes(year):
    p = os.path.join(DATA_WASTE_DIR, "Pastas anuais", f"{year}_SNIS_Planilhas_RS", f"Planilha_Unidades_Lixoes_Aterros_RS_{year}.xlsx")
    if not os.path.exists(p):
        return {}
    
    wb = openpyxl.load_workbook(p, read_only=True)
    sheet = wb.active
    rows = list(sheet.iter_rows(values_only=True))
    
    # Encontra cabeçalho
    header_idx = -1
    for i, r in enumerate(rows[:15]):
        if any('UP003' in str(c).upper() for c in r) or any('TIPO DE UNIDADE' in str(c).upper() for c in r):
            header_idx = i
            break
            
    if header_idx == -1:
        return {}
        
    code_col = 0
    type_col = 6 if year < 2022 else 10
    
    # Encontra colunas exatas
    for c_i, c_val in enumerate(rows[header_idx]):
        s = str(c_val).upper()
        if 'CÓDIGO' in s or 'UNDCOD' in s:
            if c_i < 3:
                code_col = c_i
        if 'UP003' in s or 'TIPO DE UNIDADE' in s:
            type_col = c_i
            
    muni_lixao = {}
    for r in rows[header_idx+1:]:
        if len(r) > max(code_col, type_col) and r[code_col] is not None:
            c6 = clean_ibge6(r[code_col])
            t_val = str(r[type_col]).lower() if r[type_col] is not None else ""
            if c6:
                is_lixao = 1 if 'lix' in t_val or 'controlado' in t_val else 0
                muni_lixao[c6] = max(muni_lixao.get(c6, 0), is_lixao)
                
    return muni_lixao

def process_snis_population(year):
    p = os.path.join(DATA_WASTE_DIR, "Pastas anuais", f"{year}_SNIS_Planilhas_RS", f"Planilha_Informacoes_RS_{year}.xlsx")
    wb = openpyxl.load_workbook(p, read_only=True)
    sheet = wb.active
    rows = list(sheet.iter_rows(values_only=True))
    
    # Procura linha de código POP_TOT e coluna de Código do município
    code_row_idx = -1
    for i in range(5, min(16, len(rows))):
        if any(str(c).strip().upper() == 'POP_TOT' for c in rows[i]):
            code_row_idx = i
            break
            
    if code_row_idx == -1:
        return {}
        
    pop_tot_col = -1
    pop_urb_col = -1
    for c_i, c_val in enumerate(rows[code_row_idx]):
        s = str(c_val).strip().upper()
        if s == 'POP_TOT':
            pop_tot_col = c_i
        elif s == 'POP_URB':
            pop_urb_col = c_i
            
    pop_dict = {}
    for r in rows[code_row_idx+1:]:
        if len(r) > pop_tot_col and r[0] is not None:
            c6 = clean_ibge6(r[0])
            if c6:
                p_tot = to_num(r[pop_tot_col])
                p_urb = to_num(r[pop_urb_col]) if pop_urb_col != -1 and len(r) > pop_urb_col else np.nan
                pop_dict[c6] = (p_tot, p_urb)
    return pop_dict

def process_snis_year(year):
    print(f"  -> Processando SNIS {year}...")
    ext = 'xls' if year == 2021 else 'xlsx'
    p_ind = os.path.join(DATA_WASTE_DIR, "Pastas anuais", f"{year}_SNIS_Planilhas_RS", f"Planilha_Indicadores_RS_{year}.{ext}")
    
    pop_dict = process_snis_population(year)
    lixao_dict = process_snis_lixoes(year)
    
    if ext == 'xlsx':
        wb = openpyxl.load_workbook(p_ind, read_only=True)
        sheet = wb.active
        rows = list(sheet.iter_rows(values_only=True))
    else:
        book = xlrd.open_workbook(p_ind)
        sh = book.sheet_by_index(0)
        rows = [[sh.cell_value(r, c) for c in range(sh.ncols)] for r in range(sh.nrows)]
        
    #Mapeia colunas na linha de códigos (ex: IN006, IN015, etc.)
    code_row_idx = -1
    for i in range(5, min(14, len(rows))):
        if any(str(c).strip().upper().startswith('IN00') for c in rows[i]):
            code_row_idx = i
            break
            
    col_map = {}
    for c_i, c_val in enumerate(rows[code_row_idx]):
        s = str(c_val).strip().upper()
        if s.startswith('IN0'):
            col_map[s] = c_i

    #Detecta dinamicamente colunas de Município e UF (em 2022 houve deslocamento pela coluna Código do IBGE)
    muni_col, uf_col = 1, 2
    for i in range(min(12, len(rows))):
        row_upper = [str(c).strip().upper() if c is not None else '' for c in rows[i][:10]]
        if 'UF' in row_upper:
            uf_col = row_upper.index('UF')
            for idx_c, val_c in enumerate(row_upper):
                if val_c in ['MUNICÍPIO', 'MUNICIPIO']:
                    muni_col = idx_c
                    break
            break
            
    data_list = []
    # Data rows começam logo após a linha de códigos
    for r in rows[code_row_idx+1:]:
        c_code = r[0]
        c6 = clean_ibge6(c_code)
        if not c6 or len(r) < 5:
            continue
            
        muni_nome = str(r[muni_col]).strip() if len(r) > muni_col and r[muni_col] is not None else ""
        uf = str(r[uf_col]).strip() if len(r) > uf_col and r[uf_col] is not None else ""
        regiao = UF_TO_REGION.get(uf, "")
        
        # Cobertura de coleta: IN015 (pop total) ou IN016 (pop urbana)
        cob_tot = to_num(r[col_map['IN015']]) if 'IN015' in col_map and len(r) > col_map['IN015'] else np.nan
        cob_urb = to_num(r[col_map['IN016']]) if 'IN016' in col_map and len(r) > col_map['IN016'] else np.nan
        cobertura = cob_tot if not np.isnan(cob_tot) else cob_urb
        
        # Massa per capita: IN021 (kg/hab.dia) ou IN022 / IN028
        massa = to_num(r[col_map['IN021']]) if 'IN021' in col_map and len(r) > col_map['IN021'] else np.nan
        if np.isnan(massa) and 'IN022' in col_map and len(r) > col_map['IN022']:
            massa = to_num(r[col_map['IN022']])
            
        # Despesa per capita: IN006 (R$/hab)
        despesa = to_num(r[col_map['IN006']]) if 'IN006' in col_map and len(r) > col_map['IN006'] else np.nan
        
        # Custos e terceirização
        custo_coleta = to_num(r[col_map['IN024']]) if 'IN024' in col_map and len(r) > col_map['IN024'] else np.nan
        terceirizacao = to_num(r[col_map['IN017']]) if 'IN017' in col_map and len(r) > col_map['IN017'] else np.nan
        custo_unit = to_num(r[col_map['IN023']]) if 'IN023' in col_map and len(r) > col_map['IN023'] else np.nan
        
        # Entulho / RCD
        rcd = to_num(r[col_map['IN029']]) if 'IN029' in col_map and len(r) > col_map['IN029'] else np.nan
        
        # Coleta seletiva e reciclagem
        col_seletiva = to_num(r[col_map['IN030']]) if 'IN030' in col_map and len(r) > col_map['IN030'] else np.nan
        reciclagem = to_num(r[col_map['IN031']]) if 'IN031' in col_map and len(r) > col_map['IN031'] else np.nan
        empregados = to_num(r[col_map['IN001']]) if 'IN001' in col_map and len(r) > col_map['IN001'] else np.nan
        
        p_tot, p_urb = pop_dict.get(c6, (np.nan, np.nan))
        tem_lixao = lixao_dict.get(c6, np.nan)
        
        data_list.append({
            'cod_mun_ibge': c6,
            'ano': year,
            'municipio': muni_nome,
            'uf': uf,
            'regiao': regiao,
            'respondeu_modulo_rs': 1,
            'populacao_total': p_tot,
            'populacao_urbana': p_urb,
            'cobertura_coleta_pct': cobertura,
            'massa_percapita_kg_dia': massa,
            'despesa_percapita_ano': despesa,
            'custo_coleta_pct': custo_coleta,
            'terceirizacao_coleta_pct': terceirizacao,
            'custo_unitario_coleta': custo_unit,
            'rcd_percapita_kg_ano': rcd,
            'coleta_seletiva_pct': col_seletiva,
            'recuperacao_reciclaveis_pct': reciclagem,
            'taxa_empregados_1000hab': empregados,
            'presenca_lixao': tem_lixao,
            'disposicao_inadequada_pct': np.nan # não medido diretamente no SNIS como percentual
        })
        
    df_snis = pd.DataFrame(data_list)
    print(f"    Total de municípios declarantes no SNIS {year}: {len(df_snis):,}")
    return df_snis

def process_sinisa_year(year):
    print(f"  -> Processando SINISA {year}...")
    p_ind = os.path.join(DATA_WASTE_DIR, "Pastas anuais", f"{year}_SINISA_RESIDUOS_Planilhas", f"SINISA_RESIDUOS_Indicadores_{year}.xlsx")
    wb = openpyxl.load_workbook(p_ind, read_only=True)
    sheet = wb.active
    rows = list(sheet.iter_rows(values_only=True))
    
    # Linha 11 (index 10) e Linha 13 (index 12) contêm os códigos e descrições dos indicadores
    col_map = {}
    for c_i in range(len(rows[10])):
        v11 = str(rows[10][c_i]).strip().upper() if rows[10][c_i] is not None else ''
        v13 = str(rows[12][c_i]).strip().upper() if len(rows) > 12 and rows[12][c_i] is not None else ''
        if v11:
            col_map[v11] = c_i
        if v13:
            col_map[v13] = c_i
            
    data_list = []
    # Linha 14 (index 13) começa os dados
    for r in rows[13:]:
        if len(r) < 10 or r[1] is None:
            continue
            
        respondeu_str = str(r[0]).strip().lower() if r[0] is not None else ""
        respondeu = 1 if respondeu_str == 'sim' else 0
        
        c6 = clean_ibge6(r[1])
        if not c6:
            continue
            
        muni_nome = str(r[2]).strip() if r[2] is not None else ""
        uf = str(r[3]).strip() if r[3] is not None else ""
        regiao = str(r[4]).strip() if r[4] is not None else UF_TO_REGION.get(uf, "")
        
        # População
        p_tot = to_num(r[col_map['DFE0001']]) if 'DFE0001' in col_map and len(r) > col_map['DFE0001'] else np.nan
        p_urb = to_num(r[col_map['DFE0002']]) if 'DFE0002' in col_map and len(r) > col_map['DFE0002'] else np.nan
        
        # Cobertura: IRS0001 (total) ou IRS0002 (urbana)
        cob_tot = to_num(r[col_map['IRS0001']]) if 'IRS0001' in col_map and len(r) > col_map['IRS0001'] else np.nan
        cob_urb = to_num(r[col_map['IRS0002']]) if 'IRS0002' in col_map and len(r) > col_map['IRS0002'] else np.nan
        cobertura = cob_tot if not np.isnan(cob_tot) else cob_urb
        
        # Massa per capita: IRS1004 (kg/hab.dia)
        massa = to_num(r[col_map['IRS1004']]) if 'IRS1004' in col_map and len(r) > col_map['IRS1004'] else np.nan
        
        # Despesa per capita: IFR2002 (R$/hab.ano)
        despesa = to_num(r[col_map['IFR2002']]) if 'IFR2002' in col_map and len(r) > col_map['IFR2002'] else np.nan
        
        # Custos e terceirização
        custo_coleta = to_num(r[col_map['IFR2009']]) if 'IFR2009' in col_map and len(r) > col_map['IFR2009'] else np.nan
        terceirizacao = to_num(r[col_map['IFR0004']]) if 'IFR0004' in col_map and len(r) > col_map['IFR0004'] else np.nan
        
        # Disposição inadequada (%)
        disp_inad = to_num(r[col_map['IRS3002']]) if 'IRS3002' in col_map and len(r) > col_map['IRS3002'] else np.nan
        tem_lixao = 1 if disp_inad > 0 else (0 if disp_inad == 0 else np.nan)
        
        # Pontos de entrega voluntária e coleta seletiva
        pts_entrega = to_num(r[col_map['IRS1003']]) if 'IRS1003' in col_map and len(r) > col_map['IRS1003'] else np.nan
        col_seletiva = to_num(r[col_map['IRS3001']]) if 'IRS3001' in col_map and len(r) > col_map['IRS3001'] else np.nan
        reciclagem = to_num(r[col_map['IRS3004']]) if 'IRS3004' in col_map and len(r) > col_map['IRS3004'] else np.nan
        empregados = to_num(r[col_map['IFR0001']]) if 'IFR0001' in col_map and len(r) > col_map['IFR0001'] else np.nan
        
        data_list.append({
            'cod_mun_ibge': c6,
            'ano': year,
            'municipio': muni_nome,
            'uf': uf,
            'regiao': regiao,
            'respondeu_modulo_rs': respondeu,
            'populacao_total': p_tot,
            'populacao_urbana': p_urb,
            'cobertura_coleta_pct': cobertura,
            'massa_percapita_kg_dia': massa,
            'despesa_percapita_ano': despesa,
            'custo_coleta_pct': custo_coleta,
            'terceirizacao_coleta_pct': terceirizacao,
            'custo_unitario_coleta': np.nan,
            'rcd_percapita_kg_ano': np.nan,
            'coleta_seletiva_pct': col_seletiva,
            'recuperacao_reciclaveis_pct': reciclagem,
            'taxa_empregados_1000hab': empregados,
            'presenca_lixao': tem_lixao,
            'disposicao_inadequada_pct': disp_inad
        })
        
    df_sinisa = pd.DataFrame(data_list)
    print(f"    Total de municípios no SINISA {year}: {len(df_sinisa):,} (Respondentes: {df_sinisa['respondeu_modulo_rs'].sum():,})")
    return df_sinisa

def main():
    t0 = time.time()
    print("\n[+] Iniciando harmonização dos dados de Resíduos Sólidos (2019-2024)...")
    dfs = []
    
    #Processa SNIS 2019 a 2022
    for y in [2019, 2020, 2021, 2022]:
        df_y = process_snis_year(y)
        dfs.append(df_y)
        
    # Processa SINISA 2023 e 2024
    for y in [2023, 2024]:
        df_y = process_sinisa_year(y)
        dfs.append(df_y)
        
    panel_residuos = pd.concat(dfs, ignore_index=True)
    panel_residuos.sort_values(by=['ano', 'cod_mun_ibge'], inplace=True)
    
    #Tratamentos básicos de consistência (limites percentuais plausíveis 0 a 100%)
    pct_cols = ['cobertura_coleta_pct', 'custo_coleta_pct', 'terceirizacao_coleta_pct', 
                'coleta_seletiva_pct', 'recuperacao_reciclaveis_pct', 'disposicao_inadequada_pct']
    for c in pct_cols:
        panel_residuos.loc[panel_residuos[c] < 0, c] = np.nan
        panel_residuos.loc[panel_residuos[c] > 100, c] = 100.0
        
    # Despesa e massa: remover valores negativos
    panel_residuos.loc[panel_residuos['despesa_percapita_ano'] < 0, 'despesa_percapita_ano'] = np.nan
    panel_residuos.loc[panel_residuos['massa_percapita_kg_dia'] < 0, 'massa_percapita_kg_dia'] = np.nan
    
    # Exportação
    out_parquet = os.path.join(OUTPUT_DIR, "painel_residuos_2019_2024.parquet")
    out_csv = os.path.join(OUTPUT_DIR, "painel_residuos_2019_2024.csv")
    panel_residuos.to_parquet(out_parquet, index=False)
    panel_residuos.to_csv(out_csv, index=False)
    
    print(f"\n[OK] Painel de Resíduos Sólidos gerado com sucesso!")
    print(f"  Linhas totais: {len(panel_residuos):,}")
    print(f"  Respondentes declarantes: {panel_residuos['respondeu_modulo_rs'].sum():,} ({panel_residuos['respondeu_modulo_rs'].mean()*100:.1f}%)")
    print(f"  Arquivos salvos:")
    print(f"    - {out_parquet}")
    print(f"    - {out_csv}")
    print(f"Tempo total de execução: {time.time() - t0:.1f} segundos.\n")

if __name__ == '__main__':
    main()
