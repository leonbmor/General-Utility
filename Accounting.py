#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# LJJ ACCOUNTING

import pandas as pd
import numpy as np
import math
import os
import csv
import yfinance as yf
import psycopg2
import requests
import json
import tkinter as tk
import re
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from datetime import date
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sqlalchemy import create_engine

def round_down(n, decimals = 0): 
    multiplier = 10 ** decimals 
    return int(math.floor(n * multiplier) / multiplier)

def Set_DF(dframe):
    
    dframe.index = dframe[dframe.columns.values.tolist()[0]]
    dframe.index.name = dframe.columns.values.tolist()[0]
    New_df = dframe.drop(dframe.columns.values.tolist()[0], axis=1)
    
    return New_df

def DD_Index(dframe):
    
    dframe['dummy'] = dframe.index
    dframe.drop_duplicates(['dummy'], inplace=True)
    dframe.drop('dummy', axis=1, inplace=True)
    New_df = dframe
    
    return New_df 

def open_df(*args):
    
    open_str = args[0]    
    query_open = 'SELECT * FROM ' + open_str
    opened_df = pd.read_sql_query(query_open, engine)
    opened_df = Set_DF(opened_df)
    opened_df = DD_Index(opened_df)
    opened_df = opened_df.sort_index()
    
    return opened_df

def openF_df(*args):
    
    open_str = args[0]    
    query_open = 'SELECT * FROM ' + open_str
    opened_df = pd.read_sql_query(query_open, Fengine)
    opened_df = Set_DF(opened_df)
    opened_df = DD_Index(opened_df)
    opened_df = opened_df.sort_index()
    
    return opened_df

def summarize(base_df, prod, dtField):
    
    summary_s = base_df[[dtField, 'Taxable']].groupby(base_df[dtField].map(lambda x: x.year))['Taxable'].sum()
    summary_s.name = prod
    try:
        summary_s = round(summary_s, 2)
    except:
        pass
    
    return summary_s

def format_col(base_df):
    
    for c_col in base_df.columns:
        base_df[c_col] = base_df[c_col].map(lambda x: '{:,.2f}'.format(x))
        
    return base_df

def taxableAVG(taxable_df, mult):
    
    m = 1
    if (mult):
        m = 100
    taxable_df[accMethod] = 0
    taxable_df[accMethod + ' Acc'] = 0
    taxable_df['Balance'] = 0
    taxable_df['Taxable'] = 0
    taxable_df['OCA'] = 'O'
    taxable_df['Trade Px'] = round(abs(taxable_df['Cash Amount'] / taxable_df['Quantity'] / m), 3)
    taxable_df = taxable_df.sort_values(['Trade Date', 'Equity', 'Quantity'], ascending = [True, True, False])
    taxable_df.index = np.arange(taxable_df.shape[0])    
    for c_idx in taxable_df.index:
        c_eq = taxable_df.loc[c_idx, 'Equity']
        taxable_df.loc[c_idx, accMethod] = abs(round(taxable_df.loc[c_idx, 'Cash Amount'] / taxable_df.loc[c_idx, 'Quantity'], 3))
        taxable_df.loc[c_idx, accMethod + ' Acc'] = abs(round(taxable_df.loc[c_idx, 'Cash Amount Acc'] / taxable_df.loc[c_idx, 'Quantity'], 3))
        taxable_df.loc[c_idx, 'Balance'] = taxable_df.loc[c_idx, 'Quantity']
        if (c_idx > 0):
            if (taxable_df[taxable_df.index < c_idx][taxable_df[taxable_df.index < c_idx]['Equity'] == c_eq]['Quantity'].sum() != 0):
                taxable_df.loc[c_idx, 'Balance'] = taxable_df.loc[c_idx, 'Quantity'] + taxable_df[taxable_df.index < c_idx][taxable_df[taxable_df.index < c_idx]['Equity'] == c_eq]['Balance'].iloc[-1]
                if (np.sign(taxable_df.loc[c_idx, 'Balance']) == np.sign(taxable_df.loc[c_idx, 'Quantity'])):
                    taxable_df.loc[c_idx, 'OCA'] = 'A'
                    taxable_df.loc[c_idx, accMethod] = (abs(taxable_df.loc[c_idx, 'Cash Amount']) + 
                                                        abs(taxable_df[taxable_df.index < c_idx][taxable_df[taxable_df.index < c_idx]['Equity'] == c_eq][accMethod].iloc[-1] * 
                                                            taxable_df[taxable_df.index < c_idx][taxable_df[taxable_df.index < c_idx]['Equity'] == c_eq]['Balance'].iloc[-1])) / abs(taxable_df.loc[c_idx, 'Balance'])
                    taxable_df.loc[c_idx, accMethod] = round(taxable_df.loc[c_idx, accMethod], 3)
                    taxable_df.loc[c_idx, accMethod + ' Acc'] = (abs(taxable_df.loc[c_idx, 'Cash Amount Acc']) + 
                                                        abs(taxable_df[taxable_df.index < c_idx][taxable_df[taxable_df.index < c_idx]['Equity'] == c_eq][accMethod + ' Acc'].iloc[-1] * 
                                                            taxable_df[taxable_df.index < c_idx][taxable_df[taxable_df.index < c_idx]['Equity'] == c_eq]['Balance'].iloc[-1])) / abs(taxable_df.loc[c_idx, 'Balance'])
                    taxable_df.loc[c_idx, accMethod + ' Acc'] = round(taxable_df.loc[c_idx, accMethod + ' Acc'], 3)                    
                else:
                    taxable_df.loc[c_idx, 'OCA'] = 'C'              
                    taxable_df.loc[c_idx, accMethod] = taxable_df[taxable_df.index < c_idx][taxable_df[taxable_df.index < c_idx]['Equity'] == c_eq][accMethod].iloc[-1]
                    taxable_df.loc[c_idx, accMethod + ' Acc'] = taxable_df[taxable_df.index < c_idx][taxable_df[taxable_df.index < c_idx]['Equity'] == c_eq][accMethod + ' Acc'].iloc[-1]


    close_l = taxable_df[taxable_df['OCA'] == 'C'].index.tolist()
    taxable_df.loc[close_l, 'Taxable'] = round(taxable_df.loc[close_l, 'Cash Amount'] + taxable_df.loc[close_l, 'Quantity'] * 
                                               taxable_df.loc[close_l, accMethod], 2)
    taxable_df[accMethod] = round(taxable_df[accMethod] / m, 3)
    taxable_df[accMethod + ' Acc'] = round(taxable_df[accMethod + ' Acc'] / m, 3)
    
    return taxable_df

def taxableFIFO(taxable_df):

    taxable_df[accMethod] = 0
    taxable_df[accMethod + ' Acc'] = 0
    taxable_df['Taxable'] = 0
    taxable_df['OCA'] = 'O'
    taxable_df['Trade Px'] = round(abs(taxable_df['Cash Amount'] / taxable_df['Quantity']), 3)
    taxable_df = taxable_df.sort_values(['Trade Date', 'Equity', 'Quantity'], ascending = [True, True, False])
    taxable_df.index = np.arange(taxable_df.shape[0])    
    for c_idx in taxable_df.index:
        c_eq = taxable_df.loc[c_idx, 'Equity']
        cBase_df = taxable_df[taxable_df.index < c_idx][taxable_df[taxable_df.index < c_idx]['Equity'] == c_eq]
        taxable_df.loc[c_idx, 'FIFOPx'] = abs(round(taxable_df.loc[c_idx, 'Cash Amount'] / taxable_df.loc[c_idx, 'Quantity'], 3))
        taxable_df.loc[c_idx, 'FIFOPx Acc'] = abs(round(taxable_df.loc[c_idx, 'Cash Amount Acc'] / taxable_df.loc[c_idx, 'Quantity'], 3))        
        if (cBase_df.shape[0] > 0):
            cBase_df = cBase_df.loc[cBase_df[cBase_df['OCA'] == 'O'].index[-1]: ]

            
            if (c_idx > 0):
                if (cBase_df['Quantity'].sum() != 0):
                    if (np.sign(cBase_df['Quantity'].sum()) == np.sign(taxable_df.loc[c_idx, 'Quantity'])):
                        taxable_df.loc[c_idx, 'OCA'] = 'A' 
                    else:
                        taxable_df.loc[c_idx, 'OCA'] = 'C'
                        if (np.sign(taxable_df.loc[cBase_df.index[0], 'Quantity']) == np.sign(taxable_df.loc[c_idx, 'Quantity'])):
                            cBase_df.loc[cBase_df.index[-1], 'OCA'] = 'O'

                        cOpen_df = cBase_df[cBase_df['OCA'] != 'C']
                        cClose_df = cBase_df[cBase_df['OCA'] == 'C']
                        cOpen_df['Cumulative'] = abs(cOpen_df['Quantity'].expanding().sum().astype(int))
                        c_start = 0
                        index_s = cOpen_df.index[0] 
                        if (cClose_df.shape[0] > 0):
                            c_start = abs(cClose_df['Quantity'].sum())
                            if (c_start > cOpen_df['Cumulative'].iloc[0]):
                                index_s = cOpen_df[cOpen_df['Cumulative'] > c_start].index[0]     
                        else:
                            c_start = 0                      
                        c_end = min(c_start + abs(taxable_df.loc[c_idx, 'Quantity']), abs(cOpen_df['Quantity'].sum()))
                        if (c_start + abs(taxable_df.loc[c_idx, 'Quantity']) > cOpen_df['Cumulative'].iloc[0]):
                            index_e = cOpen_df[cOpen_df['Cumulative'] >= c_end].index[0]
                        else:
                            index_e = cOpen_df.index[0]   

                        if (index_s == index_e):
                            taxable_df.loc[c_idx, accMethod] = cOpen_df.loc[index_s, 'Trade Px']
                            taxable_df.loc[c_idx, accMethod + ' Acc'] = abs(cOpen_df.loc[index_s, 'Cash Amount Acc'] / cOpen_df.loc[index_s, 'Quantity'])
                        else:  
                            ttlCash = (min(cOpen_df.loc[index_s, 'Cumulative'], c_end) - c_start) * cOpen_df.loc[index_s, 'Trade Px']
                            ttlCashAcc = (min(cOpen_df.loc[index_s, 'Cumulative'], c_end) - c_start) * (cOpen_df.loc[index_s, 'Cash Amount Acc'] / cOpen_df.loc[index_s, 'Quantity'])
                            while (index_s != index_e):   
                                c_start = cOpen_df.loc[index_s, 'Cumulative']                            
                                index_s = cOpen_df.index[cOpen_df.index.tolist().index(index_s) + 1]                              
                                cPx = cOpen_df.loc[index_s, 'Trade Px']
                                cPxAcc = cOpen_df.loc[index_s, 'Cash Amount Acc'] / cOpen_df.loc[index_s, 'Quantity']
                                ttlCash += (min(cOpen_df.loc[index_s, 'Cumulative'], c_end) - c_start) * cPx
                                ttlCashAcc += (min(cOpen_df.loc[index_s, 'Cumulative'], c_end) - c_start) * cPxAcc                                                      
                            taxable_df.loc[c_idx, accMethod] = abs(ttlCash / taxable_df.loc[c_idx, 'Quantity'])
                            taxable_df.loc[c_idx, accMethod + ' Acc'] = abs(ttlCashAcc / taxable_df.loc[c_idx, 'Quantity'])


    close_l = taxable_df[taxable_df['OCA'] == 'C'].index.tolist()
    taxable_df.loc[close_l, 'Taxable'] = round(taxable_df.loc[close_l, 'Cash Amount'] + taxable_df.loc[close_l, 'Quantity'] * 
                                               taxable_df.loc[close_l, accMethod], 2)
    taxable_df[accMethod] = round(taxable_df[accMethod], 3)
    taxable_df[accMethod + ' Acc'] = round(taxable_df[accMethod + ' Acc'], 3)
    
    return taxable_df

def FXconvUSD(base_l, y_list):
    
    d_list = pd.Series(y_list).map(lambda x: sEURFX_df[sEURFX_df.index < datetime(x + 1, 1, 1).date()].index[-1])
    base_l = base_l + [base_l[-1]] * (len(y_list) - len(base_l))
    base_l = list(np.array(base_l) * pd.Series(d_list).map(lambda x: sEURFX_df.loc[x, 'USDUSD']))
    
    return base_l

def FXconvEUR(base_l, y_list):
    
    d_list = pd.Series(y_list).map(lambda x: sEURFX_df[sEURFX_df.index < datetime(x + 1, 1, 1).date()].index[-1])
    base_l = base_l + [base_l[-1]] * (len(y_list) - len(base_l))
    base_l = list(np.array(base_l) * pd.Series(d_list).map(lambda x: sEURFX_df.loc[x, 'EURUSD']))
    
    return base_l

    
def get_excPxs(dropLAST, cutoff_dt):
    
    def on_selection(value):
        global cy
        cy = value
        root.destroy()   
    def on_selectionT(value):
        global ct
        ct = value
        root.destroy()           
    
    excludedPX_df = openF_df('acct_excluded_pxs')
    if (dropLAST == False) & (cutoff_dt.year not in excludedPX_df.index):
        excludedPX_df.loc[cutoff_dt.year] = np.nan
    modY_l = excludedPX_df.index.tolist()
    modT_l = excludedPX_df.columns.tolist()

    print(excludedPX_df.columns.tolist())
    print('')
    modPx_loop = True  
    save_pxR = False
    print(excludedPX_df.iloc[-1]) 
    print('')
    while (modPx_loop):  
        modPx_T = input('Modify prices ((P)revious year, Ticker name or empty)? ').upper()
        print('')
        if (modPx_T == 'P'):                                    
            save_pxR = True
            root = tk.Tk()
            tkvar = tk.StringVar(root)
            popupMenu = tk.OptionMenu(root, tkvar, *modT_l, command = on_selectionT)
            tk.Label(root, text = "Choose Ticker:").grid(row = 0, column = 0)
            popupMenu.grid(row = 1, column = 0)
            root.mainloop()
            choice_ticker = ct
            root = tk.Tk()
            tkvar = tk.StringVar(root)
            popupMenu = tk.OptionMenu(root, tkvar, *modY_l, command = on_selection)
            tk.Label(root, text = "Choose year:").grid(row = 0, column = 0)
            popupMenu.grid(row = 1, column = 0)
            root.mainloop()       
            choice_year = cy
            valid_px = False
            while (not valid_px):
                try:
                    excludedPX_df.loc[choice_year, choice_ticker] = float(input('Enter px for ' + modPx_T + ' for ' + str(choice_year) + ': '))
                    valid_px = True
                except:
                    print('again...')            
        elif (modPx_T != ''):
            valid_t = False
            choice_year = modY_l[-1]
            if (modPx_T not in excludedPX_df.columns):
                newT_str = input(modPx_T + ' not on DB, include? (Y or empty)').upper()
                if (newT_str == 'Y'):
                    excludedPX_df[modPx_T] = np.nan
                    valid_t = True
            else:
                valid_t = True
            if (valid_t):
                save_pxR = True                                
                valid_px = False
                while (not valid_px):
                    try:
                        excludedPX_df.loc[choice_year, modPx_T] = float(input('Enter px for ' + modPx_T + ' for ' + str(choice_year) + ': '))
                        valid_px = True
                    except:
                        print('again...')
            else:
                print('again...')
        else:
            modPx_loop = False
    if (save_pxR):
        print(excludedPX_df)
        print('')        
        excludedPX_df.to_sql('acct_excluded_pxs', Fengine, index = True, if_exists = 'replace')  
        
    return excludedPX_df    

def getSplits():

    Tsplit_l = ['AVGO', 'NVDA', 'GOOG', 'PANW', 'SHOP', 'SMCI', 'TSLA']
    Rsplit_l = [10, 10, 20, 2, 10, 10, 3]
    Dsplit_l = [datetime(2024, 7, 15).date(), datetime(2024, 6, 10).date(), datetime(2022, 7, 18).date(), 
                datetime(2024, 12, 16).date(), datetime(2022, 6, 29).date(), datetime(2024, 10, 1).date(), datetime(2022, 8, 25).date()]
    Data = {'Ratios': Rsplit_l, 'Dates': Dsplit_l}
    split_df = pd.DataFrame(Data, index = Tsplit_l)

    Tsplit2_l = ['NVDA', 'PANW', 'TSLA']
    Rsplit2_l = [4, 3, 5]
    Dsplit2_l = [datetime(2021, 7, 20).date(), datetime(2022, 9, 14).date(), datetime(2020, 8, 31).date()]
    Data2 = {'Ratios': Rsplit2_l, 'Dates': Dsplit2_l}
    split2_df = pd.DataFrame(Data2, index = Tsplit2_l)
    
    return split_df, split2_df

def complementBonds(bond_df):
    
    bond_df.loc[0] = [datetime(2022, 12, 22).date(), 'TB02NOV23', 'USD', -50000 * .9551]
    bond_df.loc[1] = [datetime(2022, 12, 22).date(), 'T3250824', 'USD', -66000 * .9849]
    bond_df.loc[2] = [datetime(2022, 12, 22).date(), 'GS21250924', 'EUR', -99000 * .9898]
    bond_df = bond_df.sort_values(by = ['Trade Date', 'Equity'], ascending = [True, False])
    bond_df['Quantity'] = [50, 66, 99, 145, -145, -15, -35, -66, -99, 408, -16]
    
    return bond_df

def complementFwds(fwd_df):
    
    fwd_df.loc[0] = [datetime(2021, 2, 17).date(), 'SHAREHOLDERCONT - XAU - 1485.48', 'XAU', 148.1]
    fwd_df.loc[1] = [datetime(2021, 2, 17).date(), 'SHAREHOLDERCONT - XAU - 1485.48', 'EUR', -220000]
    fwd_df.loc[2] = [datetime(2021, 2, 17).date(), 'SHAREHOLDERCONT - XPD - 1980', 'XAU', 50]
    fwd_df.loc[3] = [datetime(2021, 2, 17).date(), 'SHAREHOLDERCONT - XPD - 1980', 'EUR', -99000]
    fwd_df = fwd_df.sort_index()
    
    return fwd_df

def SendToPrint(base_df):
       
    for c_col in base_df.columns:
        try:
            base_df[c_col] = base_df[c_col].map(lambda x: '{:,.2f}'.format(x)).values   
        except:
            pass
    print(base_df)
    
    return

def getList(rel_name, year_list):
    
    def on_selection(value):
        global cy
        cy = value
        root.destroy()    
    
    try:
        curr_df = openF_df(rel_name)
    except:
        conf = False
        while (not conf):
            ccy = input('Enter currency: ').upper()
            confirm = input('Confirm ' + ccy + ' (empty or N)? ').upper()
            if (confirm == ''):
                curr_df = pd.DataFrame(0, index = pd.Index(year_list, name = 'index'), columns = [ccy])
                conf = True
    print('Current ' + rel_name)
    SendToPrint(curr_df.copy())
    print('')
    new_version = False
    missY_l = [y for y in year_list if y not in curr_df.index]
    for c_my in missY_l:
        valid = False
        while (not valid):
            newL = input('New line for ' + str(c_my) + ' (empty to repeat last): ')
            if (newL == ''):
                newL = curr_df.iloc[-1].iloc[0]
            try:
                curr_df.loc[c_my] = float(newL)
                new_version = True
                valid = True
            except:
                print('again...')

    modPrev_loop = True  
    while (modPrev_loop): 
        modPrev = input('Modify previous ' + str(curr_df.columns[0]) + ' input - (P)revious year, (A)dd, number or empty ').upper()
        try:
            newL = float(modPrev)
            modPrev = 'C'
        except:
            pass
        print('')    
        choice_year = year_list[-1]
        if (modPrev == 'A'):                        
            valid = False
            while (not valid):
                try:
                    newL = input('New ' + str(curr_df.columns[0]) + ' input for ' + str(choice_year) + ' (empty to cancel): ')
                    if (newL == ''):
                        newL = 0
                        valid = True
                    else:
                        newL = float(newL)
                        valid = True
                except:
                    print('again...') 
            curr_df.loc[choice_year] += newL
            new_version = True
        elif (modPrev == 'C'):
            curr_df.loc[choice_year] = newL
            new_version = True                                                    
        elif (modPrev == 'P'):
            modY_l = curr_df.index.tolist()
            root = tk.Tk()
            tkvar = tk.StringVar(root)
            popupMenu = tk.OptionMenu(root, tkvar, *modY_l, command = on_selection)
            tk.Label(root, text = "Choose year:").grid(row = 0, column = 0)
            popupMenu.grid(row = 1, column = 0)
            root.mainloop()       
            choice_year = cy
            valid = False
            while (not valid):
                try:
                    newL = input('New ' + str(curr_df.columns[0]) + ' total for ' + str(choice_year) + ' (empty to cancel): ')
                    if (newL == ''):
                        newL = 0
                        valid = True
                    else:
                        newL = float(newL)
                        valid = True
                except:
                    print('again...')

            if (newL == 0):
                modPrev_loop = False
            else:
                curr_df.loc[choice_year] = newL
                new_version = True
        else:
            modPrev_loop = False
        if (new_version):
            SendToPrint(curr_df.copy())
            print('')                        
    curr_df = curr_df.sort_index()
    c_list = curr_df[curr_df.columns[0]].values.tolist()
    if (new_version): 
        curr_df.to_sql(rel_name, Fengine, index = True, if_exists = 'replace')
        
    return c_list

def getLJJ(year_list):   
    
    LJJAUM_df = openF_df('ljj_aum')
    new_version = False
    
    missY_l = [y for y in year_list if y not in LJJAUM_df.index]    
    for ny in missY_l:
        LJJAUM_df.loc[ny] = LJJAUM_df.iloc[-1]
        new_version = True
    LJJAUM_df = LJJAUM_df.sort_index()
        
    SendToPrint(LJJAUM_df.copy())
    valid = False
    while (not valid):
        newAUM = input('New number for last year (empty to skip): ')
        if (newAUM != ''):
            try:    
                conf = input('Confirm ' + '{:,.0f}'.format(float(newAUM)) + '? (N or empty)').upper()
                if (conf == ''):
                    print('')
                    valid = True
                    LJJAUM_df.loc[LJJAUM_df.index[-1], 'Value'] = float(newAUM)
                    new_version = True
                else:
                    print('again...')                    
            except:
                print('again...')
        else:
            print('')
            valid = True
        if (new_version):
            SendToPrint(LJJAUM_df.copy())
            print('')                        
    if (new_version):
        LJJAUM_df.to_sql('ljj_aum', Fengine, index = True, if_exists = 'replace')     
        
    return LJJAUM_df

def getExtras():   
    
    extras_df = Set_DF(pd.read_sql_query('SELECT * FROM ljj_extras', Fengine))
    new_version = False        
    SendToPrint(extras_df.copy())
        
    more_inputs = True
    while (more_inputs):
        newEXTRA = input('New extra k move for LJJ (Y or empty)? ').upper()
        if (newEXTRA == 'Y'):          
            valid = False
            new_version = True        
            while (not valid):
                try:
                    dt = datetime.today().date()
                    dt_str = input('Enter date (empty for today or YY/MM/DD): ')
                    if (dt_str != ''):
                        dt = datetime.strptime(dt_str, '%y/%m/%d').date()
                    valid_ccy = False
                    while not (valid_ccy):
                        ccy = input('Enter CCY (empty for EUR): ')
                        if (ccy == ''):
                            ccy = 'EUR'
                            valid_ccy = True
                        elif (ccy in ccy_l):
                            valid_ccy = True
                    value = float(input('Enter new extra value: '))
                    valid = True
                except:
                    print('again...')
            newDate_l = extras_df.index.tolist() + [dt]
            newCCY_l = extras_df['Currency'].tolist() + [ccy]
            newVal_l = extras_df['Cash Amount'].tolist() + [value] 
            extras_df = pd.DataFrame({'Currency': newCCY_l, 'Cash Amount': newVal_l}, index = newDate_l)            
        else:            
            more_inputs = False
        if (new_version):
            SendToPrint(extras_df.copy())
            print('')                     
    extras_df = extras_df.sort_index()
    if (1):
        extras_df.to_sql('ljj_extras', Fengine, index = True, if_exists = 'replace')     
        
    return extras_df

def fixExc(excludedPX_df, yf_l):

    init_dt = datetime(excludedPX_df.index[0], 7, 1).date()
    n_days = Pxs_df[Pxs_df.index >= init_dt].shape[0]
    addYF_df = yf.download(yf_l, period = str(n_days) + 'd', auto_adjust = False, multi_level_index = False)
    addYF_df = addYF_df.resample('Y').last()['Adj Close']
    for add_t in yf_l:
        excludedPX_df[add_t] = round(addYF_df[add_t], 3).values
        
    print(excludedPX_df)
    print('')        
    excludedPX_df.to_sql('acct_excluded_pxs', Fengine, index = True, if_exists = 'replace')        

    return excludedPX_df

def prepareUBSfile(base_df):
    
    UBSActionsKW_l = ['BUY', 'SELL', 'PURCHASE', 'SALE', 'PREMIUM', 'PAYMENT', 'DEPOSIT', 
                      'WITHDRAWL', 'FEE', 'DIVIDEND', 'EXERCISING', 'ASSIGNMENT']

    base_df = base_df[base_df[base_df[base_df.columns[0]] == 'Trade date'].index[0]:]
    base_df.columns = base_df.iloc[0]
    base_df = base_df.drop(base_df.index[0])
    base_df[['Debit', 'Credit']] = base_df[['Debit', 'Credit']].fillna(0)
    base_df = base_df.dropna(how = 'all', axis = 1)

    OptBase_df = base_df[(base_df['Description1'].map(lambda x: x.find('Call')) != -1) | (base_df['Description1'].map(lambda x: x.find('Put')) != -1)]
    OptBase_df['Description1'] = OptBase_df['Description1'].map(lambda x: re.sub('Reg.shs', '', x))
    base_df.loc[OptBase_df.index, 'Description1'] = OptBase_df['Description1'].values
    base_df.loc[base_df[base_df['Description1'].map(lambda x: x.upper().find('SERVICE')) != -1].index, ['Description1', 'Description2']] = 'FEE'
    base_df.loc[base_df[base_df['Description2'].map(lambda x: x.upper().find('EXPENSES')) != -1].index, ['Description1', 'Description2']] = 'FEE'
    base_df.loc[base_df[base_df['Description3'] == 'credit'].index, ['Description1', 'Description2']] = 'DEPOSIT'
    base_df.loc[base_df[base_df['Description3'] == 'credit'].index, base_df.columns[base_df.columns.tolist().index('Description3'):]] = np.nan
    if ('Footnotes' in base_df.columns):
        base_df.loc[base_df[base_df['Footnotes'] == 'credit'].index, ['Description1', 'Description2']] = 'DEPOSIT'
        base_df.loc[base_df[base_df['Footnotes'] == 'credit'].index, base_df.columns[base_df.columns.tolist().index('Footnotes'):]] = np.nan    
    zeroB_l = base_df[base_df[['Debit', 'Credit']].astype(float).sum(axis = 1) == 0].index.tolist()
    if (len(zeroB_l) > 0):
        base_df = base_df.drop(zeroB_l)
    prob_l = base_df[base_df['Description2'].map(lambda x: len([k for k in x.split(' ') if k.upper() in UBSActionsKW_l]) == 0)].index.tolist()
    for c_prob in prob_l:
        amd_str1 = base_df.iloc[base_df.index.tolist().index(c_prob), base_df.columns.tolist().index('Description1')]
        amd_str2 = base_df.iloc[base_df.index.tolist().index(c_prob), base_df.columns.tolist().index('Description2')]
        if (amd_str1[-1] != ';'):
            amd_str1 += ';'
        amd_str3 = ''
        found = False
        for i, c_col in enumerate(base_df.columns[base_df.columns.tolist().index('Description2'): ]):
            if (found):
                if (amd_str3 == ''):
                    amd_str3 += base_df.iloc[base_df.index.tolist().index(c_prob), base_df.columns.tolist().index('Description2') + i]
                else:
                    if (pd.notna(base_df.iloc[base_df.index.tolist().index(c_prob), base_df.columns.tolist().index('Description2') + i])):
                        amd_str3 += ';' + base_df.iloc[base_df.index.tolist().index(c_prob), base_df.columns.tolist().index('Description2') + i]
            else:
                if (type(base_df.iloc[base_df.index.tolist().index(c_prob), base_df.columns.tolist().index('Description2') + i]) == str):
                    if (len([t for t in base_df.iloc[base_df.index.tolist().index(c_prob), base_df.columns.tolist().index('Description2') + i].split(' ') if t.upper() in UBSActionsKW_l]) == 0):
                        amd_str1 += base_df.iloc[base_df.index.tolist().index(c_prob), base_df.columns.tolist().index('Description2') + i]
                    else:
                        found = True
                        amd_str2 = base_df.iloc[base_df.index.tolist().index(c_prob), base_df.columns.tolist().index('Description2') + i]
        base_df.loc[c_prob, 'Description1'] = amd_str1
        base_df.loc[c_prob, 'Description2'] = amd_str2
        base_df.loc[c_prob, 'Description3'] = amd_str3  
        base_df.iloc[base_df.index.tolist().index(c_prob), (base_df.columns.tolist().index('Description3') + 1): ] = np.nan
    base_df = base_df[base_df.columns[: base_df.columns.tolist().index('Description3') + 1]] 
    
    return base_df 

def formatUBS(UBS_df, allCS_df):    
    
    opt_dict1 = {'Call': 'Put', 'Put': 'Call'}
    opt_dict2 = {'Short': 'SALE', 'Long': 'PURCHASE'}
    fwd_dict1 = {'sold': 'SALE', 'bought': 'PURCHASE'}   
    eqtOpt_l = ['Buy', 'Sell', 'open', 'close'] 

    allUBS_df = pd.DataFrame(np.nan, index = np.arange(UBS_df.shape[0]), columns = allCS_df.columns)
    UBS_df.index = np.arange(UBS_df.shape[0])
    allUBS_df['Trade Date'] = UBS_df['Trade date'].map(lambda x: datetime.strptime(x, '%m/%d/%Y').date()).values
    allUBS_df['Currency'] = UBS_df['Currency']
    allUBS_df['Cash Amount'] = UBS_df[['Credit', 'Debit']].astype(float).sum(axis = 1).values
    allUBS_df.loc[UBS_df[UBS_df['Description1'].map(lambda x: x.lower().find('shs')) != -1].index, 'Narrative'] = UBS_df.loc[UBS_df[UBS_df['Description1'].map(lambda x: x.lower().find('shs')) != -1].index, 'Description1'].map(lambda x: ' - Settlement of ' + x.split(' ')[-2][1:-2] + ' ' + x.split(' ')[-1] + ' - ').tolist()    

    ndf_l = UBS_df[UBS_df['Description1'].map(lambda x: x.upper().find('FOREX PURCHASE')) != -1].index.tolist()
    for c_ndf in ndf_l:
        if (allUBS_df.loc[c_ndf, 'Cash Amount'] > 0):
            allUBS_df.loc[c_ndf, 'Narrative'] = ' NDF USD / BRL - SETTLEMENT'
            UBS_df.loc[c_ndf, 'Description1'] = ' NDF USD / BRL - SETTLEMENT'
        elif (allUBS_df.loc[c_ndf, 'Cash Amount'] < 0):
            allUBS_df.loc[c_ndf, 'Narrative'] = ' NDF USD / BRL - SETTLEMENT'
            UBS_df.loc[c_ndf, 'Description1'] = ' NDF USD / BRL - SETTLEMENT'
        else:
            allUBS_df = allUBS_df.drop(c_ndf)    
    
    tsy_l = UBS_df[UBS_df['Description1'].map(lambda x: x.upper().find('TREASURY')) != -1].index.tolist()
    for c_tsy in tsy_l:
        if (allUBS_df.loc[c_tsy, 'Cash Amount'] > 0):
            allUBS_df.loc[c_tsy, 'Narrative'] = 'SALE OF ' + UBS_df.loc[c_tsy, 'Description1'].split('; ')[-1] + ' - PRINCIPAL'
        elif (allUBS_df.loc[c_tsy, 'Cash Amount'] < 0):
            allUBS_df.loc[c_tsy, 'Narrative'] = 'PURCHASE OF ' + UBS_df.loc[c_tsy, 'Description1'].split('; ')[-1] + ' - PRINCIPAL'
        else:
            allUBS_df = allUBS_df.drop(c_tsy)

    allUBS_df.loc[allUBS_df[allUBS_df['Narrative'].notna()][allUBS_df[allUBS_df['Narrative'].notna()]['Cash Amount'] > 0].index, 'Narrative'] = allUBS_df.loc[allUBS_df[allUBS_df['Narrative'].notna()][allUBS_df[allUBS_df['Narrative'].notna()]['Cash Amount'] > 0].index, 'Narrative'].map(lambda x: 'CREDIT' + x)
    allUBS_df.loc[allUBS_df[allUBS_df['Narrative'].notna()][allUBS_df[allUBS_df['Narrative'].notna()]['Cash Amount'] < 0].index, 'Narrative'] = allUBS_df.loc[allUBS_df[allUBS_df['Narrative'].notna()][allUBS_df[allUBS_df['Narrative'].notna()]['Cash Amount'] < 0].index, 'Narrative'].map(lambda x: 'DEBIT' + x)
    allUBS_df.loc[allUBS_df[allUBS_df['Narrative'].notna()].index, 'Narrative'] = allUBS_df.loc[allUBS_df[allUBS_df['Narrative'].notna()].index, 'Narrative'].index.map(lambda x: allUBS_df.loc[x, 'Narrative'] + UBS_df.loc[allUBS_df[allUBS_df['Narrative'].notna()].index]['Description3'].replace({'': '/Amt.'}).map(lambda x: x.split('/Amt.')[1].split(';')[0]).loc[x])
    allUBS_df.loc[UBS_df[UBS_df['Description2'].map(lambda x: x.find('FX Vanilla')) != -1].index, 'Narrative'] = UBS_df[UBS_df['Description2'].map(lambda x: x.find('FX Vanilla')) != -1]['Description1'].map(lambda x: opt_dict2[x.split(' ')[0]] + ', ' + x.split('/')[0][-3:] + ' ' + x.split(' ')[1].upper() + ' / ' + x.split('/')[1][:3] + ' ' + opt_dict1[x.split(' ')[1]].upper() + ',' + x.split(';')[-1]).values
    allUBS_df.loc[UBS_df[UBS_df['Description2'].map(lambda x: x.find('FX Spot')) != -1].index, 'Narrative'] = UBS_df[UBS_df['Description2'].map(lambda x: x.find('FX Spot')) != -1]['Description1'].map(lambda x: fwd_dict1[x.split(' ')[1]] + ' ' + x.split(' ')[2][:-1] + ' / ' + fwd_dict1[x.split(' ')[4]] + ' ' + x.split(' ')[5][:-1] + ' - maturity forward ' + x.split('-')[-1])

    Cancel_l = UBS_df['Description1'][UBS_df['Description1'].map(lambda x: x.upper().find('CANCELLATION')) != -1].index.tolist()
    for c_cancel in Cancel_l:
        cDrop_l = [c_cancel]
        primary_l = allUBS_df[allUBS_df['Cash Amount'] == -allUBS_df.loc[c_cancel, 'Cash Amount']].index.tolist()
        allUBS_df.loc[primary_l, 'Narrative'] = UBS_df.loc[primary_l, 'Description2'].map(lambda x: x.split('/')[0]).values
        cDrop_l.append(allUBS_df.loc[primary_l, 'Narrative'][allUBS_df.loc[primary_l, 'Narrative'] == UBS_df.loc[c_cancel, 'Description2'].split('/')[0]].index[0])
        allUBS_df = allUBS_df.drop(cDrop_l)
        UBS_df = UBS_df.drop(cDrop_l)

    forex_df = allUBS_df[UBS_df['Description1'].map(lambda x: x.find('forex')) != -1]
    for c_fx in forex_df.index:
        if (forex_df.loc[c_fx, 'Cash Amount'] > 0):
            allUBS_df.loc[c_fx, 'Narrative'] = 'PURCHASE ' + UBS_df.loc[c_fx, 'Currency'] + ' - spot - ' + UBS_df.loc[c_fx, 'Description2'].split('/')[0]
        elif (forex_df.loc[c_fx, 'Cash Amount'] < 0):        
            allUBS_df.loc[c_fx, 'Narrative'] = 'SALE ' + UBS_df.loc[c_fx, 'Currency'] + ' - spot - ' + UBS_df.loc[c_fx, 'Description2'].split('/')[0]
        else:
            allUBS_df = allUBS_df.drop(c_fx)

    missing_l = UBS_df[allUBS_df['Narrative'].isna()].index.tolist()
    mlDesc1_s = UBS_df.loc[missing_l, 'Description1']
    mlDesc2_s = UBS_df.loc[missing_l, 'Description2']
    mlDesc3_s = UBS_df.loc[missing_l, 'Description3']        
        
    Reversal_l = mlDesc2_s[mlDesc2_s.map(lambda x: x.upper().find('REVERSAL')) != -1].index.tolist()
    na_l = allUBS_df[allUBS_df['Narrative'].isna()].index
    allUBS_df.loc[na_l, 'Narrative'] = UBS_df.loc[na_l, 'Value date']
    for c_rever in Reversal_l:
        cDrop_l = [c_rever]
        vd = allUBS_df.loc[c_rever, 'Narrative']
        amt = allUBS_df.loc[c_rever, 'Cash Amount']
        exc_df = allUBS_df[allUBS_df['Narrative'] == vd]
        cDrop_l.append(exc_df[exc_df['Cash Amount'] == -exc_df.loc[c_rever, 'Cash Amount']].index[0])
        allUBS_df = allUBS_df.drop(cDrop_l)
        UBS_df = UBS_df.drop(cDrop_l)
        mlDesc1_s = mlDesc1_s.drop(cDrop_l)
        mlDesc2_s = mlDesc2_s.drop(cDrop_l)
        mlDesc3_s = mlDesc3_s.drop(cDrop_l)
    allUBS_df.loc[[i for i in na_l if i in allUBS_df.index], 'Narrative'] = np.nan        

    allUBS_df.loc[UBS_df[UBS_df['Description1'].map(lambda x: x.find('WITHDRAWAL')) != -1].index, 'Narrative'] = 'WITHDRAWAL'
    allUBS_df.loc[UBS_df[UBS_df['Description1'].map(lambda x: x.find('DEPOSIT')) != -1].index, 'Narrative'] = 'DEPOSIT'    

    EqtOpt_l = mlDesc2_s[mlDesc2_s.map(lambda x: len([t for t in x.split(' ') if t in eqtOpt_l]) == 2)].index.tolist()
    EqtOptD1_s = mlDesc1_s.loc[EqtOpt_l].map(lambda x: x.split(' ')[0].upper() + x.split(';')[-1] + ' ' + [t for t in x.split(')')[0][::-1].split('(')[0][::-1].split(' ') if t in allT_l][0] + ', ')
    EqtOptD2_s = mlDesc3_s.loc[EqtOpt_l].map(lambda x: x.split(';')[0].split(' ')[-1])
    allUBS_df.loc[EqtOpt_l, 'Narrative'] = EqtOptD1_s.index.map(lambda x: EqtOptD1_s.loc[x] + EqtOptD2_s.loc[x]).values

    Fwd_df = UBS_df[UBS_df['Description2'].map(lambda x: x.upper().find('FX FORWARD')) != -1]
    Fwd2_df = UBS_df[UBS_df['Description2'].map(lambda x: x.upper().find('FX SWAP')) != -1]
    Fwd_df = pd.concat([Fwd_df, Fwd2_df])    
    FwdDesc_s = Fwd_df['Description1'].map(lambda x: x.split(';')[-1].split(' ')[1] + ' ' + x.split(';')[0].split(' ')[-1] + ' / ' + x.split(';')[1].split(' ')[-1] + ' - SETTLEMENT')
    for c_fwd in Fwd_df.index:
        if (allUBS_df.loc[c_fwd, 'Cash Amount'] > 0):
            allUBS_df.loc[c_fwd, 'Narrative'] = 'CREDIT ' + FwdDesc_s.loc[c_fwd]
        elif (allUBS_df.loc[c_fwd, 'Cash Amount'] < 0):
            allUBS_df.loc[c_fwd, 'Narrative'] = 'DEBIT ' + FwdDesc_s.loc[c_fwd]
        else:
            allUBS_df = allUBS_df.drop(c_fwd)
            
    Swap_df = UBS_df[UBS_df['Description2'].map(lambda x: x.upper().find('REFIX')) != -1]
    allUBS_df.loc[Swap_df.index, 'Narrative'] = 'SWAP EURIBOR'
    allUBS_df.loc[UBS_df[UBS_df['Description1'] == 'FEE'].index, 'Narrative'] = 'FEE'

    allUBS_df.loc[allUBS_df[UBS_df['Description2'] == 'Dividend'].index, 'Narrative'] = 'CASH DIVIDEND'
    allUBS_df.loc[allUBS_df[allUBS_df['Narrative'] == 'WITHDRAWAL'][allUBS_df[allUBS_df['Narrative'] == 'WITHDRAWAL']['Cash Amount'] > 0].index, 'Narrative'] = 'ADJ'
    allUBS_df = allUBS_df.sort_values(by = ['Trade Date']) 
    allUBS_df.index = np.arange(allUBS_df.shape[0]) 
    

    return allUBS_df

def on_selection_asset(value):
    global target_asset
    target_asset = value
    root.destroy()  

import warnings

warnings.filterwarnings('ignore')
dbase = "visiblealpha_laptop"
Fdbase = "factormodel_db"
cnxn_string = ("postgresql+psycopg2://{username}:{pswd}""@{host}:{port}/{database}")
engine = create_engine(cnxn_string.format(username = "postgres", pswd = "akf7a7j5", host = "localhost", 
                                          port = 5432, database = dbase))
Fengine = create_engine(cnxn_string.format(username = "postgres", pswd = "akf7a7j5", host = "localhost", 
                                           port = 5432, database = Fdbase))

ticker_dikt = {'MELI0000': 'MELI', 'GOOG/US': 'GOOG', 'EBR/B': 'ELET6', 'GOL0000': 'GOLL4', 'IWM0000': 'IWM', 
               'CSUHD': 'CELH', 'BIDU0000': 'BIDU', 'IBE/D': 'IBE', 'RDDTUS': 'RDDT'}
jokers_dikt = {'FB': 'META', 'SQ': 'XYZ', 'ZI': 'GTM'}
exception_dikt = {'CREDIT - Settlement of IBE/D 0000 S ES06445809L2 - 2,227.000000': 'EXCEPTION IBE', 
                  'CREDIT - Settlement of EDF FP FR0010242511 - 4,058.000000': 'CREDIT - Settlement of EDF FP FR0010242511 - 4,000.000000',
                  'DEBIT - Settlement of IBE SM ES0144580Y14 - 50.000000': 'EXCEPTION II IBE', 
                  'CREDIT - Settlement of IBE SM ES0144580Y14 - 2,277.000000': 'CREDIT - Settlement of IBE SM ES0144580Y14 - 2,227.000000'}

warnings.filterwarnings('ignore')
Pxs_df = openF_df('prices_relation').sort_index()
FX_df = openF_df('fxprices_relation').sort_index()
#FX_df.loc[FX_df.index[-1], 'EURUSD'] = 1 #############################################

Pxs_df['WISE LN'] *= Pxs_df.index.map(lambda x: FX_df.loc[x, 'GBPUSD']).values / 100
Pxs_df['SIVB US'] = 0
for c_joker in jokers_dikt:
    if ((jokers_dikt[c_joker] + ' US') not in Pxs_df.columns):
        Pxs_df[jokers_dikt[c_joker] + ' US'] = Pxs_df[c_joker + ' US']

today = datetime.today().date()
today = today + timedelta((2 - today.weekday() % 5) * (today.weekday() // 5))
tomorrow = today + timedelta(1)
tomorrow = tomorrow + timedelta((2 - tomorrow.weekday() % 5) * (tomorrow.weekday() // 5))

ccy_l = ['EUR', 'GBP', 'DKK', 'SEK', 'INR', 'CHF', 'CAD', 'MXN', 'BRL', 'JPY', 'AUD', 'NZD', 'XAU', 
         'XPD', 'XAG', 'XPT']
swap_l = ['SWAP', 'EURIBOR']
fee_l = ['FEE']
dvd_l = ['DIVIDEND', 'DIVIDENDS', 'TAX', 'ADJ']
PM_l = ['XAUUSD', 'XAGUSD', 'XPDUSD', 'XPTUSD']
saxoOptKV_l = ['Options Price Reporting Authority', 'Inter Bank']
saxoEqtyKV_l = ['NASDAQ', 'NASDAQ (Small cap)', 'New York Stock Exchange', 'NYSE American', 
                'OTC Markets Group (Pink Sheets) – No Information', 'Euronext Amsterdam', 
                'New York Stock Exchange (ARCA)']
saxoDepositKV_l = ['Unknown']

ticker_dict = {'RTP US': 'JOBY US'}
dt_CDI = datetime(2024, 12, 31).date()
LJJAUM_init = 3444532.17
alien_lines = 15
swapCCY = 'EUR'
cutoff_dt = FX_df.index[-1]
adjCDI = False
bundleALL = False
dropLAST = False
realizeFX = False
ExpAsUgo = True

accountingEUR = True ##############
optSTR = False ##############
fifo = True ##############

acc_ccy = 'EUR'
if (realizeFX):
    print('-----------------------FX VARIATION VS. EUR: REALIZED-----------------------')
else:
    print('----------------------FX VARIATION VS. EUR: UNREALIZED----------------------')
print('')

dl = input('Drop current year (Y/y or empty): ').upper()
if (dl == 'Y'):
    dropLAST = True
    
accMethod = 'AvgOpenPx' 
if (fifo):
    accMethod = 'FIFOPx'
    
all_tickers = pd.read_csv(r"C:\Users\Utilizador\OneDrive\Documentos\Malta\LJJ\tickers_all.csv")
allT_l = pd.unique(all_tickers['tickers'].map(lambda x: x.split(' ')[0]).replace({'IWMUSD': 'IWM', 'BRZUUSD': 'BRZU'})).tolist()
all_tickers.index = all_tickers['tickers'].map(lambda x: x.split(' ')[0])
all_tickers['tickers'] = all_tickers['tickers'].map(lambda x: x[:-3]).replace({' GY': ' GR'})
allT_l.append('VALE')
allT_l.append('MELI0000')
allT_l.append('TSM')
allT_l.append('GOOG/US')
allT_l.append('FB')
allT_l.append('GHVI')
allT_l.append('EBR/B') 
allT_l.append('GOL0000')
allT_l.append('IWM0000')
allT_l.append('SQ')
allT_l.append('BRZU')
allT_l.append('CSUHD') 
allT_l.append('BIDU0000')
allT_l.append('ABCD')
allT_l.append('ARK')

inception_cs = pd.read_csv(r"C:\Users\Utilizador\OneDrive\Documentos\Malta\LJJ\96923_inception.csv")
all_cs = pd.read_csv(r"C:\Users\Utilizador\OneDrive\Documentos\Malta\LJJ\96923_ALL.csv")
all_saxo = pd.read_csv(r"C:\Users\Utilizador\OneDrive\Documentos\Malta\LJJ\ALL_SAXO.csv")
opts = pd.read_csv(r"C:\Users\Utilizador\OneDrive\Documentos\Malta\LJJ\96923_ALL_EQUITIES.csv")

allCS_df = pd.DataFrame(all_cs)
allSAXO_df = pd.DataFrame(all_saxo)
incCS_df = pd.DataFrame(inception_cs)

saxoOpts_df = allSAXO_df[allSAXO_df['Exchange Description'].map(lambda x: x in saxoOptKV_l)]
saxoEqty_df = allSAXO_df[allSAXO_df['Exchange Description'].map(lambda x: x in saxoEqtyKV_l)]
saxoDepo_df = allSAXO_df[allSAXO_df['Exchange Description'].map(lambda x: x in saxoDepositKV_l)]
saxoFees_df = allSAXO_df[allSAXO_df['Exchange Description'].isna()]
saxoDVD_df = saxoEqty_df[saxoEqty_df['Event'].map(lambda x: x.find('dividend')) != -1]
saxoFees2_df = saxoEqty_df[saxoEqty_df['Event'].map(lambda x: x.find('Finance')) != -1]
saxoEqty_df = saxoEqty_df.drop(saxoDVD_df.index.tolist() + saxoFees2_df.index.tolist())
saxoFees_df = pd.concat([saxoFees_df, saxoFees2_df])
saxoDepo_df = saxoDepo_df[['Trade Date', 'Instrument currency', 'Booked Amount']]
saxoDepo_df.index = saxoDepo_df['Trade Date'].map(lambda x: datetime.strptime(x, '%d-%b-%y').date())
saxoDepo_df = saxoDepo_df.rename(columns = {'Instrument currency': 'Currency', 'Booked Amount': 'Cash Amount'}).drop(['Trade Date'], axis = 1)


for c_problem in ticker_dict:
    rep_l = allCS_df[allCS_df['Narrative'].map(lambda x: x.find(c_problem)) != -1].index.tolist()
    for c_r in rep_l:
        allCS_df.loc[c_r, 'Narrative'] = allCS_df.loc[c_r, 'Narrative'].split(c_problem)[0] + ticker_dict[c_problem] + allCS_df.loc[c_r, 'Narrative'].split(c_problem)[1]
allCS_df = allCS_df.drop(['Group', 'Account'], axis = 1)
allCS_df['Date'] = allCS_df['Date'].map(lambda x: datetime.strptime(x, '%m/%d/%Y').date())
allCS_df = allCS_df.rename(columns = {'Date': 'Trade Date'})
allCS_df = allCS_df.T.dropna(how = 'all').T.sort_values(by = ['Trade Date'])
allCS_df.index = np.arange(allCS_df.shape[0])
allCS_df = allCS_df[allCS_df['Trade Date'] <= cutoff_dt]
allCS_df['Narrative'] = allCS_df['Narrative'].replace(exception_dikt)

if (adjCDI):
    changeDtCDI_l = allCS_df[allCS_df['Narrative'].map(lambda x: x.find('CDI')) != -1].index.tolist()
    allCS_df.loc[changeDtCDI_l, 'Trade Date'] = dt_CDI

incCS_df['Trade Date'] = incCS_df['Trade Date'].map(lambda x: datetime.strptime(x, '%d-%b-%y').date())
incCS_df = incCS_df.T.dropna(how = 'all').T.sort_values(by = ['Trade Date'])
LJJ_init = incCS_df.iloc[0]['Trade Date']
incCS_df['Equity'] = incCS_df['Ticker'].map(lambda x: x.split(' ')[0]).replace({'IWMUSD': 'IWM', 'BRZUUSD': 'BRZU'})
incCS_df['Currency'] = incCS_df['Ticker'].map(lambda x: x[-3:])
incCS_df['Cash Amount'] = incCS_df['Px'] * incCS_df['Quantity'] * (-1)
incCS_df = incCS_df[['Trade Date', 'Currency', 'Cash Amount', 'Equity', 'Quantity']]

allCS_df['Equity'] = np.nan
fwdNF_l = [' (near)', ' (far)']
try:
    UBSfool_d = {'Reg.shs SanDisk Corp. (SNDKV); US80004C2008': 'Reg.shs SanDisk Corp. (SNDK); US80004C2008', 
                 'Reg.shs Redwire Corp.; US75776W1036': 'Reg.shs Redwire Corp. (RDW); US75776W1036'}    
    ubs_usd = pd.read_csv(r"C:\Users\Utilizador\OneDrive\Documentos\Malta\LJJ\UBS_USD_ALL.csv")
    ubs_usd.loc[ubs_usd[ubs_usd[ubs_usd.columns[12]].fillna('').map(lambda x: x.find('payment')) != -1].index, ubs_usd.columns[10]] = 'WITHDRAWAL'
    ubs_usd.loc[ubs_usd[ubs_usd[ubs_usd.columns[13]].fillna('').map(lambda x: x.find('payment')) != -1].index, ubs_usd.columns[10]] = 'WITHDRAWAL'
    ubs_usd[ubs_usd.columns[10]] = ubs_usd[ubs_usd.columns[10]].replace(UBSfool_d)
    for c_kw in fwdNF_l:
            for c_fn in ubs_usd[ubs_usd[ubs_usd.columns[10]].fillna('').map(lambda x: x.find(c_kw)) != -1].index.tolist():
                ubs_usd.loc[c_fn, ubs_usd.columns[10]] = ubs_usd.loc[c_fn, ubs_usd.columns[10]].split(c_kw)[0] + ubs_usd.loc[c_fn, ubs_usd.columns[10]].split(c_kw)[1]    
    UBS_df = prepareUBSfile(ubs_usd)
except:
    print('No USD file from UBS')
try:
    ubs_aud = pd.read_csv(r"C:\Users\Utilizador\OneDrive\Documentos\Malta\LJJ\UBS_AUD_ALL.csv") 
    for c_kw in fwdNF_l:
            for c_fn in ubs_aud[ubs_aud[ubs_aud.columns[10]].fillna('').map(lambda x: x.find(c_kw)) != -1].index.tolist():
                ubs_aud.loc[c_fn, ubs_aud.columns[10]] = ubs_aud.loc[c_fn, ubs_aud.columns[10]].split(c_kw)[0] + ubs_aud.loc[c_fn, ubs_aud.columns[10]].split(c_kw)[1]    
    audUBS_df = prepareUBSfile(ubs_aud)
    if (audUBS_df.shape[0] > 0):
        UBS_df = pd.concat([UBS_df, audUBS_df])
        UBS_df.index = np.arange(UBS_df.shape[0])
except:
    print('No AUD file from UBS')
try:
    UBSfool_d = {'Parts Sociales Adyen N.V. (ADYEN); NL0012969182': 'Reg.shs Adyen N.V. (ADYEN); NL0012969182', 
                 'Reg.shs ASM International NV; NL0000334118': 'Reg.shs ASM International NV (ASM); NL0000334118'}
    ubs_eur = pd.read_csv(r"C:\Users\Utilizador\OneDrive\Documentos\Malta\LJJ\UBS_EUR_ALL.csv") 
    ubs_eur.loc[ubs_eur[ubs_eur[ubs_eur.columns[12]].fillna('').map(lambda x: x.find('payment')) != -1].index, ubs_eur.columns[10]] = 'WITHDRAWAL'
    ubs_eur.loc[ubs_eur[ubs_eur[ubs_eur.columns[13]].fillna('').map(lambda x: x.find('payment')) != -1].index, ubs_eur.columns[10]] = 'WITHDRAWAL'
    ubs_eur[ubs_eur.columns[10]] = ubs_eur[ubs_eur.columns[10]].replace(UBSfool_d)
    for c_kw in fwdNF_l:
            for c_fn in ubs_eur[ubs_eur[ubs_eur.columns[10]].fillna('').map(lambda x: x.find(c_kw)) != -1].index.tolist():
                ubs_eur.loc[c_fn, ubs_eur.columns[10]] = ubs_eur.loc[c_fn, ubs_eur.columns[10]].split(c_kw)[0] + ubs_eur.loc[c_fn, ubs_eur.columns[10]].split(c_kw)[1]
    eurUBS_df = prepareUBSfile(ubs_eur)
    if (eurUBS_df.shape[0] > 0):
        UBS_df = pd.concat([UBS_df, eurUBS_df])
        UBS_df.index = np.arange(UBS_df.shape[0])    
except:
    print('No EUR file from UBS')
try:
    ubs_mxn = pd.read_csv(r"C:\Users\Utilizador\OneDrive\Documentos\Malta\LJJ\UBS_MXN_ALL.csv") 
    for c_kw in fwdNF_l:
            for c_fn in ubs_mxn[ubs_mxn[ubs_mxn.columns[10]].fillna('').map(lambda x: x.find(c_kw)) != -1].index.tolist():
                ubs_mxn.loc[c_fn, ubs_mxn.columns[10]] = ubs_mxn.loc[c_fn, ubs_mxn.columns[10]].split(c_kw)[0] + ubs_mxn.loc[c_fn, ubs_mxn.columns[10]].split(c_kw)[1]
    mxnUBS_df = prepareUBSfile(ubs_mxn)
    if (mxnUBS_df.shape[0] > 0):
        UBS_df = pd.concat([UBS_df, mxnUBS_df])
        UBS_df.index = np.arange(UBS_df.shape[0])    
except:
    print('No MXN file from UBS')
try:
    ubs_chf = pd.read_csv(r"C:\Users\Utilizador\OneDrive\Documentos\Malta\LJJ\UBS_CHF_ALL.csv") 
    for c_kw in fwdNF_l:
            for c_fn in ubs_chf[ubs_chf[ubs_chf.columns[10]].fillna('').map(lambda x: x.find(c_kw)) != -1].index.tolist():
                ubs_chf.loc[c_fn, ubs_chf.columns[10]] = ubs_chf.loc[c_fn, ubs_chf.columns[10]].split(c_kw)[0] + ubs_chf.loc[c_fn, ubs_chf.columns[10]].split(c_kw)[1]    
    chfUBS_df = prepareUBSfile(ubs_eur)
    if (chfUBS_df.shape[0] > 0):
        UBS_df = pd.concat([UBS_df, chfUBS_df])
        UBS_df.index = np.arange(UBS_df.shape[0])  
except:
    print('No CHF file from UBS')            

UBSOPtMess_dict = {'A 31.10.2025    IT008326; Invesco QQQ     41887067': 'Call Invesco QQQ Trust Series I; 94285521', 
                   'A 29.10.2025    IT117354; Invesco QQQ     41887067': 'Call Invesco QQQ Trust Series I; 94285523', 
                   'A 29.10.2025    IT071451; ARK Innovation  25949494': 'Call ARK Innovation ETF; 93635102'}
UBS_df['Description1'] = UBS_df['Description1'].replace(UBSOPtMess_dict)    

OptRev_l = UBS_df[UBS_df['Description2'].map(lambda x: x.find('Exercising;Reversal')) != -1].index.tolist() + UBS_df[UBS_df['Description2'].map(lambda x: x.find('Assignment;Reversal')) != -1].index.tolist()
for OptRev_c in OptRev_l:
    PairRev_c = UBS_df[UBS_df[['Debit', 'Credit']].astype(float).sum(axis = 1) == -UBS_df.loc[OptRev_c, ['Debit', 'Credit']].astype(float).sum()].index
    UBS_df = UBS_df.drop([OptRev_c, PairRev_c.values[0]])

allUBS_df = formatUBS(UBS_df, allCS_df)
allUBSOpts_s = allUBS_df[(allUBS_df['Narrative'].map(lambda x: x.find('CALL')) != -1) | 
                         (allUBS_df['Narrative'].map(lambda x: x.find('PUT')) != -1)]['Narrative']

allFXspot_df = allUBS_df[allUBS_df['Narrative'].map(lambda x: x.find('spot')) != -1]
spotCodes_l = pd.unique(allFXspot_df['Narrative'].map(lambda x: x.split(' - ')[-1])).tolist()
for c_code in spotCodes_l:
    SallFXspot_df = allFXspot_df[allFXspot_df['Narrative'].map(lambda x: x.find(c_code)) != -1]
    for c_FXidx in SallFXspot_df.index:
        allUBS_df.loc[c_FXidx, 'Narrative'] = allFXspot_df.loc[c_FXidx, 'Narrative'].split(' - ')[0] + ' / ' + SallFXspot_df.drop(c_FXidx).iloc[0]['Narrative'].split(' - ')[0] + ', ' + allFXspot_df.loc[c_FXidx, 'Narrative'].split(' - ')[-1] +  ' - spot'

try:
    UBSOptDts_df = openF_df('ubs_options')
except:    
    UBSOptDts_df = pd.DataFrame(np.nan, index = allUBSOpts_s.values, columns = ['Exp Date', 'Related Asset', 'Quantity'])
for new_opt in [o for o in allUBSOpts_s.tolist() if o not in UBSOptDts_df.index.tolist()]:
    UBSOptDts_df.loc[new_opt] = np.nan    
for c_idx in UBSOptDts_df[UBSOptDts_df['Related Asset'].isna().values].index:    
    good_dt = False
    while (not good_dt):
        new_dt = input('Enter expiry date for ' + c_idx + ' (YY/MM/DD): ')
        unwind = input('Unwind (Y or empty)? ').upper()
        try:
            UBSOptDts_df.loc[c_idx, 'Quantity'] = int(c_idx.split(', ')[-1])
        except:
            pass
        if (unwind == 'Y'):
            UBSasset = c_idx.split(',')[0].split(' ')[-1]
            ALLasset_l = pd.unique(opts[opts['Asset'].map(lambda x: x.find(UBSasset)) != -1]['Asset']).tolist()                                    
            ALLasset_l.append('')
            root = tk.Tk()
            tkvar = tk.StringVar(root)
            popupMenu = tk.OptionMenu(root, tkvar, *ALLasset_l, command = on_selection_asset)
            tk.Label(root, text = "Choose an existing asset: ").grid(row = 0, column = 0)
            popupMenu.grid(row = 1, column = 0)
            root.mainloop()                        
            if (target_asset != ''):
                UBSOptDts_df.loc[c_idx, 'Related Asset'] = target_asset            
        try:
            UBSOptDts_df.loc[c_idx, 'Exp Date'] = datetime.strptime(new_dt, '%y/%m/%d').date()
            good_dt = True
        except:
            print('again...')              
for c_narr in UBSOptDts_df[UBSOptDts_df['Related Asset'].isna()].index:
    UBSOptDts_df.loc[c_narr, 'Related Asset'] = c_narr + ' - ' + UBSOptDts_df.loc[c_narr, 'Exp Date'].strftime('%b %d %Y')            
UBSOptDts_df.to_sql('ubs_options', Fengine, index = True, if_exists = 'replace')
allCS_df = pd.concat([allCS_df, allUBS_df])


year_list = pd.unique(allCS_df['Trade Date'].map(lambda x: x.year)).tolist()
summaryEqIR_s = pd.Series(np.nan, index = year_list, name = 'Realized')

excludedPX_df = get_excPxs(dropLAST, cutoff_dt)
split_df, split2_df = getSplits()
expensesEUR_l = getList('expenses_ljj', year_list)
LAInt_l = getList('interest_ljj', year_list)
dividends_l = getList('dividends_ljj', year_list)
taxes_l = getList('taxes_ljj', year_list)
eqoAccMTM_l = getList('eqtopts_ljj', year_list)
swapAccPV_l = getList('swaps_ljj', year_list)
fxAccMTM_l = getList('fxall_ljj', year_list)
fxoAccMTM_l = getList('fxoptions_ljj', year_list)
LJJAUM_df = getLJJ(year_list)
extras_df = getExtras()

allCS_df = allCS_df.sort_values(by = ['Trade Date']) 
allCS_df.index = np.arange(allCS_df.shape[0])
for c_idx in allCS_df.index:
    key = re.sub(r'[^/^-^.^A-Z^0-9^ ]', '', allCS_df.loc[c_idx, 'Narrative'].upper())
    narr_l = key.split(' ')
    ccy_f = [c for c in narr_l if c in ccy_l]
    swap_f = [c for c in narr_l if c in swap_l]
    dvd_f = [c for c in narr_l if c in dvd_l]
    fee_f = [c for c in narr_l if c in fee_l]
    if (len(dvd_f) > 0):
        allCS_df.loc[c_idx, 'Equity'] = 'DIVIDENDS'
    elif (len(swap_f) > 0):
        allCS_df.loc[c_idx, 'Equity'] = 'SWAP'
    elif (len(ccy_f) > 0):
        allCS_df.loc[c_idx, 'Equity'] = 'FX'     
    elif (narr_l[0] == 'DEBIT') | (narr_l[0] == 'CREDIT'):
        narr_l = pd.Series(narr_l).replace(ticker_dikt).tolist()
        tk_l = [t for t in narr_l if t in allT_l]
        if (len(tk_l) > 0):
            allCS_df.loc[c_idx, 'Equity'] = allCS_df.loc[c_idx, 'Narrative'].split('Settlement of ')[1].split(' ')[0]                       
            
try:
    saxoEqty_df = saxoEqty_df.drop(saxoEqty_df[saxoEqty_df['Event'].map(lambda x: x.find('Transfer')) != -1].index.tolist())
except:
    pass
try:
    saxoEqty_df = saxoEqty_df.drop(saxoEqty_df[saxoEqty_df['Event'] == 'Change'].index)
except:
    pass

NsaxoEqty_df = pd.DataFrame(np.nan, index = saxoEqty_df.index, columns = allCS_df.columns)
NsaxoEqty_df['Trade Date'] = saxoEqty_df['Trade Date'].map(lambda x: datetime.strptime(x, '%d-%b-%y').date()).values
NsaxoEqty_df['Currency'] = saxoEqty_df['Instrument currency'].values
NsaxoEqty_df['Equity'] = saxoEqty_df['Instrument Symbol'].map(lambda x: x.split(':')[0])
NsaxoEqty_df['Cash Amount'] = saxoEqty_df['Booked Amount'].values
NsaxoEqty_df['Narrative'] = saxoEqty_df.index.map(lambda x: saxoEqty_df.loc[x, 'Instrument ISIN'] + ',' + saxoEqty_df.loc[x, 'Event'].split(' ')[1])
NsaxoEqty_df.loc[NsaxoEqty_df[NsaxoEqty_df['Cash Amount'] > 0].index, 'Narrative'] = NsaxoEqty_df[NsaxoEqty_df['Cash Amount'] > 0].index.map(lambda x: 'CREDIT - Settlement of ' + NsaxoEqty_df.loc[x, 'Equity'] + ' ' + NsaxoEqty_df.loc[x, 'Narrative'].split(',')[0] + ' - ' + str(abs(int(NsaxoEqty_df.loc[x, 'Narrative'].split(',')[1]))))
NsaxoEqty_df.loc[NsaxoEqty_df[NsaxoEqty_df['Cash Amount'] < 0].index, 'Narrative'] = NsaxoEqty_df[NsaxoEqty_df['Cash Amount'] < 0].index.map(lambda x: 'DEBIT - Settlement of ' + NsaxoEqty_df.loc[x, 'Equity'] + ' ' + NsaxoEqty_df.loc[x, 'Narrative'].split(',')[0] + ' - ' + str(abs(int(NsaxoEqty_df.loc[x, 'Narrative'].split(',')[1]))))

excFXst_df = saxoOpts_df[saxoOpts_df['Exchange Description'] == 'Inter Bank']
excFXst_df = excFXst_df[excFXst_df['Booked Amount'].map(lambda x: abs(x)) < 10]
saxoOpts_df = saxoOpts_df.drop(excFXst_df.index)
NsaxoOpts_df = pd.DataFrame(np.nan, index = saxoOpts_df.index, columns = allCS_df.columns)

NsaxoOpts_df['Trade Date'] = saxoOpts_df['Trade Date'].map(lambda x: datetime.strptime(x, '%d-%b-%y').date()).values
NsaxoOpts_df['Currency'] = saxoOpts_df['Instrument currency'].values
NsaxoOpts_df['Cash Amount'] = saxoOpts_df['Booked Amount'].values
NsaxoOpts_df.loc[saxoOpts_df[saxoOpts_df['Exchange Description'] == 'Inter Bank'].index, 'Equity'] = 'FX'

SaxoFxOpts_df = NsaxoOpts_df[NsaxoOpts_df['Equity'] == 'FX']
NsaxoOpts_df.loc[SaxoFxOpts_df[SaxoFxOpts_df['Cash Amount'] < 0].index, 'Narrative'] = 'PURCHASE, '
NsaxoOpts_df.loc[SaxoFxOpts_df[SaxoFxOpts_df['Cash Amount'] > 0].index, 'Narrative'] = 'SALE, '
SaxoFxCalls_df = saxoOpts_df.loc[SaxoFxOpts_df.index][saxoOpts_df.loc[SaxoFxOpts_df.index]['Instrument'].map(lambda x: x[-1]) == 'C']
SaxoFxPuts_df = saxoOpts_df.loc[SaxoFxOpts_df.index][saxoOpts_df.loc[SaxoFxOpts_df.index]['Instrument'].map(lambda x: x[-1]) == 'P']
NsaxoOpts_df.loc[SaxoFxCalls_df.index, 'Narrative'] = NsaxoOpts_df.loc[SaxoFxCalls_df.index].index.map(lambda x: NsaxoOpts_df.loc[x, 'Narrative'] + saxoOpts_df.loc[x, 'Instrument Symbol'][:3] + ' CALL / ' + saxoOpts_df.loc[x, 'Instrument Symbol'][-3:] + ' PUT, ' + SaxoFxCalls_df.loc[x, 'Instrument'].split(' ')[2] + ', ' + datetime.strptime(SaxoFxCalls_df.loc[x, 'Instrument'].split(' ')[1], '%Y-%m-%d').strftime('%d %b %Y') + ' - PREMIUM')
NsaxoOpts_df.loc[SaxoFxPuts_df.index, 'Narrative'] = NsaxoOpts_df.loc[SaxoFxPuts_df.index].index.map(lambda x: NsaxoOpts_df.loc[x, 'Narrative'] + saxoOpts_df.loc[x, 'Instrument Symbol'][:3] + ' PUT / ' + saxoOpts_df.loc[x, 'Instrument Symbol'][-3:] + ' CALL, ' + SaxoFxPuts_df.loc[x, 'Instrument'].split(' ')[2] + ', ' + datetime.strptime(SaxoFxPuts_df.loc[x, 'Instrument'].split(' ')[1], '%Y-%m-%d').strftime('%d %b %Y') + ' - PREMIUM')

saxoEqtyOpts_df = saxoOpts_df[saxoOpts_df['Exchange Description'] == 'Options Price Reporting Authority']
NsaxoOpts_df.loc[saxoEqtyOpts_df[saxoEqtyOpts_df['Instrument'].map(lambda x: x[-1]) == 'C'].index, 'Narrative'] = NsaxoOpts_df.loc[saxoEqtyOpts_df[saxoEqtyOpts_df['Instrument'].map(lambda x: x[-1]) == 'C'].index, 'Narrative'].index.map(lambda x: 'CALL ' + saxoEqtyOpts_df.loc[x, 'Instrument Symbol'].split('/')[1].split(':')[0] + ' ' + saxoEqtyOpts_df.loc[x, 'Instrument Symbol'].split('/')[0] + ', ' + str(abs(int(saxoEqtyOpts_df.loc[x, 'Event'].split(' @')[0].split(' ')[-1]))))
NsaxoOpts_df.loc[saxoEqtyOpts_df[saxoEqtyOpts_df['Instrument'].map(lambda x: x[-1]) == 'P'].index, 'Narrative'] = NsaxoOpts_df.loc[saxoEqtyOpts_df[saxoEqtyOpts_df['Instrument'].map(lambda x: x[-1]) == 'P'].index, 'Narrative'].index.map(lambda x: 'PUT ' + saxoEqtyOpts_df.loc[x, 'Instrument Symbol'].split('/')[1].split(':')[0] + ' ' + saxoEqtyOpts_df.loc[x, 'Instrument Symbol'].split('/')[0] + ', ' + str(abs(int(saxoEqtyOpts_df.loc[x, 'Event'].split(' @')[0].split(' ')[-1]))))

if (NsaxoOpts_df[NsaxoOpts_df['Narrative'].isna()].shape[0] > 0):
    print('MISSING OPTION CLASSIFICATION ON SAXO FILE!!!!!!!!!!!!!!!!!!!!')
    print(NsaxoOpts_df[NsaxoOpts_df['Narrative'].isna()])
    
NsaxoDvd_df = pd.DataFrame(np.nan, index = saxoDVD_df.index, columns = allCS_df.columns)
NsaxoDvd_df['Trade Date'] = saxoDVD_df['Trade Date'].map(lambda x: datetime.strptime(x, '%d-%b-%y').date()).values
NsaxoDvd_df['Currency'] = saxoDVD_df['Instrument currency'].values
NsaxoDvd_df['Equity'] = 'DIVIDENDS'
NsaxoDvd_df['Cash Amount'] = saxoDVD_df['Booked Amount'].values
NsaxoDvd_df['Narrative'] = 'CASH DIVIDEND'  

NsaxoFees_df = pd.DataFrame(np.nan, index = saxoFees_df.index, columns = allCS_df.columns)
NsaxoFees_df['Trade Date'] = saxoFees_df['Trade Date'].map(lambda x: datetime.strptime(x, '%d-%b-%y').date()).values
NsaxoFees_df['Currency'] = saxoFees_df['Instrument currency'].values
NsaxoFees_df['Cash Amount'] = saxoFees_df['Booked Amount'].values
NsaxoFees_df['Narrative'] = saxoFees_df['Event'].values
NsaxoFees_df['Equity'] = 'FEE'

allCS_df = pd.concat([allCS_df, NsaxoEqty_df, NsaxoOpts_df, NsaxoDvd_df, NsaxoFees_df]).sort_values(by = ['Trade Date'])
allCS_df.index = np.arange(allCS_df.shape[0])  

allCS_df['Equity'] = allCS_df['Equity'].replace(jokers_dikt).replace(ticker_dikt)
for c_joker in jokers_dikt:
    r_joker = jokers_dikt[c_joker]
    c_allCS_df = allCS_df[allCS_df['Narrative'].map(lambda x: x.find(c_joker)) != -1]
    allCS_df.loc[c_allCS_df.index, 'Narrative'] = c_allCS_df['Narrative'].map(lambda x: x.split(c_joker)[0] + r_joker + x.split(c_joker)[1])    
                                            

allCS_df['Currency'] = allCS_df['Currency'].map(lambda x: re.sub(r'[^A-Z]', '', x))
incCS_df['Currency'] = incCS_df['Currency'].map(lambda x: re.sub(r'[^A-Z]', '', x))
        
nanIdx_l = allCS_df[allCS_df['Equity'].isna()].index.tolist()
fx_l = allCS_df[allCS_df['Equity'] == 'FX'].index.tolist()
swap_l = allCS_df[allCS_df['Equity'] == 'SWAP'].index.tolist()
dvd_l = allCS_df[allCS_df['Equity'] == 'DIVIDENDS'].index.tolist()
fee_l = allCS_df[allCS_df['Equity'] == 'FEE'].index.tolist()
eqIdx_l = [i for i in allCS_df.index if i not in (nanIdx_l + fx_l + swap_l + dvd_l + fee_l)]

fx_df = allCS_df.loc[fx_l]
dvd_df = allCS_df.loc[dvd_l]
swap_df = allCS_df.loc[swap_l]

ccy_l = pd.unique(fx_df['Currency'].tolist())
sFX_df = pd.DataFrame(np.nan, index = FX_df.index, columns = ['Dummy'])
for c_ccy in ccy_l:
    c_ccy = re.sub(r'[^A-Z]', '', c_ccy)
    if (c_ccy != 'USD'):
        c_ccy = 'USD' + c_ccy
        try:
            Cccy_df = FX_df.T[FX_df.columns.map(lambda x: x.find(c_ccy)) != -1].T
            if (Cccy_df.shape[1] > 0):
                Cccy_df = Cccy_df ** (-1)
                Cccy_df.columns = Cccy_df.rename(columns = {c_ccy: c_ccy[-3:] + 'USD'}).columns.tolist()
            else:
                c_ccy = c_ccy[-3:] + 'USD'
                Cccy_df = FX_df.T[FX_df.columns.map(lambda x: x.find(c_ccy)) != -1].T
            sFX_df = sFX_df.join(Cccy_df)
        except:
            print(c_ccy + ' not a ccy')
sFX_df = sFX_df.drop(['Dummy'], axis = 1)  
sFX_df['USDUSD'] = 1
sFX_df['EUREUR'] = 1
if (accountingEUR):
    sEURFX_df = (sFX_df.T / sFX_df['EURUSD'].values).T
else:
    acc_ccy = 'USD'
    sEURFX_df = sFX_df.copy()
PM_df = (Pxs_df[PM_l].loc[sEURFX_df.index].T * sEURFX_df['USDUSD'].values).T
sEURFX_df = sEURFX_df.join(PM_df)       
    
LJJAUM_df['AUM ' + acc_ccy] = LJJAUM_df['Value'] * LJJAUM_df.index.map(lambda x: sEURFX_df[sEURFX_df.index.map(lambda x: x.year) == int(x)][LJJAUM_df.loc[x, 'Currency'] + 'USD'].iloc[-1])
LJJAUM_df.loc[0] = ['EUR', LJJAUM_init, LJJAUM_init * sEURFX_df.loc[LJJ_init, 'EURUSD']]
LJJAUM_df = LJJAUM_df.sort_index()
LJJAUM_Delta_s = round((LJJAUM_df['AUM ' + acc_ccy] - LJJAUM_df['AUM ' + acc_ccy].shift(1)).dropna(), 2)
LJJAUM_Delta_s.name = 'PnL (' + acc_ccy + ')'    

expenses_l = FXconvEUR(expensesEUR_l, year_list)
eqoMTM_l = FXconvUSD(eqoAccMTM_l, year_list)
fxMTM_l = FXconvUSD(fxAccMTM_l, year_list)
fxoMTM_l = FXconvUSD(fxoAccMTM_l, year_list)
swapPV_l = FXconvUSD(swapAccPV_l, year_list)
expenses_s = pd.Series(expenses_l, index = year_list) 

swapPV_s = pd.Series(swapAccPV_l, index = pd.Index(year_list, name = 'USD'))
swapAccPV_df = pd.DataFrame(swapPV_l, index = year_list, columns = [acc_ccy])
swapAccMtM_s = (swapAccPV_df[acc_ccy] - swapAccPV_df[acc_ccy].shift(1)).fillna(0)
if (swapCCY != acc_ccy):
    if (swapCCY == 'EUR'):
        swapPV_s = pd.Series(swapAccPV_l, index = pd.Index(year_list, name = 'USD')) / swapPV_s.index.map(lambda x: sEURFX_df[sEURFX_df.index < datetime(x + 1, 1, 1).date()].index[-1]).map(lambda x: sEURFX_df.loc[x, 'EURUSD']).values
    swapMtM_s = (swapPV_s - swapPV_s.shift(1)).fillna(0) * swapPV_s.index.map(lambda x: sEURFX_df[sEURFX_df.index.map(lambda y: y.year) == x].iloc[-1][swapCCY + 'USD']).values
    swapAccMtM_s = (swapAccPV_df[acc_ccy] - swapAccPV_df[acc_ccy].shift(1)).fillna(0)    
    RswapPV_df = swapAccMtM_s - swapMtM_s
else:
    swapMtM_s = swapAccMtM_s
RswapPV_df = swapAccMtM_s - swapMtM_s

opts_df = pd.DataFrame(opts)
delete_l = opts_df[opts_df['Asset'].map(lambda x: x.find('DELIVERY')) != -1].index.tolist()
EQcalls_df = opts_df[opts_df['Asset'].map(lambda x: x.find('CALL ')) != -1].dropna(how = 'all', axis = 1)
EQputs_df = opts_df[opts_df['Asset'].map(lambda x: x.find('PUT ')) != -1].dropna(how = 'all', axis = 1)
EQopts_df = pd.concat([EQcalls_df, EQputs_df])
EQopts_df = EQopts_df[EQopts_df['Deal'].map(lambda x: x.find('Free')) == -1]
EQopts_df['Currency'] = EQopts_df['Currency'].map(lambda x: re.sub(r'[^A-Z]', '', x))
EQopts_df['Trade Date'] = EQopts_df['Trade Date'].map(lambda x: datetime.strptime(x, '%m/%d/%Y').date())
EQopts_df['Cash Date'] = EQopts_df['Cash Date'].map(lambda x: datetime.strptime(x, '%m/%d/%Y').date())
EQopts_df['Cash Amount'] = EQopts_df['Cash Amount'].map(lambda x: float(re.sub(r'[^-^.^0-9]', '', str(x)))) * (-1)
EQopts_df = EQopts_df.sort_index()

addUBSEqtOpts_df = allUBS_df[allUBS_df['Narrative'].map(lambda x: (x.split(',')[0].split(' ')[-1] in allT_l) & (x.split(' ')[0] in ['CALL', 'PUT']))]
addUBSEqtOpts_df['Quantity'] = UBSOptDts_df.loc[addUBSEqtOpts_df['Narrative'].values, 'Quantity'].values
addUBSEqtOpts_df['Narrative'] = UBSOptDts_df.loc[addUBSEqtOpts_df['Narrative'].values, 'Related Asset'].values

for add_l in addUBSEqtOpts_df.index:
    EQopts_df = EQopts_df.sort_index()
    EQopts_df.loc[EQopts_df.index[-1] + 1] = np.nan
    EQopts_df.loc[EQopts_df.index[-1], 'Group'] = 'Derivatives'
    EQopts_df.loc[EQopts_df.index[-1], 'Account'] = 96923
    EQopts_df.loc[EQopts_df.index[-1], 'Id'] = addUBSEqtOpts_df.loc[add_l, 'Narrative'].split(' ')[1]
    EQopts_df.loc[EQopts_df.index[-1], 'Asset'] = addUBSEqtOpts_df.loc[add_l, 'Narrative']
    EQopts_df.loc[EQopts_df.index[-1], 'Trade Date'] = addUBSEqtOpts_df.loc[add_l, 'Trade Date']
    try:
        EQopts_df.loc[EQopts_df.index[-1], 'Cash Date'] = Pxs_df[Pxs_df.index > addUBSEqtOpts_df.loc[add_l, 'Trade Date']].index[0]
    except:
        EQopts_df.loc[EQopts_df.index[-1], 'Cash Date'] = tomorrow    
    if (addUBSEqtOpts_df.loc[add_l, 'Cash Amount'] < 0):
        EQopts_df.loc[EQopts_df.index[-1], 'Quantity'] = int(addUBSEqtOpts_df.loc[add_l, 'Quantity'])
        EQopts_df.loc[EQopts_df.index[-1], 'Deal'] = 'BUY'
    else:
        EQopts_df.loc[EQopts_df.index[-1], 'Quantity'] = int(addUBSEqtOpts_df.loc[add_l, 'Quantity']) * (-1)
        EQopts_df.loc[EQopts_df.index[-1], 'Deal'] = 'SELL'
    EQopts_df.loc[EQopts_df.index[-1], 'Currency'] = addUBSEqtOpts_df.loc[add_l, 'Currency']
    EQopts_df.loc[EQopts_df.index[-1], 'Unit Price'] = round(abs(addUBSEqtOpts_df.loc[add_l, 'Cash Amount'] / EQopts_df.loc[EQopts_df.index[-1], 'Quantity'] / 100), 2)
    EQopts_df.loc[EQopts_df.index[-1], 'Cash Amount'] = addUBSEqtOpts_df.loc[add_l, 'Cash Amount']

EQopts_df['Cash Amount Acc'] = EQopts_df['Cash Amount'].values
EQopts_df = EQopts_df[EQopts_df['Trade Date'] <= cutoff_dt]
EQopts_df['Cash Amount'] *= EQopts_df.index.map(lambda x: sEURFX_df.loc[EQopts_df.loc[x, 'Trade Date'], EQopts_df.loc[x, 'Currency'] + 'USD']).values
EQopts_df['Quantity'] = EQopts_df['Quantity'].astype(float).astype(int)

EqOptExercise_df = UBS_df[UBS_df['Description2'].map(lambda x: x in ['Exercising', 'Assignment'])]
EqOptExercise_df[['Debit', 'Credit']] = EqOptExercise_df[['Debit', 'Credit']].astype(float)
for c_exe in EqOptExercise_df.index:
    if (EqOptExercise_df.loc[c_exe, 'Description2'] == 'Exercising') & (EqOptExercise_df.loc[c_exe, ['Debit', 'Credit']].sum() < 0):
        opt = 'CALL'
        q = int(int(EqOptExercise_df.loc[c_exe, 'Description3'].split('Number/Amt. ')[1].split(';')[0]) / 100)
    elif (EqOptExercise_df.loc[c_exe, 'Description2'] == 'Exercising') & (EqOptExercise_df.loc[c_exe, ['Debit', 'Credit']].sum() > 0):
        opt = 'PUT'
    elif (EqOptExercise_df.loc[c_exe, 'Description2'] == 'Assignment') & (EqOptExercise_df.loc[c_exe, ['Debit', 'Credit']].sum() < 0):
        opt = 'PUT'
    elif (EqOptExercise_df.loc[c_exe, 'Description2'] == 'Assignment') & (EqOptExercise_df.loc[c_exe, ['Debit', 'Credit']].sum() > 0):
        opt = 'CALL'
    asset_desc = opt + ' ' + EqOptExercise_df.loc[c_exe, 'Description3'].split('price: ')[1].split(';')[0][:-4] + ' ' + EqOptExercise_df.loc[c_exe, 'Description1'].split('(')[1].split(')')[0] + ', 100 - ' + datetime.strptime(EqOptExercise_df.loc[c_exe, 'Trade date'], '%m/%d/%Y').date().strftime('%b %d %Y')    

    prev_df = EQopts_df[EQopts_df['Asset'].map(lambda x: x.find(asset_desc)) != -1]
    EQopts_df.loc[EQopts_df.index[-1] + 1] = np.nan
    
    EQopts_df.loc[EQopts_df.index[-1], ['Group', 'Account']] = ['Derivatives', '96923']
    EQopts_df.loc[EQopts_df.index[-1], 'Asset'] = asset_desc
    EQopts_df.loc[EQopts_df.index[-1], ['Trade Date', 'Cash Date']] = datetime.strptime(asset_desc.split('- ')[1], '%b %d %Y').date()
    EQopts_df.loc[EQopts_df.index[-1], 'Quantity'] = -prev_df['Quantity'].sum()
    EQopts_df.loc[EQopts_df.index[-1], ['Unit Price', 'Cash Amount', 'Cash Amount Acc']] = 0
    EQopts_df.loc[EQopts_df.index[-1], 'Currency'] = prev_df.iloc[-1]['Currency']
    if (EQopts_df.loc[EQopts_df.index[-1], 'Quantity'] > 0):
        EQopts_df.loc[EQopts_df.index[-1], 'Deal'] = 'BUY'
    elif (EQopts_df.loc[EQopts_df.index[-1], 'Quantity'] < 0):
        EQopts_df.loc[EQopts_df.index[-1], 'Deal'] = 'SELL'
    else:
        EQopts_df = EQopts_df.drop(EQopts_df.index[-1])

EQopts_df = EQopts_df.sort_values(by = ['Trade Date'], ascending = False)
EQopts_df.index = np.arange(EQopts_df.shape[0]) + 1

opts_df = EQopts_df[['Asset', 'Trade Date', 'Quantity', 'Currency', 'Cash Amount', 'Cash Amount Acc']]
opts_df['Currency'] = opts_df['Currency'].map(lambda x: re.sub(r'[^A-Z]', '', x))


opts_df['Expiry'] = opts_df['Asset'].map(lambda x: datetime.strptime(re.sub(' ', '', x.split('- ')[1]), '%b%d%Y').date())
opts_df['Equity'] = opts_df['Asset'].map(lambda x: re.sub(r'[^0-9^A-Z]','', x.upper()))
opts_df = opts_df.drop(['Asset'], axis = 1).sort_values(by = ['Trade Date'])
opts_df.index = np.arange(opts_df.shape[0])
nvda_df = opts_df[opts_df['Equity'].map(lambda x: x.find('NVDA')) != -1]
nvda_df['New Asset'] = nvda_df['Equity'].map(lambda x: re.sub(r'[^0-9]', '', x.split('NVDA')[0])).astype(float).astype(int)
nvdaR_df = nvda_df[nvda_df['New Asset'] > 300]
nvdaR_l = nvdaR_df.index.tolist()
nvdaR_df['Quantity'] *= 10
nvdaR_df['New Asset 1'] = nvdaR_df['Equity'].map(lambda x: re.sub(r'[^A-Z]', '', x.split('NVDA')[0]))
nvdaR_df['New Asset 1'] = nvdaR_df.index.map(lambda x: nvdaR_df.loc[x, 'New Asset 1'] + str(int(nvdaR_df.loc[x, 'New Asset'] / 10)) + 'NVDA' + nvdaR_df.loc[x, 'Equity'].split('NVDA')[1])
opts_df.loc[nvdaR_l, 'Quantity'] = nvdaR_df['Quantity'].values
opts_df.loc[nvdaR_l, 'Equity'] = nvdaR_df['New Asset 1'].values

eqtyCS_df = allCS_df.loc[eqIdx_l]
eqtyCS_df['Quantity'] = eqtyCS_df['Narrative'].map(lambda x: int(re.sub(r'[^0-9]', '', x.split(' - REVERSAL')[0].split('- ')[-1].split('.')[0])))
eqtyCS_df = eqtyCS_df.drop(['Narrative'], axis = 1)
for c_ticker in split_df.index:
    idx_l = eqtyCS_df[eqtyCS_df['Equity'] == c_ticker][eqtyCS_df[eqtyCS_df['Equity'] == c_ticker]['Trade Date'] <= split_df.loc[c_ticker, 'Dates']].index
    eqtyCS_df.loc[idx_l, 'Quantity'] *= split_df.loc[c_ticker, 'Ratios']
for c_ticker in split2_df.index:
    idx_l = eqtyCS_df[eqtyCS_df['Equity'] == c_ticker][eqtyCS_df[eqtyCS_df['Equity'] == c_ticker]['Trade Date'] <= split2_df.loc[c_ticker, 'Dates']].index
    eqtyCS_df.loc[idx_l, 'Quantity'] *= split2_df.loc[c_ticker, 'Ratios']    

Feqty_df = pd.concat([incCS_df, eqtyCS_df])
Feqty_df['Cash Amount'] = Feqty_df['Cash Amount'].map(lambda x: float(re.sub(r'[^-^.^0-9]', '', str(x))))
Feqty_df['Quantity'] = Feqty_df['Quantity'].map(lambda x: abs(x)) * Feqty_df['Cash Amount'].map(lambda x: np.sign(x)) * (-1)
Feqty_df.index = np.arange(Feqty_df.shape[0])
Feqty_df = Feqty_df.sort_values(by = ['Trade Date'])

NeqtyCS_df = allCS_df.loc[nanIdx_l].copy()
NeqtyCS_df['Equity'] = NeqtyCS_df['Narrative'].map(lambda x: x.split(' ')[0])
eqtyOPT_l = NeqtyCS_df[NeqtyCS_df['Equity'] == 'DEBIT'].index.tolist() + NeqtyCS_df[NeqtyCS_df['Equity'] == 'CREDIT'].index.tolist()
nanIdx_l = [i for i in NeqtyCS_df.index if i not in eqtyOPT_l]
NeqtyCS_df = NeqtyCS_df.loc[nanIdx_l]
missing_df = allCS_df.loc[eqtyOPT_l][allCS_df.loc[eqtyOPT_l]['Narrative'].map(lambda x: len(x.split('Settlement of ')[-1].split(' '))) > 1].drop(['Currency'], axis = 1).dropna(axis = 1)

bond_l = NeqtyCS_df[NeqtyCS_df['Equity'] == 'PURCHASE'].index.tolist() + NeqtyCS_df[NeqtyCS_df['Equity'] == 'SALE'].index.tolist() + NeqtyCS_df[NeqtyCS_df['Equity'] == 'PRINCIPAL'].index.tolist()
bond_df = NeqtyCS_df.loc[bond_l]
bondBU_df = bond_df.copy()
bond_df = bond_df.drop(['Equity'], axis = 1).rename(columns = {'Narrative': 'Equity'})
bond_df['Equity'] = bond_df['Equity'].map(lambda x: x.split('OF ')[1].split(' ')[0])
bond_df['Cash Amount'] = bond_df['Cash Amount'].map(lambda x: float(re.sub(r'[^-^.^0-9]', '', str(x))))

bond_df = complementBonds(bond_df)
Feqty_df = pd.concat([Feqty_df, bond_df])
Feqty_df.index = np.arange(Feqty_df.shape[0])
Feqty_df['Quantity'] = Feqty_df['Quantity'].astype(float).astype(int)
Feqty_df['Cash Amount Acc'] = Feqty_df['Cash Amount'].values
Feqty_df['Cash Amount'] *= Feqty_df.index.map(lambda x: sEURFX_df.loc[Feqty_df.loc[x, 'Trade Date'], Feqty_df.loc[x, 'Currency'] + 'USD']).values
Feqty_df['Cash Amount'] = Feqty_df['Cash Amount'].map(lambda x: round(x, 2))

nanIdx_l = [i for i in NeqtyCS_df.index if i not in bond_l]
NeqtyCS_df = NeqtyCS_df.loc[nanIdx_l]

int_l = NeqtyCS_df[NeqtyCS_df['Equity'] == 'INTEREST'].index.tolist()
int_df = NeqtyCS_df.loc[int_l]
nanIdx_l = [i for i in NeqtyCS_df.index if i not in int_l]
NeqtyCS_df = NeqtyCS_df.loc[nanIdx_l]
        
cash_l = NeqtyCS_df[NeqtyCS_df['Equity'] == 'DEPOSIT'].index.tolist() + NeqtyCS_df[NeqtyCS_df['Equity'] == 'WITHDRAWAL'].index.tolist()
cash_df = NeqtyCS_df.loc[cash_l].sort_index()
Scash_df = cash_df[['Currency', 'Cash Amount']].copy()
Scash_df['Cash Amount'] = Scash_df['Cash Amount'].map(lambda x: float(re.sub(r'[^-^.^0-9]', '', str(x))))
Scash_df.index = cash_df['Trade Date'].tolist()

Scash_df = pd.concat([Scash_df, extras_df, saxoDepo_df]).sort_index()
Scash_df['Trade Date'] = Scash_df.index
Scash_df.index = np.arange(Scash_df.shape[0])
Scash_df['Value (' + acc_ccy + ')'] = Scash_df['Cash Amount'] * Scash_df.index.map(lambda x: sEURFX_df.loc[Scash_df.loc[x, 'Trade Date'], Scash_df.loc[x, 'Currency'] + 'USD']).values
Scash_s = round(Scash_df[['Trade Date', 'Value (' + acc_ccy + ')']].groupby(['Trade Date'])['Value (' + acc_ccy + ')'].sum().replace({0: np.nan}).dropna(), 2)
YScash_s = pd.DataFrame(0, index = year_list, columns = ['dummy']).join(Scash_s.groupby(Scash_s.index.map(lambda x: x.year)).sum()).fillna(0)
YScash_s = YScash_s[YScash_s.columns[-1]]

nanIdx_l = [i for i in NeqtyCS_df.index if i not in cash_l]
NeqtyCS_df = NeqtyCS_df.loc[nanIdx_l]

pfee_l = NeqtyCS_df[NeqtyCS_df['Equity'] == 'AMOUNT'].index.tolist() + NeqtyCS_df[NeqtyCS_df['Equity'] == 'ADR'].index.tolist() + NeqtyCS_df[NeqtyCS_df['Equity'] == 'EXERCISE'].index.tolist() + NeqtyCS_df[NeqtyCS_df['Equity'] == 'AGENCY'].index.tolist() + NeqtyCS_df[NeqtyCS_df['Equity'] == 'REVERSAL'].index.tolist() + NeqtyCS_df[NeqtyCS_df['Equity'].map(lambda x: x.find('FEE')) != -1].index.tolist()
fee_df = pd.concat([NeqtyCS_df.loc[pfee_l], allCS_df.loc[fee_l]]).sort_values(by = ['Trade Date'])
exc_l = NeqtyCS_df[NeqtyCS_df['Equity'] == 'EXCEPTION'].index.tolist()
exc_df = NeqtyCS_df.loc[exc_l]

print('')
print('')
print('******************* ALIEN LINES: ' + str(missing_df.shape[0] + exc_df.shape[0]) + ' *******************')
if ((missing_df.shape[0] + exc_df.shape[0]) != alien_lines):
    print('**NEW ALIEN, ATENTION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print('')
print(missing_df['Narrative'])
print(exc_df)
print('')

if (optSTR):
    EQoptSTR_df = opts_df[opts_df['Cash Amount'] == 0]
    opts_df = opts_df.drop(EQoptSTR_df.index)
    EQoptSTR_df['Bet Structure'] = ['NO', 'NO', 'NO', 'NO', 'NO', 'YES', 'YES', 'NO', 'NO', 'YES', 'NO', 
                                    'YES', 'YES', 'YES', 'NO', 'NO', 'YES', 'YES', 'YES', 'YES']
    print('POSSIBLE OPTION STRUCTURES:')
    print(EQoptSTR_df[['Equity', 'Expiry', 'Quantity', 'Bet Structure']])    
    if (EQoptSTR_df[EQoptSTR_df['Bet Structure'] == 'YES']['Quantity'].sum() != 0):
        print('############################# WRONG OPTIONS PAIRING #############################')
        print('############################# WRONG OPTIONS PAIRING #############################')

    EQoptSTR_df = EQoptSTR_df[EQoptSTR_df['Bet Structure'] == 'YES']
    EQoptSTR_df['Asset'] = EQoptSTR_df['Equity'].map(lambda x: re.sub(r'[^A-Z]', '', x[:-9]))
    EQoptSTR_df['Quantity'] *= EQoptSTR_df['Equity'].map(lambda x: int(x[:-9][-3:])).values
    EQoptSTR_df['Init Premium'] = EQoptSTR_df.index.map(lambda x: opts_df[opts_df['Equity'] == EQoptSTR_df.loc[x, 'Equity']]['Cash Amount'].sum())
    EQoptSTR_df['Init Date'] = EQoptSTR_df.index.map(lambda x: opts_df[opts_df['Equity'] == EQoptSTR_df.loc[x, 'Equity']]['Trade Date'].iloc[0])
    dt_l = pd.unique(EQoptSTR_df['Trade Date']).tolist()
    excOPTSTR_l = opts_df[opts_df['Equity'].map(lambda x: (x in EQoptSTR_df['Equity'].tolist()))].index.tolist()    
    opts_df = opts_df.drop(excOPTSTR_l)
    EQoptSTR_df['k'] = EQoptSTR_df['Equity'].map(lambda x: int(re.sub(r'[^0-9]', '', x[:-12])))
    dt_l = pd.unique(EQoptSTR_df['Trade Date']).tolist()
    EqOpSTR_df = pd.DataFrame(index = dt_l, columns = ['Equity', 'Taxable'])
    EqOpSTR_df['Date'] = dt_l
    for c_dt in dt_l:
        sEQoptSTR_df = EQoptSTR_df[EQoptSTR_df['Trade Date'] == c_dt]
        casset_l = pd.unique(sEQoptSTR_df['Asset']).tolist()
        for c_asset in casset_l:
            EqOpSTR_df.loc[c_dt, 'Equity'] = c_asset
            EqOpSTR_df.loc[c_dt, 'Taxable'] = round(sEQoptSTR_df[['Quantity', 'k']].prod(axis = 1).sum() *  
                                                    sEURFX_df.loc[c_dt, sEQoptSTR_df[sEQoptSTR_df['Asset'] == c_asset]['Currency'].iloc[0] + 'USD'] + 
                                                    sEQoptSTR_df['Init Premium'].sum(), 2)
    EqOpSTR_df['Date'] = dt_l      
    EQoptSTR_df['Asset'] = EQoptSTR_df['Asset'].map(lambda x: re.sub('PUT', '', re.sub('CALL', '', x)))
    EqOpSTR_df.index = np.arange(EqOpSTR_df.shape[0])    

    excFeqty_df = pd.DataFrame(index = ['dummy'], columns = Feqty_df.columns)
    for i, c_dt in enumerate(EQoptSTR_df['Expiry'].tolist()):
        c_ticker = EQoptSTR_df.iloc[i]['Asset']
        c_qty = EQoptSTR_df.iloc[i]['Quantity']
        c_cash = round(EQoptSTR_df.iloc[i]['k'] * c_qty / 1000, 0)
        c_date = EQoptSTR_df.iloc[i]['Expiry']
        sFeqty_df = Feqty_df[Feqty_df['Trade Date'] >= c_dt].copy()

        idx_st =  sFeqty_df.index[0]
        s_ticker = sFeqty_df.loc[idx_st, 'Equity']
        s_qty = sFeqty_df.loc[idx_st, 'Quantity']   
        s_date = sFeqty_df.loc[idx_st, 'Trade Date']
        s_cash = round(sFeqty_df.loc[idx_st, 'Cash Amount'] / sEURFX_df.loc[Feqty_df.loc[idx_st, 'Trade Date'], Feqty_df.loc[idx_st, 'Currency'] + 'USD'] / 1000, 0)
        while (c_ticker != s_ticker) | (c_qty != -s_qty) | ((c_cash != s_cash) & (c_date != s_date)):
            idx_st = sFeqty_df.loc[idx_st:].index[1]
            s_ticker = sFeqty_df.loc[idx_st, 'Equity']
            s_qty = sFeqty_df.loc[idx_st, 'Quantity']  
            s_date = sFeqty_df.loc[idx_st, 'Trade Date']
            s_cash = round(sFeqty_df.loc[idx_st, 'Cash Amount'] / sEURFX_df.loc[Feqty_df.loc[idx_st, 'Trade Date'], Feqty_df.loc[idx_st, 'Currency'] + 'USD'] / 1000, 0)
        excFeqty_df.loc[idx_st] = sFeqty_df.loc[idx_st]

    excFeqty_df = excFeqty_df.drop(['dummy'])
    excEqt_l = excFeqty_df.index.tolist()
    Feqty_df = Feqty_df.drop(excEqt_l)

if (fifo):
    Feqty_df = taxableFIFO(Feqty_df)
    opts_df = taxableFIFO(opts_df)    
else:    
    Feqty_df = taxableAVG(Feqty_df, False)
    opts_df = taxableAVG(opts_df, True)

x = Feqty_df[['Equity', 'Cash Amount', 'Taxable', 'Quantity']].groupby(['Equity'])[['Cash Amount', 'Taxable', 'Quantity']].sum()
z = x[x['Quantity'] == 0]
z['Diff'] = round(abs(z['Cash Amount'] - z['Taxable']) / 100, 0)
print('')
print('Equities realized check:')
print(z.sort_values(by = ['Diff'], ascending = False)[:10])
print('')

expO_l = pd.unique(opts_df['Equity']).tolist()
expO_df = pd.DataFrame(index = expO_l, columns = ['Quantity', 'Taxable'])
expO_df['Quantity'] = opts_df[['Equity', 'Quantity']].groupby(opts_df['Equity'])['Quantity'].sum()
expO_df['Taxable'] = opts_df[['Equity', 'Cash Amount']].groupby(opts_df['Equity'])['Cash Amount'].sum()
expO_df['Expiry'] = expO_df.index.map(lambda x: opts_df[opts_df['Equity'] == x].iloc[0].loc['Expiry'])
expO_df['Equity'] = expO_df.index
expO_df['Quantity'] = expO_df['Quantity'].replace({0: np.nan})
expO_df = expO_df.dropna()
expO_df.index = np.arange(expO_df.shape[0])
for c_idx in expO_df.index.tolist():
    c_eq = expO_df.loc[c_idx, 'Equity']
    Sopts_df = opts_df[opts_df['Equity'] == c_eq]
    st = Sopts_df[Sopts_df['OCA'] == 'O'].index[-1]
    end = Sopts_df.loc[st:].index[-1]
    expO_df.loc[c_idx, 'Taxable'] -= Sopts_df.loc[st: end]['Taxable'].sum()
    
unEqty_s = pd.Series(0, index = year_list)
unACCEqty_s = pd.Series(0, index = year_list)
for c_year in year_list:
    last_dt = Pxs_df[Pxs_df.index < datetime(c_year + 1, 1, 1).date()].index[-1]
    sFeqty_df = Feqty_df[Feqty_df['Trade Date'].map(lambda x: x.year) <= c_year]
    cSUM_s = sFeqty_df[['Equity', 'Quantity']].groupby(['Equity'])['Quantity'].sum().replace({0: np.nan}).dropna().astype(int)
    cSUM_df = pd.DataFrame(cSUM_s)
    cSUM_df['Currency'] = np.nan
    cSUM_df[accMethod] = np.nan 
    cSUM_df[accMethod + ' Acc'] = np.nan
    for c_eq in cSUM_s.index:
        cSUM_df.loc[c_eq, 'Currency'] = sFeqty_df[sFeqty_df['Equity'] == c_eq]['Currency'].iloc[-1]
        cSUM_df.loc[c_eq, accMethod] = (abs(sFeqty_df[sFeqty_df['Quantity'] >= 0][sFeqty_df[sFeqty_df['Quantity'] >= 0]['Equity'] == c_eq]['Cash Amount'].sum()) / 
                                        sFeqty_df[sFeqty_df['Quantity'] >= 0][sFeqty_df[sFeqty_df['Quantity'] >= 0]['Equity'] == c_eq]['Quantity'].sum())
        cSUM_df.loc[c_eq, accMethod + ' Acc'] = (abs(sFeqty_df[sFeqty_df['Quantity'] >= 0][sFeqty_df[sFeqty_df['Quantity'] >= 0]['Equity'] == c_eq]['Cash Amount Acc'].sum()) / 
                                                 sFeqty_df[sFeqty_df['Quantity'] >= 0][sFeqty_df[sFeqty_df['Quantity'] >= 0]['Equity'] == c_eq]['Quantity'].sum())
    stockO_l = pd.unique([t for t in cSUM_df.index.tolist() if t in all_tickers.index.tolist()]).tolist()
    EqPx_l = pd.Series(stockO_l).map(lambda x: all_tickers.drop_duplicates().loc[x, 'tickers'])
    EqPx_l = [t for t in pd.Series(EqPx_l).replace({'SHOP US': 'SHOP CN'}).map(lambda x: re.sub(' GY', ' GR', x)).tolist() if t in Pxs_df.columns.tolist()]
    exc_l = [t for t in cSUM_df.index if t not in pd.Series(EqPx_l).map(lambda x: x.split(' ')[0]).tolist()]      
    pxs_s = pd.Series(pd.Series(EqPx_l).replace({'SHOP US': 'SHOP CN'}).map(lambda x: re.sub(' GY', ' GR', x)).map(lambda x: Pxs_df.loc[last_dt, x]).values, index = 
                      pd.Series(EqPx_l).map(lambda x: x.split(' ')[0]), name = 'RT Pxs').map(lambda x: round(x, 2))  
    cSUM_df.loc[pxs_s.index, str(c_year) + ' Closing Acc'] = pxs_s.values
    try:
        cSUM_df.loc[exc_l, str(c_year) + ' Closing Acc'] = excludedPX_df.loc[c_year, exc_l]
    except:
        yf_l = [s for s in exc_l if s not in excludedPX_df.columns]
        excludedPX_df = fixExc(excludedPX_df, yf_l)
        cSUM_df.loc[exc_l, str(c_year) + ' Closing Acc'] = excludedPX_df.loc[c_year, exc_l]        
        
    cSUM_df.loc[pxs_s.index, str(c_year) + ' Closing'] = pxs_s.values * cSUM_df.loc[pxs_s.index, 'Currency'].map(lambda x: sEURFX_df.loc[last_dt, x + 'USD'])
    cSUM_df.loc[exc_l, str(c_year) + ' Closing'] = excludedPX_df.loc[c_year, exc_l] * cSUM_df.loc[exc_l, 'Currency'].map(lambda x: sEURFX_df.loc[last_dt, x + 'USD'])
    
    cSUM_df['Unrealized (' + acc_ccy + ')'] = cSUM_df['Quantity'] * (cSUM_df[str(c_year) + ' Closing'] - cSUM_df[accMethod])
    cSUM_df['Unrealized (' + acc_ccy + ')'] = cSUM_df['Unrealized (' + acc_ccy + ')'].map(lambda x: round(x, 2))
    cSUM_df['Unrealized Acc'] = cSUM_df['Quantity'] * (cSUM_df[str(c_year) + ' Closing Acc'] - cSUM_df[accMethod + ' Acc']) * cSUM_df.index.map(lambda x: sEURFX_df.loc[last_dt, cSUM_df.loc[x, 'Currency'] + 'USD'])       
    cSUM_df['Unrealized Acc'] = cSUM_df['Unrealized Acc'].map(lambda x: round(x, 2))    
    unEqty_s.loc[c_year] = cSUM_df['Unrealized (' + acc_ccy + ')'].sum()
    unACCEqty_s.loc[c_year] = cSUM_df['Unrealized Acc'].sum()
    
    exc_l = [t for t in exc_l if t not in excludedPX_df.loc[c_year].dropna().index.tolist()]
    print(str(c_year) + ' missing stocks:')
    print(exc_l)
    print('')                  
unEqty_s = unEqty_s - unEqty_s.shift(1).fillna(0)
unACCEqty_s = unACCEqty_s - unACCEqty_s.shift(1).fillna(0)  
RunEqty_s = unEqty_s - unACCEqty_s

eqoRP_s = pd.Series(0, index = year_list)
eqoACCRP_s = pd.Series(0, index = year_list)
for c_year in year_list:
    Sopts_df = opts_df[opts_df['Trade Date'].map(lambda x: x.year) <= c_year]
    Sopts_df = Sopts_df[Sopts_df['Expiry'].map(lambda x: x.year) > c_year]
    sSopts_df = Sopts_df[['Equity', 'Quantity', 'Cash Amount', 'Cash Amount Acc']].groupby(['Equity'])[['Quantity', 'Cash Amount', 'Cash Amount Acc']].sum().replace({0: np.nan}).dropna()
    sSopts_df['Currency'] = ''
    for c_eq in sSopts_df.index:
        sSopts_df.loc[c_eq, 'Currency'] = opts_df[opts_df['Equity'] == c_eq]['Currency'].iloc[-1]
        sSopts_df.loc[c_eq, 'Cash Amount'] = opts_df[opts_df['Equity'] == c_eq]['Cash Amount'].sum() - opts_df[opts_df['Equity'] == c_eq]['Taxable'].sum()
        sSopts_df.loc[c_eq, 'Cash Amount Acc'] = opts_df[opts_df['Equity'] == c_eq]['Cash Amount Acc'].sum() - (opts_df[opts_df['Equity'] == c_eq]['Taxable'] * opts_df[opts_df['Equity'] == c_eq].index.map(lambda x: 1 / (sEURFX_df.loc[opts_df.loc[x, 'Trade Date'], opts_df.loc[x, 'Currency'] + 'USD'])).values).sum()
    eqoRP_s.loc[c_year] = sSopts_df['Cash Amount'].sum()
    eqoACCRP_s.loc[c_year] = sSopts_df['Cash Amount Acc'].sum()

eqoMTM_df = pd.DataFrame(eqoRP_s, columns = ['Residual Premium (' + acc_ccy + ')'])
eqoAccMTM_df = pd.DataFrame(eqoACCRP_s, columns = ['Residual Premium (Acc)'])
eqoMTM_df['Unrealized'] = eqoMTM_l
eqoAccMTM_df['Unrealized'] = eqoAccMTM_l
eqoMTM_s = round(eqoMTM_df.sum(axis = 1), 2)
eqoAccMTM_s = round(eqoAccMTM_df.sum(axis = 1), 2)
unEqOp_s = eqoMTM_s - eqoMTM_s.shift(1).fillna(0)
unAccEqOp_s = eqoAccMTM_s - eqoAccMTM_s.shift(1).fillna(0)
RunEqOp_s = unEqOp_s - unAccEqOp_s    
    
swap_df['Cash Amount'] = swap_df['Cash Amount'].map(lambda x: float(re.sub(r'[^-^.^0-9]', '', str(x))))
swap_df['Taxable'] = (swap_df['Cash Amount'] * swap_df.index.map(lambda x: sEURFX_df.loc[swap_df.loc[x, 'Trade Date'], swap_df.loc[x, 'Currency'] + 'USD']).values).map(lambda x: round(x, 2))
fee_df['Cash Amount'] = fee_df['Cash Amount'].astype(str).map(lambda x: float(re.sub(r'[^-^.^0-9]', '', x)))
fee_df['Taxable'] = (fee_df['Cash Amount'] * fee_df.index.map(lambda x: sEURFX_df.loc[fee_df.loc[x, 'Trade Date'], fee_df.loc[x, 'Currency'] + 'USD']).values).map(lambda x: round(x, 2))
int_df['Cash Amount'] = int_df['Cash Amount'].astype(str).map(lambda x: float(re.sub(r'[^-^.^0-9]', '', x)))
int_df['Taxable'] = (int_df['Cash Amount'] * int_df.index.map(lambda x: sEURFX_df.loc[int_df.loc[x, 'Trade Date'], int_df.loc[x, 'Currency'] + 'USD']).values).map(lambda x: round(x, 2))
dvd_df['Cash Amount'] = dvd_df['Cash Amount'].astype(str).map(lambda x: float(re.sub(r'[^-^.^0-9]', '', x)))
dvd_df['Taxable'] = (dvd_df['Cash Amount'] * dvd_df.index.map(lambda x: sEURFX_df.loc[dvd_df.loc[x, 'Trade Date'], dvd_df.loc[x, 'Currency'] + 'USD']).values).map(lambda x: round(x, 2))

fx_df = fx_df.drop(['Equity'], axis = 1).rename(columns = {'Narrative': 'Equity'})
fx_df['Cash Amount'] = fx_df['Cash Amount'].map(lambda x: float(re.sub(r'[^-^.^0-9]', '', str(x))))

FXOpts_df = fx_df[fx_df['Equity'].map(lambda x: x.find('CALL')) != -1].copy()

suspectFXOpt_df = FXOpts_df[FXOpts_df['Equity'].map(lambda x: len(x.split(' '))) < 10]
for c_sus in suspectFXOpt_df.index:
    FXOpts_df.loc[c_sus, 'Equity'] = (FXOpts_df.loc[c_sus, 'Equity'] + ', ' + 
                                      UBSOptDts_df.loc[FXOpts_df.loc[c_sus, 'Equity'], 'Exp Date'].strftime('%d %b %Y') + ' - PREMIUM')
    
FXNDF_df = fx_df[fx_df['Equity'].map(lambda x: x.find('NDF')) != -1].copy()
FXNDF_df['Taxable'] = FXNDF_df['Cash Amount']

FXOpts_df['Nature'] = FXOpts_df['Equity'].map(lambda x: re.sub(r'[^A-Z]', '', x.split('-')[1]))
FXOpts_df['Equity'] = FXOpts_df['Equity'].map(lambda x: x.split(' -')[0])
FXOpts_df['Equity'] = FXOpts_df['Equity'].map(lambda x: re.sub(r'[^A-Z^0-9^.]','', re.sub('0000', '', re.sub('SALE', '', re.sub('PURCHASE', '', re.sub('CREDIT', '', re.sub('DEBIT', '', x.upper())))))))

DOfwd_df = FXOpts_df[FXOpts_df['Nature'] == 'SETTLEMENT']
DOfwd_df = DOfwd_df[DOfwd_df['Equity'].map(lambda x: x.find('BRL')) == -1]
FXOpts_df.loc[DOfwd_df.index, 'Cash Amount'] = 0

FXOpts_df['Cash Amount Acc'] = FXOpts_df['Cash Amount'].values
FXOpts_df['Cash Amount'] *= FXOpts_df.index.map(lambda x: sEURFX_df.loc[FXOpts_df.loc[x, 'Trade Date'], FXOpts_df.loc[x, 'Currency'] + 'USD']).values
FXOpts_df['Cash Amount'] = FXOpts_df['Cash Amount'].map(lambda x: round(x, 2))
FXOpts_df['Taxable'] = FXOpts_df['Cash Amount']
FXOpts_df.index = np.arange(FXOpts_df.shape[0])
FXOpts_df['Expiry'] = FXOpts_df['Equity'].map(lambda x: datetime.strptime(x[-9:], '%d%b%Y').date())

pFXOpts_df = FXOpts_df[FXOpts_df['Nature'] == 'PREMIUM']
PremiumOUT_df = pd.DataFrame(0, index = year_list, columns = ['Outstanding Premium (' + acc_ccy + ')'])
PremiumAccOUT_df = pd.DataFrame(0, index = year_list, columns = pd.unique(pFXOpts_df['Currency']).tolist())
for c_year in year_list:
    last_dt = FX_df[FX_df.index < datetime(c_year + 1, 1, 1).date()].index[-1]
    SpFXOpts_df = pFXOpts_df[pFXOpts_df['Trade Date'].map(lambda x: x.year) == c_year]
    if (SpFXOpts_df.shape[0] > 0):
        SpFXOpts_df = SpFXOpts_df[SpFXOpts_df['Expiry'].map(lambda x: x.year) > c_year]
        if (SpFXOpts_df.shape[0] > 0):
            PremiumOUT_df.loc[c_year] = SpFXOpts_df['Cash Amount'].sum()       
            FXOPccyBD_df = SpFXOpts_df[['Currency', 'Cash Amount Acc']].groupby(['Currency']).sum()
            PremiumAccOUT_df.loc[c_year, FXOPccyBD_df.index] = FXOPccyBD_df['Cash Amount Acc']
for c_pccy in PremiumAccOUT_df.columns:
    PremiumAccOUT_df[c_pccy] = (PremiumAccOUT_df[c_pccy] - PremiumAccOUT_df[c_pccy].shift(1)).fillna(0) * PremiumAccOUT_df.index.map(lambda x: sEURFX_df.loc[last_dt, c_pccy + 'USD'])
PremiumAccOUT_s = PremiumAccOUT_df.sum(axis = 1)    
PremiumOUT_df['FX Opts YE PV (' + acc_ccy + ')'] = fxoMTM_l
PremiumOUT_df = (PremiumOUT_df - PremiumOUT_df.shift(1)).fillna(0)
PremiumOUT_s = PremiumOUT_df['Outstanding Premium (' + acc_ccy + ')']
RfxoMtM_s = PremiumOUT_s - PremiumAccOUT_s
PremiumOUT_df['Outstanding Premium (' + acc_ccy + ')'] = PremiumAccOUT_s
fxoMtM_s = PremiumOUT_df.sum(axis = 1)
fxoMtM_s.name = 'FXO MtM'

fxfMtM_s = pd.Series(np.array(fxMTM_l) - np.array(fxoMTM_l), index = year_list)
fxfMtM_s = (fxfMtM_s - fxfMtM_s.shift(1)).fillna(0)

fwd_df = fx_df[fx_df['Equity'].map(lambda x: x.find('CALL')) == -1]
fwd_df = fwd_df[fwd_df['Equity'].map(lambda x: x.find('NDF')) == -1]

fwd_df = complementFwds(fwd_df)
fwd_df['Equity'] = fwd_df['Equity'].map(lambda x: re.sub(r'[^A-Z^0-9^.]','', re.sub('FORWARD', '', re.sub('SPOT', '', re.sub('MATURITY', '', re.sub('0000', '', re.sub('SALE', '', re.sub('PURCHASE', '', x.upper()))))))))
fwd_df['Taxable'] = (fwd_df['Cash Amount'] * fwd_df.index.map(lambda x: sEURFX_df.loc[fwd_df.loc[x, 'Trade Date'], fwd_df.loc[x, 'Currency'] + 'USD']).values).map(lambda x: round(x, 2))

LJJEqtyTrades_df = Feqty_df[['Trade Date', 'Equity', 'Cash Amount Acc', 'Quantity']]
LJJEqtyTrades_df['Trade Px'] = round(LJJEqtyTrades_df['Cash Amount Acc'] / LJJEqtyTrades_df['Quantity'], 4)
LJJEqtyTrades_df = LJJEqtyTrades_df.drop(['Cash Amount Acc'], axis = 1)
LJJEqtyTrades_df.to_sql('all_ljj_trades', Fengine, index = True, if_exists = 'replace')

summaryEq_s = summarize(Feqty_df, 'A- Realized Equities', 'Trade Date')
summarySw_s = summarize(swap_df, 'A- Realized Swaps', 'Trade Date')
summaryFee_s = summarize(fee_df, 'C- Fees', 'Trade Date')
summaryIR_s = summarize(int_df, 'AA- Interest', 'Trade Date')
summaryDv_s = summarize(dvd_df, 'AA- Dvd', 'Trade Date')
summaryNDF_s = summarize(FXNDF_df, 'A- Realized NDF', 'Trade Date')
summaryFwd_s = summarize(fwd_df, 'A- Realized Fwd', 'Trade Date')
EqOptEU_s = summarize(opts_df, 'EqOpt', 'Trade Date')  
if (ExpAsUgo):
    summaryFXO_s = summarize(FXOpts_df[FXOpts_df['Expiry'] <= Pxs_df.index[-1]], 'A- Realized FXO', 'Expiry')
    EqOptEX_s = summarize(expO_df[expO_df['Expiry'] <= Pxs_df.index[-1]], 'EqOpt', 'Expiry')
else:    
    summaryFXO_s = summarize(FXOpts_df, 'A- Realized FXO', 'Expiry')
    EqOptEX_s = summarize(expO_df, 'EqOpt', 'Expiry')
if (optSTR):
    EqOptST_s = summarize(EqOpSTR_df, 'EqOptST', 'Date')
    EqOpt_s = pd.DataFrame(EqOptEX_s).join(EqOptEU_s, rsuffix = '_UN', how = 'outer').join(EqOptST_s).fillna(0).sum(axis = 1)
else:
    EqOpt_s = pd.DataFrame(EqOptEX_s).join(EqOptEU_s, rsuffix = '_UN', how = 'outer').fillna(0).sum(axis = 1)
EqOpt_s.name = 'A- Realized EqOpt'
summaryProc_s = summaryFee_s + summaryIR_s
summaryProc_s.name = 'AA- Proceeds'
    
if (bundleALL):    
    summaryCCY_s = pd.DataFrame(pd.Series(0, index = year_list)).join(summaryNDF_s).join(summaryFwd_s).join(summaryFXO_s).fillna(0).sum(axis = 1)
    summaryCCY_s.name = 'A- Realized Currencies'
    summaryAEq_s = pd.DataFrame(summaryEq_s).join(EqOpt_s, how = 'outer').fillna(0).sum(axis = 1)
    summaryAEq_s.name = 'A- Realized Equities'
    summaryALL_df = pd.DataFrame(summaryAEq_s).join(summarySw_s, how = 'outer').join(summaryCCY_s, how = 'outer').join(summaryProc_s, how = 'outer').fillna(0)
else:
    summaryALL_df = pd.DataFrame(summaryEq_s).join(EqOpt_s, how = 'outer').join(summarySw_s, how = 'outer').join(summaryNDF_s, how = 'outer').join(summaryFwd_s, how = 'outer').join(summaryFXO_s, how = 'outer').join(summaryProc_s, how = 'outer').fillna(0)

summaryALL_df['AT0- Realized ALL Prod'] = summaryALL_df.sum(axis = 1)
summaryALL_df = summaryALL_df.join(summaryDv_s).fillna(0)

summaryALL_df['D- Net K Moves'] = YScash_s
summaryALL_df['DT- PC P&L ex-LA'] = LJJAUM_Delta_s - YScash_s
summaryALL_df['B- Unrealized Equities'] = unEqty_s
summaryALL_df['B- Unrealized Eq Opts'] = unEqOp_s
summaryALL_df['B- Unrealized Swaps'] = swapMtM_s
summaryALL_df['B- Unrealized FXO'] = fxoMtM_s
if (realizeFX):  
    summaryALL_df['B- Unrealized Equities'] -= RunEqty_s
    summaryALL_df['A- FX realized Equities'] = RunEqty_s
    summaryALL_df['B- Unrealized Eq Opts'] -= RunEqOp_s
    summaryALL_df['A- FX realized Eq Opts'] = RunEqOp_s    
    summaryALL_df['B- Unrealized Swaps'] -= RswapPV_df
    summaryALL_df['FX realized Swaps'] = RswapPV_df
    summaryALL_df['B- Unrealized FXO'] -= RfxoMtM_s
    summaryALL_df['A- FX realized FXO'] = RfxoMtM_s    
    summaryALL_df['A- FX realized Forwards'] = fxfMtM_s
else:
    summaryALL_df['B- Unrealized Equities'] -= RunEqty_s
    summaryALL_df['B- Unrealized Equities FX'] = RunEqty_s    
    summaryALL_df['B- Unrealized FX FWD'] = fxfMtM_s
if (dropLAST):
    if (Pxs_df.index[-1].year in summaryALL_df.index):
        summaryALL_df = summaryALL_df.drop(Pxs_df.index[-1].year)

summaryALL_df['BT- Unrealized ALL Prod'] = round(summaryALL_df.T[summaryALL_df.columns.map(lambda x: x.find('Unrealized ')) != -1].sum(), 2)
summaryALL_df['BT- Unrealized (' + acc_ccy + ')'] = summaryALL_df['DT- PC P&L ex-LA'] - (summaryALL_df['AT0- Realized ALL Prod'] + summaryALL_df['AA- Dvd'] + summaryALL_df['AA- Proceeds'])
summaryALL_df['AT1- Realized ex-LA/Exp (' + acc_ccy + ')'] = summaryALL_df['AT0- Realized ALL Prod'] + summaryALL_df['AA- Dvd'] + summaryALL_df['AA- Proceeds']
if (realizeFX):
    summaryALL_df['AT1- Realized ex-LA/Exp (' + acc_ccy + ')'] += (RunEqty_s + RunEqOp_s + RswapPV_df + fxfMtM_s + RfxoMtM_s)
    summaryALL_df['BT- Unrealized (' + acc_ccy + ')'] -= (RunEqty_s + RunEqOp_s + RswapPV_df + fxfMtM_s + RfxoMtM_s)

summaryALL_df['AA- Expenses (' + acc_ccy + ')'] = expenses_s * (-1)  
summaryALL_df['AA- LA Int'] = LAInt_l[: summaryALL_df.shape[0]]
summaryALL_df['AT2- Realized (' + acc_ccy + ')'] = summaryALL_df['AT1- Realized ex-LA/Exp (' + acc_ccy + ')'] + summaryALL_df['AA- LA Int'] + summaryALL_df['AA- Expenses (' + acc_ccy + ')']
summaryALL_df['DT- Ttl Accounting (' + acc_ccy + ')'] = summaryALL_df['AT2- Realized (' + acc_ccy + ')'] + summaryALL_df['BT- Unrealized (' + acc_ccy + ')']
summaryALL_df['F- Taxes (' + acc_ccy + ')'] = taxes_l[: summaryALL_df.shape[0]]
summaryALL_df['F- Taxes (' + acc_ccy + ')'] *= summaryALL_df.index.map(lambda x: sEURFX_df.loc[FX_df[FX_df.index < datetime(x + 1, 1, 1).date()].index[-1], acc_ccy + 'USD'])
summaryALL_df['F- After Tax Ttl Acct (' + acc_ccy + ')'] = summaryALL_df['DT- Ttl Accounting (' + acc_ccy + ')'] - summaryALL_df['F- Taxes (' + acc_ccy + ')']
summaryALL_df['F- Dividends (' + acc_ccy + ')'] = dividends_l[: summaryALL_df.shape[0]]
summaryALL_df['F- Dividends (' + acc_ccy + ')'] *= summaryALL_df.index.map(lambda x: sEURFX_df.loc[FX_df[FX_df.index < datetime(x + 1, 1, 1).date()].index[-1], acc_ccy + 'USD'])
summaryALL_df['F- Refunds (' + acc_ccy + ')'] = summaryALL_df['F- Dividends (' + acc_ccy + ')'] * .3
summaryALL_df['F- Retained Earnings (' + acc_ccy + ')'] = summaryALL_df['F- After Tax Ttl Acct (' + acc_ccy + ')'].expanding().sum() - summaryALL_df['F- Dividends (' + acc_ccy + ')'].expanding().sum()
summaryALL_df['F- Dividends Payable (' + acc_ccy + ')'] = summaryALL_df['F- Retained Earnings (' + acc_ccy + ')'].map(lambda x: max(0, x))
summaryALL_df['F- Effective Tax Rate (%)'] = (100 * summaryALL_df['F- Taxes (' + acc_ccy + ')'].expanding().sum() / summaryALL_df['DT- PC P&L ex-LA'].expanding().sum()).map(lambda x: round(x, 2))
summaryALL_df['F- Refund Shortfall (' + acc_ccy + ')'] = summaryALL_df['F- Taxes (' + acc_ccy + ')'].expanding().sum() * (6 / 7) - summaryALL_df['F- Refunds (' + acc_ccy + ')'].expanding().sum()
summaryALL_df['D- AUM (' + acc_ccy + ')'] = LJJAUM_df.loc[summaryALL_df.index, 'AUM ' + acc_ccy]
summaryALL_df.loc['ALL'] = summaryALL_df.sum()

summaryALL_df.loc['ALL', 'D- AUM (' + acc_ccy + ')'] = np.nan
summaryALL_df.loc['ALL', 'F- Dividends Payable (' + acc_ccy + ')'] = np.nan
summaryALL_df.loc['ALL', 'F- Retained Earnings (' + acc_ccy + ')'] = np.nan
summaryALL_df.loc['ALL', 'F- Effective Tax Rate (%)'] = np.nan
summaryALL_df.loc['ALL', 'F- Refund Shortfall (' + acc_ccy + ')'] = np.nan

EsummaryALL_df = format_col(summaryALL_df.copy())
if (accountingEUR):
    EsummaryALL_df.index.name = 'EUR'
else:
    EsummaryALL_df.index.name = 'USD'
    
Port_df = pd.DataFrame(Feqty_df[['Equity', 'Quantity']].groupby(Feqty_df['Equity'])['Quantity'].sum().replace({0: np.nan}).dropna().astype(int).sort_values(ascending = False))
Port_df['Avg Px'] = Port_df.index.map(lambda x: round(Feqty_df[Feqty_df['Equity'] == x][accMethod].iloc[-1] / 
                                                      sEURFX_df.loc[Feqty_df[Feqty_df['Equity'] == x]['Trade Date'].iloc[-1], 
                                                                    Feqty_df[Feqty_df['Equity'] == x]['Currency'].iloc[-1] + 'USD'], 2))

stockO_l = pd.unique([t for t in Port_df.index.tolist() if t in all_tickers.index.tolist()]).tolist()
EqPx_l = pd.Series(stockO_l).map(lambda x: all_tickers.drop_duplicates().loc[x, 'tickers'])
EqPx_l = [t for t in pd.Series(EqPx_l).replace({'SHOP US': 'SHOP CN'}).map(lambda x: re.sub(' GY', ' GR', x)).tolist() if t in Pxs_df.columns.tolist()]
pxs_s = pd.Series(pd.Series(EqPx_l).replace({'SHOP US': 'SHOP CN'}).map(lambda x: re.sub(' GY', ' GR', x)).map(lambda x: Pxs_df.iloc[-1][x]).values, index = 
                  pd.Series(EqPx_l).map(lambda x: x.split(' ')[0]), name = 'RT Pxs').map(lambda x: round(x, 2))
Port_df = Port_df.join(pxs_s).dropna()
Port_df['Unrealized (Acc)'] = round(Port_df['Quantity'] * (Port_df['RT Pxs'] - Port_df['Avg Px']), 0)
Port_df = Port_df.sort_values(by = ['Unrealized (Acc)'], ascending = False)
ttl_unr = int(Port_df['Unrealized (Acc)'].sum())
Port_df['Unrealized (Acc)'] = Port_df['Unrealized (Acc)'].map(lambda x: '{:,.0f}'.format(x))

print('')
print('TOTAL Eqty Unr (Acc): ' + '{:,.0f}'.format(ttl_unr))    
print('')

if (accMethod == 'FIFOPx'):
    pnlThresh = 10000
    neg_init = False
    for i, c_ticker in enumerate(Port_df.index):
        x = Feqty_df[Feqty_df['Equity'] == c_ticker]
        b = x[x['Quantity'] > 0][['Quantity', 'Trade Px', 'Cash Amount']]
        b['CumB'] = b['Quantity'].expanding().sum()
        q_s = -x[x['Quantity'] < 0]['Quantity'].sum()
        y = b[b['CumB'] > q_s]
        if (y.shape[0] > 0):
            cP_df = pd.DataFrame(np.nan, index = [b.loc[: y.index[0]]['Quantity'].sum() - q_s] + y[1:]['Quantity'].tolist(), columns = ['Ticker', 'Trade Px', 'RT Px', 'P&L (' + acc_ccy + ')'])
            cP_df['Ticker'] = c_ticker
            cP_df['Trade Px'] = y['Trade Px'].map(lambda x: round(x, 2)).values
            cP_df['RT Px'] = round(Port_df.loc[c_ticker, 'RT Pxs'] * sEURFX_df.loc[sEURFX_df.index[-1], x.loc[x.index[0], 'Currency'] + 'USD'], 2)
            cP_df['P&L (' + acc_ccy + ')'] = round((cP_df['RT Px'] - cP_df['Trade Px']) * cP_df.index, 1)
            if (i == 0):
                allP_df = cP_df.copy()
            else:
                allP_df = pd.concat([allP_df, cP_df])
            if (cP_df.iloc[0]['P&L (' + acc_ccy + ')'] < 0):
                if (neg_init):
                    allNP_df = pd.concat([allNP_df, cP_df])
                else:
                    allNP_df = cP_df.copy()
                    neg_init = True

    groupAllNP_df = pd.DataFrame(allNP_df.groupby(allNP_df['Ticker']).sum().sort_values(by = ['P&L (' + acc_ccy + ')'])['P&L (' + acc_ccy + ')'])
    groupAllAP_df = pd.DataFrame(allP_df.groupby(allP_df['Ticker']).sum().sort_values(by = ['P&L (' + acc_ccy + ')'])['P&L (' + acc_ccy + ')'])
    groupAllNP_df['Position'] = Port_df.loc[groupAllNP_df.index]['Quantity']
    groupAllAP_df['Position'] = Port_df.loc[groupAllAP_df.index]['Quantity']

    print('###################### TAX OPTIMIZATION ######################')
    print('')
    print('Optimize:')
    print(groupAllNP_df[groupAllNP_df['P&L (' + acc_ccy + ')'] < -pnlThresh])  
    print('TTL: ' + str(round(groupAllNP_df[groupAllNP_df['P&L (' + acc_ccy + ')'] < -pnlThresh]['P&L (' + acc_ccy + ')'].sum(), 2)))
    print('')
    print('Per Trade Breakdown:')
    print(allNP_df[allNP_df['Ticker'].map(lambda x: x in groupAllNP_df[groupAllNP_df['P&L (' + acc_ccy + ')'] < -pnlThresh].index)][:60])
    print('')
    print('')
    print('Careful:')
    print(groupAllAP_df[groupAllAP_df['P&L (' + acc_ccy + ')'] > pnlThresh].sort_values(by = ['P&L (' + acc_ccy + ')'], ascending = False)[:60])
    print('')
else:
    print('********CURRENT PORTFOLIO********')
    print(Port_df[:60])
    if (Port_df.shape[0] > 60):
        print(Port_df[-(Port_df.shape[0] - 60):])
print('')
print('Check: ' + '{:,.0f}'.format(Feqty_df[['Equity', 'Quantity']].groupby(Feqty_df['Equity'])['Quantity'].sum().replace({0: np.nan}).dropna().astype(int).sort_values(ascending = False).sum()))
print('')
    
#print('')
#print('Tax Base 2024 (' + acc_ccy + '): ' + '{:,.0f}'.format(summaryALL_df.T[[2022, 2023, 2024]].loc['AT2- Realized (' + acc_ccy + ')'].sum()))
#print('')    

summaryYstrALL_df = summaryALL_df.T[[2022, 2023, 2024]]
summaryYstrALL_df['Ttl (since 2022)'] = summaryYstrALL_df.sum(axis = 1)
print('CURRENT SITUATION, 2024:')
print(format_col(summaryYstrALL_df.drop(['AT2- Realized (' + acc_ccy + ')']).copy()))

eurusd_s = pd.Series(pd.Series(year_list).map(lambda x: round(FX_df.loc[FX_df[FX_df.index.map(lambda y: y.year) == x].index[-1], 'EURUSD'], 4)).values, index = pd.Index(year_list, name = 'EURUSD'))
eurusd_s.loc[0] = round(FX_df.loc[datetime(2020, 12, 17).date(), 'EURUSD'], 4)
eurusd_s = eurusd_s.sort_index()
print('')
print(eurusd_s)

print('')
print('REALIZED TAX MAP:')    
EsummaryALL_df = EsummaryALL_df.replace({'nan': ''}).T.sort_index()
EsummaryALL_df.index = EsummaryALL_df.index.map(lambda x: x.split('- ')[1])

acct2025_c = (summaryALL_df['DT- Ttl Accounting (' + acc_ccy + ')'].drop(['ALL']).loc[2025] + 
          min((EsummaryALL_df.drop(['ALL'], axis = 1).loc['Realized (EUR)', 2025:].map(lambda x: re.sub(',', '', x)).astype(float).sum() - 412483) * (-.35), 0))
acct_c = (summaryALL_df['DT- Ttl Accounting (' + acc_ccy + ')'].drop(['ALL']).loc[2025:].sum() + 
          min((EsummaryALL_df.drop(['ALL'], axis = 1).loc['Realized (EUR)', 2025:].map(lambda x: re.sub(',', '', x)).astype(float).sum() - 412483) * (-.35), 0))
opt2025_dvd = min((summaryALL_df['F- Refund Shortfall (' + acc_ccy + ')'].loc[2025] / .3), acct2025_c) 

print('Cumulative REALIZED 2022-2025 (EUR): 'f"{EsummaryALL_df.loc['Realized (EUR)', [2022, 2023, 2024, 2025]].map(lambda x: re.sub(',', '', x)).astype(float).sum():,.2f}")
print('Cumulative OFFICIAL 2022-2025 (EUR): 'f"{float(re.sub(',', '', EsummaryALL_df.loc['Realized (EUR)', 2025])) - 412483:,.2f}")
print('Optimal Dividends 2025 (EUR): 'f"{opt2025_dvd:,.2f}")
print('Remaining Reserves ' + str(summaryALL_df['F- Refund Shortfall (' + acc_ccy + ')'].drop(['ALL']).index[-1]) + ' (EUR): 'f"{(acct_c - opt2025_dvd):,.2f}")
print('')
EsummaryALL_df

