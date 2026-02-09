#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#INTRADAY STOCK GRAPHS - YF

import pandas as pd
import os
import csv
import yfinance as yf
import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import warnings
from datetime import date
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

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

def on_selection(value):
    global choice_lag
    choice_lag = value  # store the user's choice
    root.destroy()  # close window
     
        
dbase = "visiblealpha_laptop"
Fdbase = "factormodel_db"
cnxn_string = ("postgresql+psycopg2://{username}:{pswd}""@{host}:{port}/{database}")
engine = create_engine(cnxn_string.format(username = "postgres", pswd = "akf7a7j5", host = "localhost", 
                                          port = 5432, database = dbase))
Fengine = create_engine(cnxn_string.format(username = "postgres", pswd = "akf7a7j5", host = "localhost", 
                                           port = 5432, database = Fdbase))

warnings.filterwarnings('ignore')
        
today = date.today()
window = 42
band_m = 2
max_limit = 1.05
min_limit = .95

query_earnings = "SELECT * FROM ed_relation"
ed_df = pd.read_sql_query(query_earnings, engine)
ed_df = Set_DF(ed_df)
ed_df = DD_Index(ed_df)
vayf_df = openF_df('va_yf')

fibonacci_seq = [0.00, 23.60, 38.20, 50.00, 61.80, 78.60, 100.00]
fibonacci_list = []
fig1 = plt.figure(figsize = (14, 10))
ax1 = fig1.add_subplot(1, 1, 1)

choiceAlgo_ticker = input('Enter ALGO Ticker: ').upper()
try:
    choice_ticker = vayf_df.loc[choiceAlgo_ticker, 'YF Ticker']
except:
    choice_ticker = choiceAlgo_ticker.split(' ')[0]
choiceNum_ticker = input('Enter NUMERAIRE Ticker, empty for USD MMAcct: ').upper()
try:
    choice_Nticker = vayf_df.loc[choiceNum_ticker.split(' ')[0], 'YF Ticker']
except:
    print('USD MMAcct Numeraire')
    choiceNum_ticker = ''

day_lag = int(input('Enter Days: '))  
start_date = today - timedelta(day_lag)

choice_list = ['5', '15', '30', '45', '60']
root = tk.Tk()
tkvar = tk.StringVar(root)
popupMenu = tk.OptionMenu(root, tkvar, *choice_list, command = on_selection)
tk.Label(root, text = "Choose a lag (minutes)").grid(row = 0, column = 0)
popupMenu.grid(row = 1, column = 0)
root.mainloop()

ticker_data = yf.Ticker(choice_ticker)
ticker_data_df = ticker_data.history(start = start_date - timedelta(days = 1), end = today + timedelta(days = 1), interval = choice_lag + "m")
if (choiceNum_ticker != ''):
    num_data = yf.Ticker(choice_Nticker)
    num_data_df = num_data.history(start = start_date - timedelta(days = 1), end = today + timedelta(days = 1), interval = choice_lag + "m")
    ticker_data_df.index = ticker_data_df.index.map(lambda x: datetime(x.year, x.month, x.day, x.hour))
    num_data_df.index = num_data_df.index.map(lambda x: datetime(x.year, x.month, x.day, x.hour))
    commonDates_l = [x for x in num_data_df.index if x in ticker_data_df.index]
    ticker_data_df = ticker_data_df.loc[commonDates_l]
    num_data_df = num_data_df.loc[commonDates_l]    
    num_df = pd.DataFrame(num_data_df['Close'].values, index = np.arange(len(num_data_df.index)), columns = [choice_Nticker])
    choice_ticker = choice_ticker + ' in ' + choiceNum_ticker
    
ticker_df = pd.DataFrame(ticker_data_df['Close'].values, index = np.arange(len(ticker_data_df.index)), columns = [choice_ticker])    
if (choiceNum_ticker != ''):
    ticker_df /= num_df.values
    ticker_df /= ticker_df.iloc[0]
PLOTticker_df = pd.DataFrame(ticker_df.values, index = ticker_data_df.index, columns = [choice_ticker])


p_degree = input('Enter Pol Degree (empty for 10): ')
if (p_degree != ''):
    p_degree = int(p_degree)
else:
    p_degree = 10
first_deriv = 1
second_deriv = 2
lin_reg = LinearRegression()
x_Values = ticker_df.index.values.reshape(-1, 1)
y_Values = np.array(ticker_df)
poly_features = PolynomialFeatures(degree = p_degree, include_bias = False)
poly_X = poly_features.fit_transform(x_Values)
LR_Fit = lin_reg.fit(poly_X, y_Values)
Predictions_df = pd.DataFrame(LR_Fit.predict(poly_X), index = ticker_df.index, columns = ['Forecast'])

upper_b = ((1 + band_m * ticker_df.pct_change().rolling(window).std() * np.sqrt(window)).shift(window) * ticker_df.shift(window)).dropna()
lower_b = ((1 - band_m * ticker_df.pct_change().rolling(window).std() * np.sqrt(window)).shift(window) * ticker_df.shift(window)).dropna()

mav04 = ticker_df.rolling(4, min_periods = 1).mean()
mav08 = ticker_df.rolling(8, min_periods = 1).mean()
mav16 = ticker_df.rolling(16, min_periods = 1).mean()
    
ax1.plot(ticker_df, c = 'k', label = choice_ticker + ' ' + str(round(ticker_df.iloc[-1][choice_ticker], 2)))
ax1.plot(Predictions_df, c = 'b', alpha = 0.15, label = 'Polynomial Fit ' + str(round(Predictions_df.iloc[-1][0], 2)))
ax1.plot(mav04, c = 'g', alpha = 0.6, linestyle = 'dashed', label = 'MAV04' + ' ' + str(round(mav04.iloc[-1][choice_ticker], 2)))
ax1.plot(mav08, c = 'b', alpha = 0.6, linestyle = 'dashed', label = 'MAV08' + ' ' + str(round(mav08.iloc[-1][choice_ticker], 2)))
ax1.plot(mav16, c = 'r', alpha = 0.6, linestyle = 'dashed', label = 'MAV16' + ' ' + str(round(mav16.iloc[-1][choice_ticker], 2)))
if (day_lag > 60):
    mav50 = ticker_df[::(ticker_df.shape[0] // day_lag) + 1].rolling(50, min_periods = 1).mean()
    ax1.plot(mav50, c = 'k', alpha = 0.6, linestyle = 'dashed', label = 'MAV50' + ' ' + str(round(mav50.iloc[-1][choice_ticker], 2)))

ax1.plot(upper_b.clip(ticker_df.min()[0] * min_limit, ticker_df.max()[0] * max_limit, axis = 1), c = 'g', linestyle = 'dotted', alpha = 0.5, label = 'Upper')
ax1.plot(lower_b.clip(ticker_df.min()[0] * min_limit, ticker_df.max()[0] * max_limit, axis = 1), c = 'r', linestyle = 'dotted', alpha = 0.5, label = 'Lower')

local_max = ticker_df[choice_ticker].max()
local_min = ticker_df[choice_ticker].min()
hl_gap = (local_max - local_min)
for f_level in fibonacci_seq:
    f_array = local_min + np.ones(len(ticker_df.index)) * (f_level / 100) * hl_gap
    f_series = pd.Series(f_array, index = ticker_df.index)
    fibonacci_list.append(round(f_series.iloc[0], 3))
    ax1.plot(f_series, alpha = 0.5)



poly_features_1d = PolynomialFeatures(degree = (p_degree - first_deriv), include_bias = False)
poly_dX = poly_features_1d.fit_transform(x_Values)
n, m = poly_dX.shape
x0 = np.ones((n, 1))
poly_dX_base = np.c_[x0, poly_dX]
poly_deriv = np.array(list(np.poly1d(np.array(list(LR_Fit.coef_[0][::-1]) + [1])).deriv())[::-1])

poly_features_2d = PolynomialFeatures(degree = (p_degree - second_deriv), include_bias = False)
poly_2dX = poly_features_2d.fit_transform(x_Values)
n2, m2 = poly_2dX.shape
x0_2 = np.ones((n2, 1))
poly_2dX_base = np.c_[x0_2, poly_2dX]
poly_2deriv = np.array(list(np.poly1d(np.array(list(poly_deriv)[::-1])).deriv())[::-1])


pd.DataFrame(poly_X.dot(LR_Fit.coef_[0]) + LR_Fit.intercept_).plot(figsize = (14, 10), c = 'g').twinx()
plt.plot(poly_dX_base.dot(poly_deriv), c = 'k')
plt.plot(20 * poly_2dX_base.dot(poly_2deriv), c = 'r')


deriv01_df = pd.DataFrame(poly_dX_base.dot(poly_deriv), columns = ['Deriv 01'])
deriv02_df = pd.DataFrame(poly_2dX_base.dot(poly_2deriv), columns = ['Deriv 02'])

d01_df = deriv01_df.copy()
d01_df['Abs'] = abs(d01_df['Deriv 01'])
d01_df['Original Index'] = d01_df.index

prev = 0
Deriv0_pts_df = pd.DataFrame(np.nan, index = [0], columns = ['Local'])
for idx in d01_df.index[1:]:
    if (d01_df.iloc[idx]['Deriv 01'] > 0) & (d01_df.iloc[prev]['Deriv 01'] < 0):
        Deriv0_pts_df.loc[idx] = 'Min'
    if (d01_df.iloc[idx]['Deriv 01'] < 0) & (d01_df.iloc[prev]['Deriv 01'] > 0):
        Deriv0_pts_df.loc[idx] = 'Max'
    prev = idx
Deriv0_pts_df.drop([0], inplace = True)

if (len(Deriv0_pts_df.index) > 1):
    p = Deriv0_pts_df.index[0]
    lows_list = []
    highs_list = []
    sign = 1
    if (Deriv0_pts_df.loc[p][0] == 'Max'):
        sign = -1
        lows_list.append((ticker_df.loc[:p][choice_ticker] * sign).max() * sign)
    else:
        highs_list.append((ticker_df.loc[:p][choice_ticker] * sign).max() * sign)
    p = 0
    for i in Deriv0_pts_df.index[:-1]:
        n = Deriv0_pts_df.index[Deriv0_pts_df.index.values.tolist().index(i) + 1]
        sign = 1
        if (Deriv0_pts_df.loc[i][0] == 'Min'):
            sign = -1
            lows_list.append((ticker_df.loc[p: n][choice_ticker] * sign).max() * sign)
        else:
            highs_list.append((ticker_df.loc[p: n][choice_ticker] * sign).max() * sign)
        p = i  
    sign = 1
    p = Deriv0_pts_df.index[-2]
    if (Deriv0_pts_df.iloc[-1][-1] == 'Max'):
        highs_list.append((ticker_df.loc[p:][choice_ticker] * sign).max() * sign)
    else:
        sign = -1
        lows_list.append((ticker_df.loc[p:][choice_ticker] * sign).max() * sign)
    lows_df = pd.DataFrame(lows_list, index = np.arange(len(lows_list)), columns = ['Levels'])
    highs_df = pd.DataFrame(highs_list, index = np.arange(len(highs_list)), columns = ['Levels'])
    lows_df['Positions'] = np.nan
    highs_df['Positions'] = np.nan
    for idx in lows_df.index:
        lows_df.loc[idx, 'Positions'] = ticker_df[ticker_df[choice_ticker] == lows_list[idx]].index[0]
    for idx in highs_df.index:
        highs_df.loc[idx, 'Positions'] = ticker_df[ticker_df[choice_ticker] == highs_list[idx]].index[0]    

    if (len(lows_df.index) > 1):  
        for j in np.arange(lows_df.shape[0] - 1):
            base_x = lows_df.loc[j, 'Positions']
            base_y = lows_df.loc[j, 'Levels']
            for i, curr in enumerate(lows_df.index[(j + 1):]):
                lx_values = np.array([base_x, lows_df.loc[curr, 'Positions']]).reshape(-1, 1)
                ly_values = np.array([base_y, lows_df.loc[curr, 'Levels']])
                Low_Fit = lin_reg.fit(lx_values, ly_values)
                lx_values = np.array([ticker_df.index[0], base_x, lows_df.loc[curr, 'Positions'], ticker_df.index[-1]]).reshape(-1, 1)
                Low_df = pd.DataFrame(Low_Fit.predict(lx_values), index = [ticker_df.index[0], base_x, lows_df.loc[curr, 'Positions'], ticker_df.index[-1]], columns = ['Low'])
                ax1.plot(Low_df.clip(ticker_df.min()[0] * min_limit, ticker_df.max()[0] * max_limit, axis = 1), c = 'g', alpha = 0.35, label = 'Support ' + str(round(Low_df.iloc[-1][0], 3)))

    if (len(highs_df.index) > 1):  
        for j in np.arange(highs_df.shape[0] - 1):
            base_x = highs_df.loc[j, 'Positions']
            base_y = highs_df.loc[j, 'Levels']
            for i, curr in enumerate(highs_df.index[(j + 1):]):
                lx_values = np.array([base_x, highs_df.loc[curr, 'Positions']]).reshape(-1, 1)
                ly_values = np.array([base_y, highs_df.loc[curr, 'Levels']])
                High_Fit = lin_reg.fit(lx_values, ly_values)
                lx_values = np.array([ticker_df.index[0], base_x, highs_df.loc[curr, 'Positions'], ticker_df.index[-1]]).reshape(-1, 1)
                High_df = pd.DataFrame(High_Fit.predict(lx_values), index = [ticker_df.index[0], base_x, highs_df.loc[curr, 'Positions'], ticker_df.index[-1]], columns = ['High'])
                ax1.plot(High_df.clip(ticker_df.min()[0] * min_limit, ticker_df.max()[0] * max_limit, axis = 1), c = 'r', alpha = 0.35, label = 'Resistance ' + str(round(High_df.iloc[-1][0], 3)))                

ax1.text(ticker_df[ticker_df[choice_ticker] == ticker_df[choice_ticker].max()].index[0], ticker_df[choice_ticker].max(), str(round(100 * (ticker_df[choice_ticker].max() / ticker_df.iloc[0][0] - 1), 2)))                
ax1.text(ticker_df.index[-1], ticker_df.iloc[-1][0], str(round(100 * (ticker_df.iloc[-1][0] / ticker_df.max().iloc[0] - 1), 2)))                
ax1.set_xticks(ticker_df.index[:: int(np.floor(ticker_df.shape[0] / 10))])
ax1.set_xticklabels(PLOTticker_df.index.map(lambda x: x.date())[:: int(np.floor(PLOTticker_df.shape[0] / 10))], rotation = 45)

if (choiceAlgo_ticker in ed_df.columns):
    yr_s = pd.Series(ed_df[choiceAlgo_ticker].index.map(lambda x: int(x.split('-')[1]) + int(x[0]) // 4), 
                     index = ed_df[choiceAlgo_ticker].index)
    eDt_df = pd.DataFrame(ed_df[choiceAlgo_ticker]).join(yr_s)
    eDt_df = eDt_df.replace({'': np.nan}).dropna()
    eDt_l = eDt_df.index.map(lambda x: datetime(int(eDt_df.loc[x, 'index']), int(eDt_df.loc[x, choiceAlgo_ticker][:2]), 
                                                int(eDt_df.loc[x, choiceAlgo_ticker][-2:])).date()).tolist()
    eDt_s = pd.Series(eDt_l)
    SeDt_s = eDt_s[eDt_s >= PLOTticker_df.index[0].date()]
    print('Last ' + str(SeDt_s.shape[0]) + ' Earnings Dates:')
    print(SeDt_s)
    print('')
    am_ed = input('Ammend Earnings Dates (empty or Y/y)? ').upper()
    if (am_ed == 'Y'):
        for e, c_ed in enumerate(SeDt_s[::-1].values):
            e += 1
            c_ED_str = str(input('Enter ' + str(e) + ' ED back (saved ' + c_ed.strftime('%d%b%y') + ') mmdd or empty to keep: '))
            save_EDdf = False
            if (c_ED_str != ''):
                try:
                    c_ED_int = int(c_ED_str)
                    ed_df.loc[ed_df.index[-e], choiceAlgo_ticker] = c_ED_str
                    SeDt_s.loc[SeDt_s.index[-e]] = datetime(int(ed_df.index[-e].split('-')[1]) + int(ed_df.index[-e][0]) // 4, 
                                                            int(c_ED_str[:2]), int(c_ED_str[-2:])).date()  
                    save_EDdf = True
                except:
                    print(c_ED_str + ' not saved')
                if (save_EDdf):
                    ed_df.to_sql('ed_relation', engine, index = True, if_exists = 'replace')
        
    for e_dt in SeDt_s.tolist():
        try:
            idx = PLOTticker_df.index.map(lambda x: x.date()).tolist().index(e_dt)
            ax1.axvline(x = idx, c = '0.25', label = e_dt.strftime('%d%b%y'), linestyle = 'dotted')
        except:
            pass

ax1.set_yticks(fibonacci_list)
ax1.set_yticklabels(fibonacci_list)
#ax1.set_ylim([lower_b.min()[0] * 0.9, upper_b.max()[0] * 1.1])

ax1.set_title(choice_ticker + ' - ' + str(day_lag) + ' days, ' + str(choice_lag) + ' minutes')
ax1.legend(loc = 'best')

#QQQ 90/60/12
#QQQ 200/60/10
#APP 250/60/15
#NVDA 500/60/20
#TSLA 325/60/8
#QQQXAU 725/60/20 ST: 2025-10-13 (-32.56 DD)
#QQQXAU 350/60/15 ST: 2025-10-13 

