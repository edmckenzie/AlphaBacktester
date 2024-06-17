import pytz
import yfinance
import requests
import threading
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from quantlab.utils import timeme
from quantlab.utils import save_pickle, load_pickle
from quantlab.utils import Portfolio

def get_sp500_tickers():
    res = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = BeautifulSoup(res.content,'html')
    table = soup.find_all('table')[0] 
    df = pd.read_html(str(table))
    tickers = list(df[0].Symbol)
    return tickers

def get_history(ticker, period_start, period_end, granularity="1d", tries=0):
    try:
        df = yfinance.Ticker(ticker).history(
            start=period_start,
            end=period_end,
            interval=granularity,
            auto_adjust=True
        ).reset_index()
    except Exception as err:
        if tries < 5:
            return get_history(ticker, period_start, period_end, granularity, tries+1)
        return pd.DataFrame()
    
    df = df.rename(columns={
        "Date":"datetime",
        "Open":"open",
        "High":"high",
        "Low":"low",
        "Close":"close",
        "Volume":"volume"
    })
    if df.empty:
        return pd.DataFrame()
    df.datetime = pd.DatetimeIndex(df.datetime.dt.date).tz_localize(pytz.utc)
    df = df.drop(columns=["Dividends", "Stock Splits"])
    df = df.set_index("datetime",drop=True)
    return df

def get_histories(tickers, period_starts,period_ends, granularity="1d"):
    dfs = [None]*len(tickers)
    def _helper(i):
        print(tickers[i])
        df = get_history(
            tickers[i],
            period_starts[i], 
            period_ends[i], 
            granularity=granularity
        )
        dfs[i] = df
    threads = [threading.Thread(target=_helper,args=(i,)) for i in range(len(tickers))]
    [thread.start() for thread in threads]
    [thread.join() for thread in threads]
    tickers = [tickers[i] for i in range(len(tickers)) if not dfs[i].empty]
    dfs = [df for df in dfs if not df.empty]
    return tickers, dfs

def get_ticker_dfs(start,end):
    from quantlab.utils import load_pickle,save_pickle
    try:
        tickers, ticker_dfs = load_pickle("dataset.obj")
    except Exception as err:
        tickers = get_sp500_tickers()
        starts=[start]*len(tickers)
        ends=[end]*len(tickers)
        tickers,dfs = get_histories(tickers,starts,ends,granularity="1d")
        ticker_dfs = {ticker:df for ticker,df in zip(tickers,dfs)}
        save_pickle("dataset.obj", (tickers,ticker_dfs))
    return tickers, ticker_dfs 

import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from quantlab.gene import GeneticAlpha
from quantlab.gene import Gene
def main():
    period_start = datetime(2000,1,1, tzinfo=pytz.utc)
    period_end = datetime(2023,1,1, tzinfo=pytz.utc)
    tickers, ticker_dfs = get_ticker_dfs(start=period_start,end=period_end)
    tickers = tickers[:50]
    
    
    #g1=Gene.str_to_gene("ls_25/75(neg(mean_12(cszscre(div(mult(volume,minus(minus(close,low),minus(high,close))),minus(high,low))))))")
    # g2=Gene.str_to_gene("neg(mean_12(minus(const_1,div(open,close))))")
    g3=Gene.str_to_gene("plus(ite(gt(mean_10(close),mean_50(close)),const_1,const_0),ite(gt(mean_20(close),mean_100(close)),const_1,const_0),ite(gt(mean_50(close),mean_200(close)),const_1,const_0))")
    
    # g1=Gene.str_to_gene("div(epsDifference,std_12(epsDifference))")
    alpha1=GeneticAlpha(insts=tickers,dfs=ticker_dfs,start=period_start,end=period_end,genome=g3)
    df1=alpha1.run_simulation()
    alpha1.get_perf_stats(plot=True)
    print(df1)
    
    #post-earnings-accouncement drift >> positive earings surprise, then go long
    '''
   
    '''
       
if __name__ == "__main__":
    main()
