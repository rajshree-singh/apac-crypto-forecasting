import pandas as pd, numpy as np, os
D='apdata'; O='panel'; os.makedirs(O,exist_ok=True)
U='/mnt/user-data/uploads/crypto/cryptodataset/'
END='2024-02-14'
def usd(f):
    # Yahoo Finance (CoinMarketCap-sourced) daily OHLCV, UTC calendar days. The ETH and LTC files of the
    # original Mendeley dataset are identical to this source; its BTC file is a 00:00-UTC snapshot series
    # from a different provider (labels shifted by one day), so BTC is taken from the same source as ETH/LTC.
    d=pd.read_csv(D+'/'+f); d['Date']=pd.to_datetime(d.Date,dayfirst=True)
    d=d.rename(columns=str.lower)[['date','open','high','low','close','volume']]
    return d.drop_duplicates('date').sort_values('date')
series={}
for c,f in [('BTC','BTC.csv'),('ETH','ETH.csv'),('LTC','LTC.csv')]:
    series[f'{c}-USD']=usd(f)
def loc(file,pair,name):
    d=pd.read_csv(f'{D}/{file}'); d=d[d.pair==pair].copy(); d['date']=pd.to_datetime(d.date)
    series[name]=d.drop(columns='pair').drop_duplicates('date').sort_values('date')
loc('upbit_raw.csv','KRW-BTC','BTC-KRW'); loc('upbit_raw.csv','KRW-ETH','ETH-KRW')
for c in ['btc','eth','ltc']:
    loc('bitbank_raw.csv',f'{c}_jpy',f'{c.upper()}-JPY'); loc('wazirx_raw.csv',f'{c}inr',f'{c.upper()}-INR')
fx=pd.read_csv(f'{D}/fred_fx.csv',parse_dates=['observation_date']).set_index('observation_date')
fx=fx.reindex(pd.date_range('2017-11-01',END)).ffill()   # FX known at NY noon; carried forward over weekends
fxcol={'KRW':'DEXKOUS','JPY':'DEXJPUS','INR':'DEXINUS'}
venue={'USD':'Global aggregate (CoinMarketCap via Yahoo Finance)','KRW':'Upbit (Seoul)','JPY':'bitbank (Tokyo)','INR':'WazirX (Mumbai)'}
info=[]
for k,d in series.items():
    d=d[(d.date>='2018-01-01')&(d.date<=END)].set_index('date')
    full=pd.date_range(d.index.min(),END); miss=full.difference(d.index)
    d=d.reindex(full); d[['open','high','low','close']]=d[['open','high','low','close']].ffill(); d['volume']=d.volume.fillna(0)
    coin,cur=k.split('-')
    if cur!='USD':
        u=series[f'{coin}-USD'].set_index('date').close.reindex(full)
        # timing: KRW/JPY local close = 15:00 UTC of date t -> latest USD daily close is date t-1.
        # INR (WazirX, UTC days) closes together with the USD series -> same date.
        lag=1 if cur in('KRW','JPY') else 0
        d['usd_ref']=u.shift(lag); d['fx']=fx[fxcol[cur]].reindex(full).shift(lag)
        d['premium']=100*(np.log(d.close)-np.log(d.usd_ref*d.fx))
    r=np.log(d.close).diff()
    info.append(dict(series=k,venue=venue[cur],start=d.index.min().date(),end=d.index.max().date(),days=len(d),
        filled_days=len(miss),zero_ret_pct=round(100*(r==0).mean(),2),ann_vol_pct=round(100*r.std()*np.sqrt(365),1),
        mean_prem=round(d.premium.mean(),2) if 'premium' in d else np.nan))
    d.index.name='date'; d.to_csv(f'{O}/{k}.csv')
info=pd.DataFrame(info); info.to_csv(f'{O}/series_info.csv',index=False); print(info.to_string())
