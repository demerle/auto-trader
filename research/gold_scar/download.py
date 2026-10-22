"""Training data for gold_scar's model: Alpaca SIP 5m bars (split/dividend adjusted) since 2016 for the model's
symbols, and CBOE's daily VIX history. Uses the Alpaca keys in .env (free data plan).

    uv run python research/gold_scar/download.py [SYMBOL ...]
"""
import os, sys, time, httpx, pandas as pd
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from autotrader.options.ml.features import SYMBOLS
DATA = REPO / 'data' / 'research' / 'gold_scar'
OUT = DATA / 'bars'; OUT.mkdir(parents=True, exist_ok=True)
for line in open(REPO / '.env'):
    if '=' in line and not line.startswith('#'):
        k,v=line.strip().split('=',1); os.environ.setdefault(k,v.strip('"'))
H={"APCA-API-KEY-ID":os.environ['ALPACA_PAPER_KEY_ID'],"APCA-API-SECRET-KEY":os.environ['ALPACA_PAPER_SECRET_KEY']}
C=httpx.Client(headers=H,timeout=60)
def pull(s,start,end):
    rows=[]; tok=None
    while True:
        p=dict(symbols=s,timeframe='5Min',start=start,end=end,limit=10000,feed='sip',adjustment='all')
        if tok: p['page_token']=tok
        for a in range(8):
            try:
                r=C.get('https://data.alpaca.markets/v2/stocks/bars',params=p)
                if r.status_code==429: time.sleep(3*(a+1)); continue
                r.raise_for_status(); break
            except httpx.HTTPError: time.sleep(3*(a+1))
        j=r.json(); rows+=j['bars'].get(s,[]); tok=j.get('next_page_token')
        if not tok: break
    df=pd.DataFrame(rows); df['t']=pd.to_datetime(df['t'],utc=True); return df.set_index('t')
END = pd.Timestamp.now(tz='UTC').normalize().strftime('%Y-%m-%dT%H:%M:%SZ')
for s in sys.argv[1:] or SYMBOLS:
    f=OUT/f'{s}.parquet'
    if f.exists(): continue
    df=pull(s,'2016-01-01T00:00:00Z',END); df.to_parquet(f); print(s,len(df),df.index.min(),flush=True)
(DATA / 'cboe').mkdir(exist_ok=True)
r = httpx.get('https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv', follow_redirects=True,
              headers={'User-Agent': 'Mozilla/5.0'}, timeout=60)
r.raise_for_status(); (DATA / 'cboe' / 'VIX.csv').write_bytes(r.content); print('VIX history', len(r.content))
