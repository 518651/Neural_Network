import pandas as pd
import requests
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
import warnings

warnings.filterwarnings("ignore")

proxy = {'https': 'http://localhost:10809'}

url = "https://web-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical"
param = {"convert": "USD", "slug": "bitcoin", "time_end": "1601510400", "time_start": "1367107200"}
content = requests.get(url=url, params=param, proxies=proxy, verify=False).json()
df = pd.json_normalize(content['data']['quotes'])

# 提取和重命名重要变量
df['Date'] = pd.to_datetime(df['quote.USD.timestamp']).dt.tz_localize(None)  # 继承子级嵌套Data
df['Low'] = df['quote.USD.low']  # USD最低价格
df['High'] = df['quote.USD.high']  # USD最高价格
df['Close'] = df['quote.USD.close']  # 闭盘USD价格
df['Volume'] = df['quote.USD.volume']  # 保留参数
df['Open'] = df['quote.USD.open']  # 开盘USD价格

# 删除原始列和冗余列
df = df.drop(
    columns=['time_open', 'time_close', 'time_high', 'time_low', 'quote.USD.low', 'quote.USD.high', 'quote.USD.open',
             'quote.USD.close', 'quote.USD.volume', 'quote.USD.market_cap', 'quote.USD.timestamp'])

#  导出数据
outdata = pd.DataFrame(df)
outdata.to_csv('BTC-COIN-DATA.csv', index=False)
outdata.to_excel('BTC-COIN-DATA.xlsx', index=False)
