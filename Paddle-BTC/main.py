import requests


proxy = {'https': 'http://localhost:10809'}

url = "https://web-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical"
param = {"convert": "USD", "slug": "bitcoin", "time_end": "1601510400", "time_start": "1367107200"}
content = requests.get(url=url, params=param, proxies=proxy, verify=False).json()
print(content)