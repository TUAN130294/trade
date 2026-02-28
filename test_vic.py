import requests

r = requests.get('https://banggia.cafef.vn/stockhandler.ashx', timeout=15)
data = r.json()

# Find VIC
for stock in data:
    if stock['a'] == 'VIC':
        print(f"VIC Real-time from CafeF:")
        print(f"  Symbol: {stock['a']}")
        print(f"  Current Price (l): {stock['l']}")
        print(f"  Change (k): {stock['k']}")
        print(f"  Ceiling (b): {stock['b']}")
        print(f"  Floor (d): {stock['d']}")
        print(f"  Reference (c): {stock['c']}")
        print(f"  Volume: {stock['totalvolume']}")
        break
