import requests

r = requests.get('http://localhost:8003/api/market/smart-signals', timeout=30)
data = r.json()

print(f'Source: {data.get("source")}')
print(f'Total signals: {data.get("count")}')
print()

for sig in data.get('signals', []):
    print(f"[{sig['type']}] {sig['name']}")
    desc = sig['description'][:100] if len(sig['description']) > 100 else sig['description']
    print(f"   {desc}")
    print(f"   Action: {sig['action']}")
    print(f"   Source: {sig.get('source', 'N/A')}")
    print()
