import sys

with open("d:/testpapertr/run_autonomous_paper_trading.py", 'r', encoding='utf-8') as f:
    content = f.read()

# Locate the start and end of the block
start_str = '@app.get("/", response_class=HTMLResponse)'
end_str = '@app.websocket("/ws/autonomous")'

start_idx = content.find(start_str)
end_idx = content.find(end_str)

if start_idx != -1 and end_idx != -1:
    new_content = content[:start_idx] + '''@app.get("/")
async def homepage():
    """Healthcheck endpoint"""
    return {"status": "ok", "service": "Autonomous Trading Model API"}

''' + content[end_idx:]
    with open("d:/testpapertr/run_autonomous_paper_trading.py", 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Successfully removed inline HTML!")
else:
    print("Could not find the target strings.")
