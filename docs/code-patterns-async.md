# AsyncIO & Concurrency Patterns

**Version:** 1.0
**Updated:** 2026-02-27
**Scope:** Python async/await patterns and concurrent execution

---

## AsyncIO Best Practices

### Pattern 1: Async Function Definition

```python
async def scan_stocks(self, symbols: List[str]) -> Dict[str, Prediction]:
    """Async function for concurrent stock scanning"""
    tasks = [self._predict_stock(symbol) for symbol in symbols]
    results = await asyncio.gather(*tasks)
    return dict(zip(symbols, results))

async def _predict_stock(self, symbol: str) -> Prediction:
    """Single stock prediction (runs concurrently)"""
    data = await self._fetch_data_async(symbol)
    return self.model.predict(data)
```

### Pattern 2: Running Concurrent Tasks

```python
# Good: Concurrent execution (all 3 tasks run in parallel)
async def run_all_scanners(self):
    """Run all scanners concurrently"""
    await asyncio.gather(
        self.model_scanner.scan(),
        self.news_scanner.scan(),
        self.position_monitor.check()
    )

# Bad: Sequential execution (tasks wait for each other)
async def run_all_scanners_bad(self):
    await self.model_scanner.scan()      # Waits
    await self.news_scanner.scan()       # Then runs
    await self.position_monitor.check()  # Then runs
```

### Pattern 3: AsyncIO Queue for Event Handling

```python
class EventBus:
    """Async event publishing/subscription system"""

    def __init__(self, maxsize: int = 1000):
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)

    async def publish(self, event: Event) -> None:
        """Publish event to queue (non-blocking)"""
        await self.queue.put(event)

    async def subscribe(self) -> Event:
        """Consume event from queue (blocking until available)"""
        return await self.queue.get()

    async def process_all_events(self, handler: Callable):
        """Process all events in queue with handler"""
        while not self.queue.empty():
            event = await self.queue.get()
            await handler(event)
            self.queue.task_done()

# Usage: producers push events, consumers pull and process
```

### Pattern 4: Async Context Managers

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def database_connection():
    """Async context manager for DB operations"""
    conn = await connect_db()
    try:
        yield conn
    finally:
        await conn.close()

# Usage: ensures connection closes even if exception occurs
async with database_connection() as conn:
    result = await conn.execute("SELECT * FROM orders")
    # Auto-closes on exit
```

### Pattern 5: Timeout Handling

```python
async def fetch_with_timeout(self, symbol: str, timeout: float = 5.0):
    """Fetch data with timeout protection"""
    try:
        result = await asyncio.wait_for(
            self._fetch_data(symbol),
            timeout=timeout
        )
        return result
    except asyncio.TimeoutError:
        logger.warning(f"Timeout fetching {symbol}")
        return None
```

### Pattern 6: Task Cancellation

```python
async def main():
    """Demo of task cancellation"""
    # Create a long-running task
    task = asyncio.create_task(long_running_operation())

    try:
        # Wait with timeout
        await asyncio.wait_for(task, timeout=30.0)
    except asyncio.TimeoutError:
        logger.warning("Operation timed out, cancelling...")
        task.cancel()

        try:
            await task  # Wait for cancellation to complete
        except asyncio.CancelledError:
            logger.info("Task cancelled successfully")
```

---

## VN-Quant Async Patterns

### Orchestrator Concurrent Scanning

```python
# In AutonomousOrchestrator
async def start(self):
    """Run all components concurrently"""
    await asyncio.gather(
        self._run_model_scanner(),      # Path A: every 3 minutes
        self._run_news_scanner(),       # Path B: every 5 minutes
        self._process_opportunities(),  # Consumer: handle opportunities
        self._monitor_positions(),      # Monitor: check exits every 60s
        self._monitor_system_health()   # Health: log metrics
    )

async def _run_model_scanner(self):
    """Path A: Model scanning (3-minute intervals)"""
    while self.is_running:
        try:
            opportunities = await self.model_scanner.scan()
            for opp in opportunities:
                await self.agent_message_queue.put(opp)
        except Exception as e:
            logger.error(f"Model scan failed: {e}")
        finally:
            await asyncio.sleep(180)  # 3 minutes

async def _process_opportunities(self):
    """Consumer: Process queued opportunities"""
    while self.is_running:
        try:
            opp = await asyncio.wait_for(
                self.agent_message_queue.get(),
                timeout=60
            )
            await self._execute_opportunity(opp)
        except asyncio.TimeoutError:
            continue  # No opportunities this check
        except Exception as e:
            logger.error(f"Opportunity processing failed: {e}")
```

### Agent Discussion (Parallel Analysis)

```python
async def discuss_opportunity(self, opportunity: Opportunity) -> AgentSignal:
    """Run all agents in parallel for discussion"""
    stock_data = await self._fetch_stock_data(opportunity.symbol)

    # Run all agents concurrently
    signals = await asyncio.gather(
        self.bull_agent.analyze(stock_data),
        self.bear_agent.analyze(stock_data),
        self.analyst_agent.analyze(stock_data),
        self.scout_agent.analyze(stock_data),
        self.risk_doctor_agent.analyze(stock_data),
        return_exceptions=True
    )

    # Filter out exceptions
    valid_signals = [s for s in signals if not isinstance(s, Exception)]

    # Chief aggregates
    final_verdict = self.chief_agent.aggregate(valid_signals)
    return final_verdict
```

### WebSocket Broadcasting

```python
# In FastAPI endpoint
async def broadcast_to_clients(message: dict):
    """Broadcast message to all connected WebSocket clients concurrently"""
    if not connected_websockets:
        return

    # Send to all clients in parallel
    tasks = [
        ws.send_json(message)
        for ws in connected_websockets
    ]

    # Execute all sends concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle failures
    for ws, result in zip(connected_websockets, results):
        if isinstance(result, Exception):
            logger.warning(f"Failed to send to client: {result}")
            connected_websockets.discard(ws)
```

---

## Performance Considerations

### Concurrent vs Sequential

**Concurrent (Async):** Best for I/O-bound operations
```python
# Concurrent: All 3 fetches happen at same time (~1s total)
await asyncio.gather(
    fetch_cafef_data(),    # 0.5s
    fetch_vps_data(),      # 0.5s
    fetch_rss_news()       # 0.5s
)
# Total: ~0.5s (parallel)
```

**Sequential (Bad):** Wasteful for I/O-bound
```python
# Sequential: Each fetch waits for previous (~1.5s total)
await fetch_cafef_data()   # 0.5s
await fetch_vps_data()     # 0.5s
await fetch_rss_news()     # 0.5s
# Total: 1.5s (serial)
```

### Memory Efficiency

```python
# Bad: Creates all 102 prediction tasks at once (memory spike)
tasks = [predict(symbol) for symbol in all_102_symbols]
results = await asyncio.gather(*tasks)

# Good: Process in batches (controlled memory)
batch_size = 10
results = []
for i in range(0, len(all_102_symbols), batch_size):
    batch = all_102_symbols[i:i+batch_size]
    batch_results = await asyncio.gather(
        *[predict(symbol) for symbol in batch]
    )
    results.extend(batch_results)
```

---

## Debugging Async Code

### Check for Blocking Operations

```python
# Bad: Blocking I/O in async function (blocks event loop)
async def bad_fetch():
    result = requests.get("http://example.com")  # BLOCKS!
    return result

# Good: Use async-compatible library
async def good_fetch():
    async with aiohttp.ClientSession() as session:
        async with session.get("http://example.com") as resp:
            return await resp.json()
```

### Timeout Prevention

```python
# Always add timeouts to async operations
try:
    result = await asyncio.wait_for(
        operation(),
        timeout=30.0
    )
except asyncio.TimeoutError:
    logger.error("Operation timed out after 30s")
```

### Task Monitoring

```python
# Monitor pending tasks (useful for debugging)
async def monitor_tasks():
    """Log pending tasks for debugging"""
    while True:
        pending = asyncio.all_tasks()
        logger.debug(f"Pending tasks: {len(pending)}")
        for task in pending:
            logger.debug(f"  - {task.get_name()}")
        await asyncio.sleep(60)
```

---

*These patterns ensure efficient, non-blocking concurrent execution throughout the VN-Quant system.*
