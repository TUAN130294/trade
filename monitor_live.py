#!/usr/bin/env python3
"""
VN-QUANT Live Monitoring Dashboard
Real-time visualization of agent signals and trades
"""

import subprocess
import json
import time
import os
import sys
from datetime import datetime
from collections import deque

class MonitoringDashboard:
    def __init__(self):
        self.signal_history = deque(maxlen=50)
        self.trade_history = deque(maxlen=30)
        self.error_history = deque(maxlen=20)
        self.start_time = time.time()

    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def get_docker_logs(self, service, tail=100):
        """Get last N lines of docker logs"""
        try:
            result = subprocess.run(
                ['docker-compose', 'logs', '--tail', str(tail), service],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout
        except:
            return ""

    def get_api_status(self):
        """Get current trading status from API"""
        try:
            result = subprocess.run(
                ['curl', '-s', 'http://localhost:5176/api/status'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.stdout:
                return json.loads(result.stdout)
        except:
            pass
        return {}

    def get_docker_ps(self):
        """Get docker container status"""
        try:
            result = subprocess.run(
                ['docker-compose', 'ps', '--format', 'table {{.Names}}\\t{{.Status}}'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout
        except:
            return ""

    def extract_signals(self):
        """Extract agent signals from logs"""
        logs = self.get_docker_logs('autonomous', 300)
        signals = []

        agent_emojis = {
            'Scout': '🔭',
            'Alex': '📊',
            'Bull': '🐂',
            'Bear': '🐻',
            'RiskDoctor': '🏥',
            'Chief': '⚖️'
        }

        for line in logs.split('\n'):
            for agent, emoji in agent_emojis.items():
                if agent in line and any(word in line for word in ['BUY', 'SELL', 'HOLD', 'WARNING', 'VERDICT']):
                    signals.append({
                        'agent': agent,
                        'emoji': emoji,
                        'message': line.strip(),
                        'timestamp': datetime.now()
                    })

        return signals[-20:]  # Last 20 signals

    def extract_trades(self):
        """Extract executed trades from logs"""
        logs = self.get_docker_logs('autonomous', 200)
        trades = []

        for line in logs.split('\n'):
            if any(keyword in line for keyword in ['ORDER EXECUTED', 'POSITION CREATED', 'POSITION EXIT']):
                is_win = '+' in line
                trades.append({
                    'message': line.strip(),
                    'type': 'win' if is_win else 'loss' if '-' in line else 'info',
                    'timestamp': datetime.now()
                })

        return trades[-15:]  # Last 15 trades

    def extract_errors(self):
        """Extract errors and warnings from logs"""
        logs = self.get_docker_logs('autonomous', 500)
        errors = []

        for line in logs.split('\n'):
            if any(keyword in line.lower() for keyword in ['error', 'warning', 'failed', 'exception']):
                errors.append({
                    'message': line.strip(),
                    'timestamp': datetime.now()
                })

        return errors[-10:]  # Last 10 errors

    def print_header(self):
        """Print dashboard header"""
        print("\n")
        print("╔════════════════════════════════════════════════════════════════════════════════╗")
        print("║                  VN-QUANT LIVE MONITORING DASHBOARD                            ║")
        print("║                  Agent Signals • Trades • System Status                        ║")
        print(f"║  Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Uptime: {int(time.time() - self.start_time)}s                               ║")
        print("╚════════════════════════════════════════════════════════════════════════════════╝")
        print()

    def print_system_status(self):
        """Print system health status"""
        print("━━━ SYSTEM STATUS ━━━")

        # Service status
        ps_output = self.get_docker_ps()
        print(ps_output)

        # Trading status
        status = self.get_api_status()
        if status:
            print(f"\n📊 Trading Status:")
            print(f"   Status: {'🟢 RUNNING' if status.get('is_running') else '🔴 STOPPED'}")
            print(f"   Mode: {'📄 PAPER' if status.get('paper_trading') else '🔴 LIVE'}")
            print(f"   Balance: {status.get('balance', 'N/A')} VND")
            print(f"   Opportunities: {status.get('statistics', {}).get('opportunities_detected', 0)}")
            print(f"   Discussions: {status.get('statistics', {}).get('agent_discussions', 0)}")
            print(f"   Orders: {status.get('statistics', {}).get('orders_executed', 0)}")

        print()

    def print_agent_signals(self):
        """Print recent agent signals"""
        print("━━━ AGENT SIGNALS (Last 20) ━━━")

        signals = self.extract_signals()
        if not signals:
            print("   No signals yet...")
        else:
            for signal in signals[-20:]:
                time_ago = int((datetime.now() - signal['timestamp']).total_seconds())
                print(f"   {signal['emoji']} {signal['agent']}: {signal['message'][:80]} ({time_ago}s ago)")

        print()

    def print_trades(self):
        """Print recent trades"""
        print("━━━ TRADES EXECUTED (Last 15) ━━━")

        trades = self.extract_trades()
        if not trades:
            print("   No trades yet...")
        else:
            for trade in trades[-15:]:
                icon = "✅" if trade['type'] == 'win' else "📉" if trade['type'] == 'loss' else "📍"
                time_ago = int((datetime.now() - trade['timestamp']).total_seconds())
                print(f"   {icon} {trade['message'][:80]} ({time_ago}s ago)")

        print()

    def print_errors(self):
        """Print errors and warnings"""
        print("━━━ WARNINGS & ERRORS (Last 10) ━━━")

        errors = self.extract_errors()
        if not errors:
            print("   ✅ No errors detected")
        else:
            for error in errors[-10:]:
                time_ago = int((datetime.now() - error['timestamp']).total_seconds())
                print(f"   ⚠️  {error['message'][:80]} ({time_ago}s ago)")

        print()

    def run(self):
        """Main monitoring loop"""
        try:
            while True:
                self.clear_screen()
                self.print_header()
                self.print_system_status()
                self.print_agent_signals()
                self.print_trades()
                self.print_errors()

                print("━" * 88)
                print("Press Ctrl+C to stop | Refreshing in 5 seconds...")
                time.sleep(5)

        except KeyboardInterrupt:
            print("\n\n✅ Monitoring stopped")
            sys.exit(0)


if __name__ == "__main__":
    dashboard = MonitoringDashboard()
    dashboard.run()
