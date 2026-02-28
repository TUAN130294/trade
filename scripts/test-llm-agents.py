#!/usr/bin/env python
"""
Quick test script for LLM-powered agent endpoints
Run: python scripts/test-llm-agents.py
"""
import os

import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def test_agent_chat():
    """Test /api/agents/chat endpoint"""
    print("\n" + "="*60)
    print("TEST 1: Agent Chat Endpoint")
    print("="*60)

    queries = [
        "phân tích MWG",
        "nên mua hay bán HPG?",
        "thị trường hôm nay thế nào?"
    ]

    for query in queries:
        print(f"\n🤔 Query: {query}")
        try:
            response = requests.post(
                f"{BASE_URL}/api/agents/chat",
                json={"query": query},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Agent: {data.get('agent')}")
                print(f"🤖 LLM Powered: {data.get('llm_powered', False)}")
                print(f"💬 Response:\n{data.get('response')[:200]}...")
            else:
                print(f"❌ Error: {response.status_code}")
        except Exception as e:
            print(f"❌ Failed: {e}")

def test_agent_analyze():
    """Test /api/agents/analyze endpoint"""
    print("\n" + "="*60)
    print("TEST 2: Agent Analysis Endpoint")
    print("="*60)

    symbols = ["MWG", "HPG", "FPT"]

    for symbol in symbols:
        print(f"\n📊 Analyzing: {symbol}")
        try:
            response = requests.post(
                f"{BASE_URL}/api/agents/analyze",
                json={"symbol": symbol},
                timeout=15
            )
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Success: {data.get('success')}")
                print(f"💰 Price: {data.get('technical', {}).get('price'):,.0f} VND")
                print(f"📈 Change: {data.get('technical', {}).get('change_pct'):.2f}%")
                print(f"🎯 Data Source: {data.get('technical', {}).get('data_source')}")

                messages = data.get('messages', [])
                print(f"\n👥 Agent Messages: {len(messages)}")
                for msg in messages[:2]:  # Show first 2
                    print(f"  {msg['emoji']} {msg['sender']}: {msg['content'][:80]}...")
            else:
                print(f"❌ Error: {response.status_code}")
        except Exception as e:
            print(f"❌ Failed: {e}")

def check_llm_proxy():
    """Check if LLM proxy is available"""
    print("\n" + "="*60)
    print("PRE-CHECK: LLM Proxy Status")
    print("="*60)

    try:
        response = requests.get(
            "http://localhost:8317/v1/models",
            headers={"Authorization": f"Bearer {os.getenv('LLM_API_KEY', '')}"},
            timeout=5
        )
        if response.status_code == 200:
            print("✅ LLM Proxy is running at http://localhost:8317/v1")
            models = response.json()
            print(f"📦 Available models: {models.get('data', [])[:3]}")
            return True
        else:
            print(f"⚠️ LLM Proxy responded with: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ LLM Proxy is NOT running at http://localhost:8317/v1")
        print("   Start it with: ccs-proxy or check CCS configuration")
        return False
    except Exception as e:
        print(f"❌ Error checking LLM proxy: {e}")
        return False

def main():
    """Run all tests"""
    print("\n🧪 Testing LLM-Powered Agent Endpoints")
    print("="*60)

    # Check LLM proxy first
    llm_available = check_llm_proxy()
    if not llm_available:
        print("\n⚠️ LLM proxy not available. Tests will use fallback responses.")
        print("   This is expected behavior - endpoints will still work.\n")

    # Test endpoints
    test_agent_chat()
    test_agent_analyze()

    print("\n" + "="*60)
    print("✅ All tests completed")
    print("="*60)

    if not llm_available:
        print("\n💡 To enable LLM responses:")
        print("   1. Start CCS proxy: ccs-proxy")
        print("   2. Or set LLM_BASE_URL and LLM_API_KEY in environment")
        print("   3. Ensure model 'claude-sonnet-4-6' is available")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏸️ Tests interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Test suite failed: {e}")
        sys.exit(1)
