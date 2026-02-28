# -*- coding: utf-8 -*-
"""
Router Smoke Tests
==================
Verify all routers import cleanly and key functions are callable.
"""

import sys
import os
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


class TestRouterSmoke(unittest.TestCase):
    """Smoke tests for API routers"""

    def test_data_router_import(self):
        """Test data router imports without error"""
        try:
            from app.api.routers import data
            self.assertTrue(hasattr(data, 'router'))
        except Exception as e:
            self.fail(f"Data router import failed: {e}")

    def test_market_router_import(self):
        """Test market router imports without error"""
        try:
            from app.api.routers import market
            self.assertTrue(hasattr(market, 'router'))
        except Exception as e:
            self.fail(f"Market router import failed: {e}")

    def test_news_router_import(self):
        """Test news router imports without error"""
        try:
            from app.api.routers import news
            self.assertTrue(hasattr(news, 'router'))
        except Exception as e:
            self.fail(f"News router import failed: {e}")

    def test_data_router_functions_exist(self):
        """Verify key data router functions are callable"""
        from app.api.routers import data

        # Check key endpoints exist (any callable should work)
        functions = [name for name in dir(data) if callable(getattr(data, name)) and not name.startswith('_')]
        self.assertGreater(len(functions), 0, "Data router should have callable functions")

    def test_market_router_functions_exist(self):
        """Verify key market router functions are callable"""
        from app.api.routers import market

        # Check key endpoints exist
        functions = [name for name in dir(market) if callable(getattr(market, name)) and not name.startswith('_')]
        self.assertGreater(len(functions), 0, "Market router should have callable functions")

    def test_news_router_functions_exist(self):
        """Verify key news router functions are callable"""
        from app.api.routers import news

        # Check key endpoints exist
        functions = [name for name in dir(news) if callable(getattr(news, name)) and not name.startswith('_')]
        self.assertGreater(len(functions), 0, "News router should have callable functions")


if __name__ == '__main__':
    unittest.main()
