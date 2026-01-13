#!/usr/bin/env python3
"""
Quick test script to place a single trade order.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from config import BotConfig
from kalshi_client import KalshiClient

async def test_trade():
    """Test placing a single trade order."""
    try:
        print("Loading configuration...")
        # Load configuration
        config = BotConfig()
        print("Configuration loaded successfully")

        print("Initializing Kalshi client...")
        # Initialize client
        client = KalshiClient(config.kalshi)
        print("Client initialized")

        print("Logging in...")
        # Login
        await client.login()
        print("Login successful")

        # Test parameters from the user's example
        ticker = "KXNFLTOTAL-26JAN12HOUPIT-33"
        side = "yes"
        amount = 1.0  # Use smaller amount for demo testing

        print(f"Testing trade: {ticker} | Side: {side} | Amount: ${amount}")

        # Place the order
        result = await client.place_order(ticker, side, amount)

        # Test that place_order is working correctly
        assert result.get("success") == True, f"Order placement failed: {result}"
        assert "order_id" in result, "Order ID not returned"
        assert "client_order_id" in result, "Client Order ID not returned"
        assert result["order_id"], "Order ID is empty"
        assert result["client_order_id"], "Client Order ID is empty"

        print("✅ Trade placed successfully!")
        print(f"Order ID: {result.get('order_id')}")
        print(f"Client Order ID: {result.get('client_order_id')}")
        print(f"Full response: {result.get('response')}")

        # Note: buy_max_cost is set internally in place_order as amount * 100 = {amount * 100}

    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_trade())