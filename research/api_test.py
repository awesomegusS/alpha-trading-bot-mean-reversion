from alpaca_trade_api.rest import REST
from dotenv import load_dotenv
import os

# --- CONFIGURATION ---
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
BASE_URL = "https://paper-api.alpaca.markets"

api = REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

def test_api_connection():
    print("=" * 40)
    print("INITIATING API PLUMBING TEST")
    print("=" * 40)
    
    # 1. Test Read Permissions (Account Status)
    try:
        account = api.get_account()
        print(f"Connection: SUCCESS")
        print(f"Account Status: {account.status}")
        print(f"Paper Buying Power: ${float(account.buying_power):,.2f}")
    except Exception as e:
        print(f"CRITICAL ERROR - Read Access Failed: {e}")
        return

    # 2. Test Write Permissions (Order Routing)
    print("\nRouting test order to exchange...")
    try:
        order = api.submit_order(
            symbol="BAC",
            qty=1,
            side='buy',
            type='market',
            time_in_force='day'
        )
        print(f"Order Routing: SUCCESS")
        print(f"Order ID: {order.id}")
        print(f"Order Status: {order.status.upper()}")
        print("\nNote: Because the market is closed, status will likely be 'ACCEPTED' rather than 'FILLED'.")
        print("It is queued for Monday morning.")
    except Exception as e:
        print(f"CRITICAL ERROR - Order Routing Failed: {e}")
        print("Check if your paper account is configured for margin/shorting.")

if __name__ == "__main__":
    test_api_connection()