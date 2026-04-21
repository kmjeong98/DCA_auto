from dotenv import load_dotenv
load_dotenv("config/.env")
from src.common.api_client import APIClient

api = APIClient()
account = api.client.account()

print("totalMarginBalance:", account.get("totalMarginBalance"))
print("totalWalletBalance:", account.get("totalWalletBalance"))
print("totalUnrealizedProfit:", account.get("totalUnrealizedProfit"))
print()
for a in account.get("assets", []):
    bal = float(a.get("walletBalance", 0))
    if bal != 0:
        print(f"  {a['asset']}: wallet={a['walletBalance']}, margin={a.get('marginBalance')}, upnl={a.get('unrealizedProfit')}")

