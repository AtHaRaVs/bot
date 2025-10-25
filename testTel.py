import os
import requests

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

print(f"Token: {TOKEN[:20]}...")
print(f"Chat ID: {CHAT_ID}")

url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

message = """
🎉 <b>SUCCESS!</b>

✅ Your bot is working!
✅ Chat ID: 1556429810
✅ Ready to receive trading notifications!

💰 Capital: ₹350
📊 Coins: MATIC, TRX, ADA, ALGO, XRP
⚡ Leverage: 3x

🚀 Start the trading bot now!
"""

data = {
    "chat_id": CHAT_ID,
    "text": message,
    "parse_mode": "HTML"
}

response = requests.post(url, data=data)

if response.status_code == 200:
    print("\n✅ SUCCESS! Check your Telegram!")
else:
    print(f"\n❌ ERROR: {response.status_code}")
    print(response.text)
