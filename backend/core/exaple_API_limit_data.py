import requests
import json
import matplotlib.pyplot as plt

# Binance API endpoint for order book data
BASE_URL = "https://api.binance.com"
ORDER_BOOK_ENDPOINT = "/api/v3/depth"

# Function to get order book data
def get_order_book(symbol, limit=10):
    # Set up the URL with parameters
    url = f"{BASE_URL}{ORDER_BOOK_ENDPOINT}"
    params = {
        'symbol': symbol,
        'limit': limit  # The number of levels to retrieve (default is 100, maximum is 5000)
    }
    
    try:
        # Make a GET request to the API
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an error for bad responses
        order_book = response.json()
        return order_book
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
    except Exception as err:
        print(f"An error occurred: {err}")

# Example: Get the order book for Bitcoin/USDT
symbol = "BTCUSDT"
order_book = get_order_book(symbol, limit=10)

# Print order book data
print(json.dumps(order_book, indent=2))

# Extract bid and ask data
bids = order_book['bids']
asks = order_book['asks']

# Convert data to float for plotting
bid_prices = [float(bid[0]) for bid in bids]
bid_volumes = [float(bid[1]) for bid in bids]
ask_prices = [float(ask[0]) for ask in asks]
ask_volumes = [float(ask[1]) for ask in asks]
# Plot the order book
plt.figure(figsize=(10, 6))

# Plot bids
plt.plot(bid_prices, bid_volumes, label='Bids', color='green', linewidth=2)

# Plot asks
plt.plot(ask_prices, ask_volumes, label='Asks', color='red', linewidth=2)

# Add labels and legend
plt.title('Limit Order Book for BTC/USDT')
plt.xlabel('Price')
plt.ylabel('Volume')
plt.legend()