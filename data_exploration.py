import yfinance as yf
import pandas as pd

# Define the ticker symbol
ticker_symbol = 'AAPL'

# Get the ticker object
ticker = yf.Ticker(ticker_symbol)

# Get historical market data (prices, volume, etc.)
historical_data = ticker.history(period="max")
print("Historical Market Data:\n", historical_data)

# Get corporate actions (both dividends and splits)
actions = ticker.actions
print("\nCorporate Actions:\n", actions)

# Get quarterly financials
quarterly_financials = ticker.quarterly_financials
print("\nQuarterly Financials:\n", quarterly_financials)

# Get quarterly balance sheet
quarterly_balance_sheet = ticker.quarterly_balance_sheet
print("\nQuarterly Balance Sheet:\n", quarterly_balance_sheet)

# Get quarterly cash flow statement
quarterly_cashflow = ticker.quarterly_cashflow
print("\nQuarterly Cash Flow Statement:\n", quarterly_cashflow)

# # Get recommendations
# recommendations = ticker.recommendations
# print("\nRecommendations:\n", recommendations)



# Get earnings calendar
earnings_calendar = ticker.earnings_dates
print("\nEarnings Calendar:\n", earnings_calendar)

# # Get historical earnings
# earnings = ticker.earnings
# print("\nHistorical Earnings:\n", earnings)

# # Get analysts' price targets
# price_targets = ticker.analyst_price_target
# print("\nAnalysts' Price Targets:\n", price_targets)

# Get major holders
major_holders = ticker.major_holders
print("\nMajor Holders:\n", major_holders)

# Get institutional holders
institutional_holders = ticker.institutional_holders
print("\nInstitutional Holders:\n", institutional_holders)

# Get mutual fund holders
mutual_fund_holders = ticker.mutualfund_holders
print("\nMutual Fund Holders:\n", mutual_fund_holders)


# Get the financials (income statement)
income_statement = ticker.financials
print("Income Statement:\n", income_statement)

# Get the balance sheet
balance_sheet = ticker.balance_sheet
print("Balance Sheet:\n", balance_sheet)

# Get the cash flow statement
cash_flow = ticker.cashflow
print("Cash Flow Statement:\n", cash_flow)