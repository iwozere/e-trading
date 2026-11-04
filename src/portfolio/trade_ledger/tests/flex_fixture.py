"""Synthetic Flex Query XML shared by the trade-ledger tests (no real account data)."""

# Mirrors the real "My Trades" layout: two statements (accounts), STK + CASH
# trades, a two-leg reverse split, an internal transfer pair, a commission refund
# among other cash transactions, SUMMARY open positions.
FLEX_XML = b"""<FlexQueryResponse queryName="My Trades" type="AF">
<FlexStatements count="2">
<FlexStatement accountId="U0000001" fromDate="20250101" toDate="20251231" period="Custom" whenGenerated="20261104;120000">
<Trades>
<Trade accountId="U0000001" assetCategory="STK" levelOfDetail="EXECUTION" currency="USD" fxRateToBase="0.79"
  symbol="ABC" description="ABC CORP" conid="111" isin="US0000000111" listingExchange="NYSE" subCategory="COMMON"
  tradeID="9001" transactionID="5001" ibOrderID="7001" ibExecID="0001.01" tradeDate="20250521" dateTime="20250521;150756"
  buySell="BUY" quantity="15" tradePrice="22" proceeds="-330" ibCommission="-1.000525" ibCommissionCurrency="USD"
  taxes="0" netCash="-331.000525" cost="331.000525" fifoPnlRealized="0" openCloseIndicator="O" notes="P" />
<Trade accountId="U0000001" assetCategory="STK" levelOfDetail="EXECUTION" currency="USD" fxRateToBase="0.79"
  symbol="ABC" conid="111" tradeID="9002" tradeDate="20250601" dateTime="20250601;100000"
  buySell="SELL" quantity="-5" tradePrice="25" proceeds="125" ibCommission="-1" ibCommissionCurrency="USD"
  taxes="" netCash="124" cost="" fifoPnlRealized="13.67" openCloseIndicator="C" notes="" />
<Trade accountId="U0000001" assetCategory="CASH" levelOfDetail="EXECUTION" currency="CHF" symbol="USD.CHF" conid="12087817"
  tradeID="9003" tradeDate="20250102" dateTime="20250102;090000" buySell="BUY" quantity="1000" tradePrice="0.9" />
<Trade accountId="U0000001" assetCategory="STK" levelOfDetail="EXECUTION" currency="USD" symbol="BAD" conid="222"
  tradeID="9004" tradeDate="20250102" dateTime="20250102;090000" buySell="BUY" quantity="abc" tradePrice="1" />
</Trades>
<CorporateActions>
<CorporateAction accountId="U0000001" assetCategory="STK" currency="USD" symbol="ABC.OLD" conid="111" isin="US0000000111"
  actionID="777" transactionID="6001" type="RS" actionDescription="ABC SPLIT 1 FOR 10" reportDate="20250703"
  dateTime="20250702;202500" quantity="-10" proceeds="0" value="0" costBasis="" levelOfDetail="DETAIL" />
<CorporateAction accountId="U0000001" assetCategory="STK" currency="USD" symbol="ABC" conid="112" isin="US0000000112"
  actionID="777" transactionID="6002" type="RS" actionDescription="ABC SPLIT 1 FOR 10" reportDate="20250703"
  dateTime="20250702;202500" quantity="1" proceeds="0" value="0" costBasis="" levelOfDetail="DETAIL" />
</CorporateActions>
<Transfers>
<Transfer accountId="U0000001" assetCategory="STK" currency="USD" symbol="VT" conid="333" type="INTERNAL" direction="OUT"
  account="U0000002" company="--" transactionID="8001" date="20251229" dateTime="20251229;060851" quantity="-24"
  transferPrice="0" positionAmount="-3417.6" />
</Transfers>
<CashTransactions>
<CashTransaction accountId="U0000001" assetCategory="STK" currency="USD" symbol="ABC" conid="111" type="Commission Adjustments"
  transactionID="7001" tradeID="9001" dateTime="20250606" amount="1" description="Refund (ABC, 15, 2025-05-21)" />
<CashTransaction accountId="U0000001" assetCategory="STK" currency="USD" symbol="ABC" conid="111" type="Dividends"
  transactionID="7002" dateTime="20250610;202000" amount="2.5" description="ABC CASH DIVIDEND" />
<CashTransaction accountId="U0000001" assetCategory="" currency="USD" symbol="" conid="" type="Broker Interest Received"
  transactionID="7003" dateTime="20250605" amount="0.4" description="USD CREDIT INT" />
</CashTransactions>
<OpenPositions>
<OpenPosition accountId="U0000001" assetCategory="STK" levelOfDetail="SUMMARY" currency="USD" symbol="ABC" conid="112"
  reportDate="20251231" position="1" costBasisPrice="221.0" costBasisMoney="221.0" markPrice="250.5" />
<OpenPosition accountId="U0000001" assetCategory="STK" levelOfDetail="LOT" currency="USD" symbol="ABC" conid="112"
  reportDate="20251231" position="1" />
</OpenPositions>
</FlexStatement>
<FlexStatement accountId="U0000002" fromDate="20250101" toDate="20251231" period="Custom" whenGenerated="20261104;120000">
<Trades />
<Transfers>
<Transfer accountId="U0000002" assetCategory="STK" currency="USD" symbol="VT" conid="333" type="INTERNAL" direction="IN"
  account="U0000001" transactionID="8002" date="20251229" dateTime="" quantity="24" transferPrice="0" positionAmount="3417.6" />
</Transfers>
<OpenPositions>
<OpenPosition accountId="U0000002" assetCategory="STK" levelOfDetail="SUMMARY" currency="USD" symbol="VT" conid="333"
  position="24" costBasisPrice="130" costBasisMoney="3120" />
</OpenPositions>
</FlexStatement>
</FlexStatements>
</FlexQueryResponse>
"""
