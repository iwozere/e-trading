Below is a **clean, formal, production-ready spec** for BUY and SELL signals based on **EOM + Support/Resistance**, optionally enhanced with **RSI, MACD, ATR, Volume, Volatility filters**.

The goal:
✔ avoid noise
✔ confirm momentum
✔ avoid false breakouts
✔ be usable in real-time trading (Backtrader, CCXT, IBKR)
✔ be fully deterministic and implementable

You get **3 BUY signals** and **3 SELL signals**, each describing:

* Preconditions
* Indicators involved
* Exact numeric logic
* Safety filters
* Optional enhancements (ATR, RSI, MACD)

---

# ✅ **BUY SIGNALS (LONG ENTRIES)**

---

# **BUY #1 — Breakout + EOM Confirmation (Momentum Breakout Entry)**

**Purpose:** Enter after a strong breakout confirmed by EOM, volume, and volatility.

### **Indicators required**

* Support/Resistance (static or dynamic swing levels)
* EOM (14)
* Volume SMA (20)
* ATR (14)

---

### **Conditions (all must be true)**

**1. Breakout**

* `Close > Resistance_Level * 1.002`
  *(breakout by at least +0.2% to avoid noise)*

**2. EOM bullish**

* `EOM > 0`
* `EOM rising: EOM[0] > EOM[-1]`

**3. Volume confirmation**

* `Volume > SMA(Volume, 20)`

**4. ATR trend filter (optional but recommended)**

* `ATR > ATR_SMA(100)`
  *(avoids low-volatility zones where breakouts fail)*

**5. No overbought RSI**

* `RSI < 70`

---

### **Result:**

→ **BUY** at market (or use stop-limit at resistance breakout level)

---

---

# **BUY #2 — Pullback to Support + EOM Reversal (Mean-Reversion Trend-Continuation)**

**Purpose:** Trend-following entry after pullback respecting support.

### **Indicators**

* S/R
* EOM
* RSI
* ATR bands

---

### **Conditions**

**1. Price bounces from support**

* `Low <= Support_Level * 1.005`
* `Close > Open` *(reversal candle)*

**2. EOM reversal**

* `EOM crosses above 0`
  *(momentum turning positive)*

**3. RSI oversold → recovery**

* `RSI < 40` and `RSI[0] > RSI[-1]`

**4. ATR volatility floor**

* `ATR(14) > ATR(100) * 0.9`

---

### **Result:**

→ **BUY** at the close of reversal candle or next bar.

---

---

# **BUY #3 — MACD Bullish Turn + S/R Break (Momentum + Trend Confirmation)**

**Purpose:** Combine trend structure (MACD) + breakout (S/R).

### **Indicators**

* S/R
* MACD (12,26,9)
* EOM
* Volume

---

### **Conditions**

**1. MACD bullish**

* `MACD_line crosses above Signal_line`
* `MACD histogram rising`

**2. Resistance pre-breakout**

* `Close in range (Resistance*0.995 … Resistance*1.002)`
  *(tight consolidation near breakout level)*

**3. EOM positive**

* `EOM > 0`

**4. Volume expansion**

* Now or predicted breakout has:
* `Volume ≥ 0.8 * Volume_SMA(20)`

---

### **Result:**

→ **BUY** on next breakout above Resistance.

---

# ✅ **SELL SIGNALS (SHORT ENTRIES / EXIT LONG)**

---

# **SELL #1 — Breakdown + EOM Negative (Momentum Breakdown)**

**Purpose:** Opposite of Buy #1 — strong bearish momentum.

### **Conditions**

**1. Breakdown**

* `Close < Support_Level * 0.998`
  *(breakdown by -0.2%)*

**2. EOM bearish**

* `EOM < 0`
* `EOM falling: EOM[0] < EOM[-1]`

**3. Volume confirmation**

* `Volume > Volume_SMA(20)`

**4. ATR confirmation**

* `ATR rising vs yesterday`

---

### **Result:**

→ **SELL** (open short or exit long)

---

---

# **SELL #2 — Resistance Reject + EOM Turns Negative (Momentum Reversal Down)**

**Purpose:** Fade failed breakout / mean reversion down.

### **Conditions**

**1. Price rejection at resistance**

* `High >= Resistance * 0.995`
* `Close < Open`
  *(bearish rejection candle)*

**2. EOM crosses below 0**

* bearish EOM momentum reversal

**3. RSI overbought → falling**

* `RSI > 60`
* `RSI[0] < RSI[-1]`

---

### **Result:**

→ **SELL / Short**

---

---

# **SELL #3 — MACD Bearish Turn + Breakdown (Trend + Structure Confirm)**

**Purpose:** Combine strong trend shift + structural breakdown.

### **Conditions**

**1. MACD bearish cross**

* `MACD_line crosses below Signal_line`
* Histogram falling

**2. Breakdown confirmation**

* `Close < Support_Level * 0.998`

**3. EOM negative**

* `EOM < 0`

**4. Volume > SMA(20)**
*(breakdowns on high volume are more reliable)*

---

### **Result:**

→ **SELL / Short**

---

# 🎯 Summary Table

| Signal  | Type              | Uses S/R | EOM | RSI | MACD | ATR | Ideal Market Type            |
| ------- | ----------------- | -------- | --- | --- | ---- | --- | ---------------------------- |
| BUY #1  | Breakout momentum | ✔        | ✔   | —   | —    | ✔   | Trending breakout            |
| BUY #2  | Pullback reversal | ✔        | ✔   | ✔   | —    | ✔   | Trend continuation after dip |
| BUY #3  | MACD+Breakout     | ✔        | ✔   | —   | ✔    | —   | Early trend change           |
| SELL #1 | Breakdown         | ✔        | ✔   | —   | —    | ✔   | Trend down breakout          |
| SELL #2 | Rejection         | ✔        | ✔   | ✔   | —    | —   | Mean reversion               |
| SELL #3 | MACD+Breakdown    | ✔        | ✔   | —   | ✔    | —   | New bearish trend            |


----------------------------

def detect_swings(self):
    i = len(self.highs) - 3
    if i < 2:
        return

    h = self.highs
    l = self.lows

    # 2-bar swing high
    if h[i] > h[i-1] and h[i] > h[i+1]:
        self.swing_highs.append(h[i])

    # 2-bar swing low
    if l[i] < l[i-1] and l[i] < l[i+1]:
        self.swing_lows.append(l[i])

def nearest_resistance(self, price):
    vals = [x for x in self.swing_highs if x > price]
    return min(vals) if vals else None

def nearest_support(self, price):
    vals = [x for x in self.swing_lows if x < price]
    return max(vals) if vals else None

def next(self):
    # обновляем историю
    self.highs.append(self.data.high[0])
    self.lows.append(self.data.low[0])

    # ищем новые swing
    if len(self.highs) > 5:
        self.detect_swings()

    # расчёт уровней
    res = self.nearest_resistance(self.data.close[0])
    sup = self.nearest_support(self.data.close[0])

    eom = self.eom[0]
    eom_prev = self.eom[-1]

    vol = self.data.volume[0]
    vol_ma = self.vol_ma[0]

    close = self.data.close[0]

    # ----- BUY -----
    if res and close > res * 1.003:
        if eom > 0 and eom > eom_prev:
            if vol > vol_ma:
                self.buy()

    # ----- SELL -----
    if sup and close < sup * 0.997:
        if eom < 0 and eom < eom_prev:
            if vol > vol_ma:
                self.sell()
