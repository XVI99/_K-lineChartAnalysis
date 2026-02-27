import pandas as pd
import numpy as np


class BacktestEngine:
    def __init__(self, initial_capital=100000.0, commission=0.0003, slippage=0.002,
                 stamp_tax=0.001, stop_loss_pct=None, trailing_stop_pct=None,
                 time_stop_days=None, time_stop_min_profit=0.02,
                 execution_mode="next_open",
                 # === Phase 2: Profit/Loss Optimization ===
                 take_profit_pct=None,
                 trailing_take_profit_pct=None,
                 trailing_take_profit_trigger=0.05,
                 partial_take_profit_pct=0.5,
                 partial_take_profit_at=2.0,
                 atr_stop_loss_multiplier=None,
                 atr_period=14,
                 # === Phase 4: Position Sizing & Market Scale ===
                 risk_per_trade=0.02,
                 position_scale=1.0,
                 max_position_pct=0.40):
        """
        Initialize the backtest engine with risk management parameters.

        Args:
            initial_capital (float): Starting cash.
            commission (float): Fee per trade (percentage, both buy and sell).
            slippage (float): Slippage per trade (percentage). Default 0.2%.
            stamp_tax (float): A-share sell-side stamp tax (0.1%). Only on sells.
            stop_loss_pct (float): Fixed stop loss percentage (e.g., 0.05 for 5%).
            trailing_stop_pct (float): Trailing stop percentage (e.g., 0.10 for 10%).
            time_stop_days (int|None): Force exit if held > N days with profit < time_stop_min_profit.
            time_stop_min_profit (float): Min profit to avoid time stop (default 2%).
            execution_mode (str): "close" = execute at current bar's close price (legacy),
                                  "next_open" = execute at next bar's open price (realistic).

            Phase 2 - Profit/Loss Optimization:
            take_profit_pct (float): Fixed take profit percentage (e.g., 0.10 for 10%).
            trailing_take_profit_pct (float): Trailing take profit percentage after trigger.
                                              Once triggered, take profit line follows price up.
            trailing_take_profit_trigger (float): Profit % to trigger trailing take profit (default 5%).
            partial_take_profit_pct (float): % of position to sell at partial take profit (default 50%).
            partial_take_profit_at (float): Risk-reward ratio to trigger partial take profit (default 2.0).
                                            E.g., 2.0 means sell partial when profit = 2 * risk.
            atr_stop_loss_multiplier (float): Use ATR-based stop loss = entry - multiplier * ATR.
                                              Overrides stop_loss_pct if provided.
            atr_period (int): ATR calculation period (default 14).
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.stamp_tax = stamp_tax
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.time_stop_days = time_stop_days
        self.time_stop_min_profit = time_stop_min_profit
        self.execution_mode = execution_mode

        # Phase 2: Profit/Loss Optimization
        self.take_profit_pct = take_profit_pct
        self.trailing_take_profit_pct = trailing_take_profit_pct
        self.trailing_take_profit_trigger = trailing_take_profit_trigger
        self.partial_take_profit_pct = partial_take_profit_pct
        self.partial_take_profit_at = partial_take_profit_at
        self.atr_stop_loss_multiplier = atr_stop_loss_multiplier
        self.atr_period = atr_period

        # Phase 4: Position Sizing & Market Scale
        self.risk_per_trade = risk_per_trade
        self.position_scale = position_scale
        self.max_position_pct = max_position_pct

        self.reset()

    def reset(self):
        """Reset the engine state."""
        self.cash = self.initial_capital
        self.holdings = 0
        self.portfolio_history = []
        self.trade_log = []
        self.equity_curve = []

        # Risk Management State
        self.entry_price = 0
        self.entry_cost_per_share = 0  # actual cost including fees
        self.high_since_entry = 0
        self.entry_bar_index = 0  # for time stop

        # Phase 2: Profit/Loss Optimization State
        self.trailing_take_profit_active = False
        self.trailing_take_profit_price = 0
        self.partial_take_profit_done = False
        self.atr_at_entry = 0
        self.dynamic_stop_loss = 0  # ATR-based stop loss price

    @staticmethod
    def _is_limit_up(close, prev_close, threshold=0.098):
        """Check if a stock hit the daily limit up (zhang ting)."""
        if prev_close <= 0:
            return False
        return (close - prev_close) / prev_close >= threshold

    @staticmethod
    def _is_limit_down(close, prev_close, threshold=0.098):
        """Check if a stock hit the daily limit down (die ting)."""
        if prev_close <= 0:
            return False
        return (prev_close - close) / prev_close >= threshold

    @staticmethod
    def _calculate_atr(df, period=14):
        """Calculate Average True Range (ATR).

        Args:
            df: DataFrame with High, Low, Close columns.
            period: ATR period (default 14).

        Returns:
            Series: ATR values.
        """
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    def run(self, df, signal_col="Signal", price_col="Close"):
        """
        Execute the backtest.

        In 'next_open' mode, signals on bar i are executed at bar i+1's Open price.
        In 'close' mode (legacy), signals execute at bar i's Close price.

        Limit-up/limit-down bars are skipped (no execution).

        Args:
            df: DataFrame with OHLCV data and signal column.
            signal_col: Column name for buy(1)/sell(-1) signals.
            price_col: Column name for reference price (used for equity calc).

        Returns:
            DataFrame: Equity curve.
        """
        self.reset()

        # Ensure data is sorted
        df = df.sort_index()

        if signal_col not in df.columns:
            print(f"Error: Signal column '{signal_col}' not found.")
            return None

        use_next_open = (self.execution_mode == "next_open") and ("Open" in df.columns)

        # Pre-compute previous close for limit checks
        prev_closes = df[price_col].shift(1)

        # === Phase 2: Pre-compute ATR if needed ===
        atr_series = None
        if self.atr_stop_loss_multiplier is not None:
            if "High" in df.columns and "Low" in df.columns:
                atr_series = self._calculate_atr(df, self.atr_period)
            else:
                print("Warning: ATR stop loss requested but High/Low data not available.")

        # --- Pending order state (for next_open mode) ---
        pending_signal = 0  # +1 buy, -1 sell
        pending_reason = ""

        # === Pre-extract numpy arrays for speed (avoids per-bar .iloc overhead) ===
        n = len(df)
        dates = df.index
        close_arr = df[price_col].values
        open_arr = df["Open"].values if "Open" in df.columns else close_arr
        signal_arr = df[signal_col].values
        prev_close_arr = prev_closes.values
        atr_arr = atr_series.values if atr_series is not None else None

        for i in range(n):
            date = dates[i]
            close_price = float(close_arr[i])
            open_price = float(open_arr[i])
            signal = int(signal_arr[i])
            prev_close = float(prev_close_arr[i]) if i > 0 else close_price

            # Get ATR value for this bar
            current_atr = float(atr_arr[i]) if atr_arr is not None else 0

            # --- Limit-up / Limit-down check on today's bar ---
            is_limit_up = self._is_limit_up(close_price, prev_close)
            is_limit_down = self._is_limit_down(close_price, prev_close)

            # --- Execute pending orders from previous bar (next_open mode) ---
            if use_next_open and pending_signal != 0:
                # Check if we can execute at today's open
                # Cannot buy on limit-up open (open = prev * 1.1), cannot sell on limit-down open
                open_limit_up = self._is_limit_up(open_price, prev_close)
                open_limit_down = self._is_limit_down(open_price, prev_close)

                if pending_signal == 1 and not open_limit_up and self.holdings == 0 and self.cash > 0:
                    self._execute_buy(date, open_price, i, current_atr)
                elif pending_signal == -1 and not open_limit_down and self.holdings > 0:
                    self._execute_sell(date, open_price, pending_reason)

                pending_signal = 0
                pending_reason = ""

            # --- Risk Management Checks (if holding) ---
            risk_sell = False
            partial_sell = False
            exit_reason = ""

            if self.holdings > 0:
                # Use close_price for risk checks (intraday monitoring)
                if close_price > self.high_since_entry:
                    self.high_since_entry = close_price

                current_profit_pct = (close_price - self.entry_price) / self.entry_price

                # === Phase 2: ATR-based Stop Loss ===
                if self.atr_stop_loss_multiplier is not None and self.dynamic_stop_loss > 0:
                    if close_price <= self.dynamic_stop_loss:
                        risk_sell = True
                        exit_reason = f"ATR Stop Loss ({self.atr_stop_loss_multiplier}*ATR)"
                elif self.stop_loss_pct is not None:
                    # Check Fixed Stop Loss (fallback)
                    sl_price = self.entry_price * (1 - self.stop_loss_pct)
                    if close_price <= sl_price:
                        risk_sell = True
                        exit_reason = "Stop Loss"

                # === Phase 2: Trailing Take Profit ===
                if not risk_sell and self.trailing_take_profit_pct is not None:
                    # Check if we should activate trailing take profit
                    if not self.trailing_take_profit_active:
                        if current_profit_pct >= self.trailing_take_profit_trigger:
                            self.trailing_take_profit_active = True
                            # Set initial trailing take profit price
                            self.trailing_take_profit_price = close_price * (1 - self.trailing_take_profit_pct)

                    # If active, update trailing price and check for exit
                    if self.trailing_take_profit_active:
                        # Move take profit line up as price rises
                        new_ttp_price = close_price * (1 - self.trailing_take_profit_pct)
                        if new_ttp_price > self.trailing_take_profit_price:
                            self.trailing_take_profit_price = new_ttp_price

                        # Check if price hit trailing take profit
                        if close_price <= self.trailing_take_profit_price:
                            risk_sell = True
                            exit_reason = "Trailing Take Profit"

                # === Phase 2: Fixed Take Profit ===
                if not risk_sell and self.take_profit_pct is not None:
                    tp_price = self.entry_price * (1 + self.take_profit_pct)
                    if close_price >= tp_price:
                        risk_sell = True
                        exit_reason = "Take Profit"

                # === Phase 2: Partial Take Profit ===
                if not risk_sell and not self.partial_take_profit_done and self.partial_take_profit_pct > 0:
                    if self.atr_at_entry > 0 and self.partial_take_profit_at > 0:
                        # Calculate risk (entry to stop loss)
                        risk_per_share = self.entry_price - self.dynamic_stop_loss if self.dynamic_stop_loss > 0 else self.entry_price * (self.stop_loss_pct or 0.05)
                        # Target profit = risk * reward ratio
                        target_profit = risk_per_share * self.partial_take_profit_at
                        target_price = self.entry_price + target_profit

                        if close_price >= target_price:
                            partial_sell = True
                            exit_reason = f"Partial Take Profit ({self.partial_take_profit_at}R)"

                # Check Trailing Stop (original)
                if self.trailing_stop_pct is not None and not risk_sell:
                    ts_price = self.high_since_entry * (1 - self.trailing_stop_pct)
                    if close_price <= ts_price:
                        risk_sell = True
                        exit_reason = "Trailing Stop"

                # Check Time Stop
                if self.time_stop_days is not None and not risk_sell:
                    bars_held = i - self.entry_bar_index
                    if bars_held >= self.time_stop_days and current_profit_pct < self.time_stop_min_profit:
                        risk_sell = True
                        exit_reason = f"Time Stop ({bars_held}d)"

            # --- Record portfolio value (using close price) ---
            current_equity = self.cash + (self.holdings * close_price)
            self.equity_curve.append({
                "Date": date, "Equity": current_equity,
                "Cash": self.cash, "Holdings": self.holdings
            })

            # --- Determine action ---
            want_sell = (signal == -1 or risk_sell) and self.holdings > 0
            want_buy = signal == 1 and self.cash > 0 and self.holdings == 0

            sell_reason = exit_reason if risk_sell else ""

            if use_next_open:
                # Queue as pending order, execute on next bar
                if want_sell:
                    pending_signal = -1
                    pending_reason = sell_reason if risk_sell else "Signal Sell"
                elif want_buy:
                    pending_signal = 1
            else:
                # Legacy mode: execute at this bar's close price
                exec_price = close_price
                if partial_sell and not is_limit_down:
                    # Execute partial sell
                    self._execute_partial_sell(date, exec_price, exit_reason)
                elif want_sell and not is_limit_down:
                    reason = sell_reason if risk_sell else ""
                    self._execute_sell(date, exec_price, reason)
                elif want_buy and not is_limit_up:
                    self._execute_buy(date, exec_price, i, current_atr)

        # --- Handle any remaining pending signal at end ---
        # (Skip: no next bar to execute on)

        # Finalize
        final_price = df[price_col].iloc[-1]
        final_equity = self.cash + (self.holdings * final_price)

        equity_df = pd.DataFrame(self.equity_curve).set_index("Date")
        return equity_df

    def _execute_buy(self, date, exec_price, bar_index, current_atr=0):
        """Execute a buy order with ATR-based position sizing.

        Position size is determined by:
        1. ATR-based risk: risk_amount / stop_distance (if ATR available)
        2. Max position cap: max_position_pct of total equity
        3. Market regime scale: position_scale (1.0=BULL, 0.5=NEUTRAL)
        Falls back to max_position_pct allocation if no ATR data.

        Args:
            date: Trade date.
            exec_price: Execution price.
            bar_index: Bar index for time stop.
            current_atr: Current ATR value for dynamic stop loss.
        """
        cost_per_share = exec_price * (1 + self.commission + self.slippage)

        # === Phase 4: ATR-based position sizing (replaces all-in) ===
        current_equity = self.cash + (self.holdings * exec_price)
        risk_amount = current_equity * self.risk_per_trade

        if self.atr_stop_loss_multiplier is not None and current_atr > 0:
            # ATR-based: size position so max loss = risk_amount
            stop_distance = current_atr * self.atr_stop_loss_multiplier
            risk_based_shares = risk_amount / stop_distance if stop_distance > 0 else 0
        else:
            # Fallback: use stop_loss_pct or default 5%
            sl_pct = self.stop_loss_pct or 0.05
            stop_distance = exec_price * sl_pct
            risk_based_shares = risk_amount / stop_distance if stop_distance > 0 else 0

        # Max position cap (don't put more than max_position_pct in one trade)
        max_invest = current_equity * self.max_position_pct
        max_affordable_shares = max_invest / cost_per_share

        # Also can't exceed available cash
        cash_max_shares = self.cash / cost_per_share

        # Apply market regime scale
        shares_to_buy = min(risk_based_shares, max_affordable_shares, cash_max_shares)
        shares_to_buy = int(shares_to_buy * self.position_scale)

        # A-share: round to lot of 100
        shares_to_buy = (shares_to_buy // 100) * 100

        # === Fallback for expensive stocks ===
        # If ATR sizing rounds to 0 (e.g., Moutai at 1400元), use max_position_pct
        if shares_to_buy == 0:
            fallback_shares = int(min(max_affordable_shares, cash_max_shares) * self.position_scale)
            fallback_shares = (fallback_shares // 100) * 100
            shares_to_buy = fallback_shares

        if shares_to_buy > 0:
            cost = shares_to_buy * cost_per_share
            self.cash -= cost
            self.holdings += shares_to_buy

            self.entry_price = exec_price
            self.entry_cost_per_share = cost_per_share
            self.high_since_entry = exec_price
            self.entry_bar_index = bar_index

            # === Phase 2: Set up ATR-based stop loss ===
            if self.atr_stop_loss_multiplier is not None and current_atr > 0:
                self.atr_at_entry = current_atr
                self.dynamic_stop_loss = exec_price - (current_atr * self.atr_stop_loss_multiplier)
            else:
                self.atr_at_entry = 0
                self.dynamic_stop_loss = 0

            # Reset Phase 2 states
            self.trailing_take_profit_active = False
            self.trailing_take_profit_price = 0
            self.partial_take_profit_done = False

            self.trade_log.append({
                "Date": date, "Type": "BUY", "Price": exec_price,
                "Shares": shares_to_buy, "Cost": cost, "Remaining_Cash": self.cash,
                "Position_Pct": round(cost / current_equity, 4) if current_equity > 0 else 0,
                "ATR": current_atr if current_atr > 0 else None,
                "Stop_Loss": self.dynamic_stop_loss if self.dynamic_stop_loss > 0 else None
            })

    def _execute_partial_sell(self, date, exec_price, exit_reason=""):
        """Execute a partial sell order (Phase 2: Partial Take Profit).

        Sells a portion of holdings based on partial_take_profit_pct.
        """
        if self.holdings <= 0:
            return

        # Calculate shares to sell (partial position)
        shares_to_sell = int(self.holdings * self.partial_take_profit_pct)
        # Round to lot of 100
        shares_to_sell = (shares_to_sell // 100) * 100

        if shares_to_sell <= 0:
            return

        # Sell cost: commission + slippage + stamp_tax (A-share sell only)
        revenue_per_share = exec_price * (1 - self.commission - self.slippage - self.stamp_tax)
        revenue = shares_to_sell * revenue_per_share

        # Calculate Trade Profit for partial position
        pnl = revenue - (shares_to_sell * self.entry_cost_per_share)
        pnl_pct = (exec_price - self.entry_price) / self.entry_price if self.entry_price > 0 else 0

        type_str = f"PARTIAL SELL ({exit_reason})" if exit_reason else "PARTIAL SELL"

        self.trade_log.append({
            "Date": date, "Type": type_str, "Price": exec_price,
            "Shares": shares_to_sell, "Revenue": revenue,
            "Remaining_Cash": self.cash + revenue, "PnL": pnl, "PnL_Pct": pnl_pct
        })

        # Update state
        self.cash += revenue
        self.holdings -= shares_to_sell
        self.partial_take_profit_done = True

        # Move stop loss to breakeven after partial take profit
        self.dynamic_stop_loss = self.entry_price  # Breakeven stop

    def _execute_sell(self, date, exec_price, exit_reason=""):
        """Execute a sell order. Includes stamp tax (sell-side only in A-shares)."""
        # Sell cost: commission + slippage + stamp_tax (A-share sell only)
        revenue_per_share = exec_price * (1 - self.commission - self.slippage - self.stamp_tax)
        revenue = self.holdings * revenue_per_share

        # Calculate Trade Profit
        pnl = revenue - (self.holdings * self.entry_cost_per_share)
        pnl_pct = (exec_price - self.entry_price) / self.entry_price if self.entry_price > 0 else 0

        type_str = "SELL" if not exit_reason else f"SELL ({exit_reason})"

        self.trade_log.append({
            "Date": date, "Type": type_str, "Price": exec_price,
            "Shares": self.holdings, "Revenue": revenue,
            "Remaining_Cash": self.cash + revenue, "PnL": pnl, "PnL_Pct": pnl_pct
        })

        self.cash += revenue
        self.holdings = 0
        self.entry_price = 0
        self.entry_cost_per_share = 0
        self.high_since_entry = 0
        self.entry_bar_index = 0

        # Reset Phase 2 states
        self.trailing_take_profit_active = False
        self.trailing_take_profit_price = 0
        self.partial_take_profit_done = False
        self.atr_at_entry = 0
        self.dynamic_stop_loss = 0

    def calculate_performance(self, benchmark_returns=None):
        """Calculates comprehensive performance metrics.

        Args:
            benchmark_returns: Optional Series of benchmark daily returns
                              for Alpha/Beta calculation.

        Returns:
            dict: Performance metrics.
        """
        if not self.equity_curve:
            return {}

        df = pd.DataFrame(self.equity_curve).set_index("Date")
        df["Returns"] = df["Equity"].pct_change()
        returns = df["Returns"].dropna()

        total_return = (df["Equity"].iloc[-1] - self.initial_capital) / self.initial_capital
        n_days = len(returns)
        annualized_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1 if n_days > 0 else 0
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 and not np.isnan(volatility) else 0

        # === Sortino Ratio (penalize only downside volatility) ===
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else 0
        sortino_ratio = (annualized_return - 0.02) / downside_vol if downside_vol > 0 else 0

        # === Max Drawdown & Recovery ===
        equity = df["Equity"]
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_drawdown = drawdown.min()

        # Calmar Ratio = Annualized Return / |Max Drawdown|
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Recovery Factor = Total Return / |Max Drawdown|
        recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # === Trade Stats ===
        trade_logs = pd.DataFrame(self.trade_log)
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
        max_consecutive_losses = 0
        max_consecutive_wins = 0
        expectancy = 0

        if not trade_logs.empty and "PnL" in trade_logs.columns:
            sells = trade_logs[trade_logs["Type"].str.contains("SELL")]
            if not sells.empty:
                wins = sells[sells["PnL"] > 0]
                losses = sells[sells["PnL"] <= 0]
                win_rate = len(wins) / len(sells)
                avg_win = wins["PnL_Pct"].mean() if not wins.empty else 0
                avg_loss = losses["PnL_Pct"].mean() if not losses.empty else 0
                total_wins = wins["PnL"].sum() if not wins.empty else 0
                total_losses = abs(losses["PnL"].sum()) if not losses.empty else 0
                profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

                # Expectancy (per trade expected return)
                expectancy = win_rate * abs(avg_win) - (1 - win_rate) * abs(avg_loss)

                # Max consecutive wins/losses
                pnl_series = sells["PnL"].values
                current_streak = 0
                max_loss_streak = 0
                max_win_streak = 0
                for pnl in pnl_series:
                    if pnl <= 0:
                        if current_streak > 0:
                            max_win_streak = max(max_win_streak, current_streak)
                            current_streak = 0
                        current_streak -= 1
                        max_loss_streak = max(max_loss_streak, abs(current_streak))
                    else:
                        if current_streak < 0:
                            max_loss_streak = max(max_loss_streak, abs(current_streak))
                            current_streak = 0
                        current_streak += 1
                        max_win_streak = max(max_win_streak, current_streak)
                max_consecutive_losses = max_loss_streak
                max_consecutive_wins = max_win_streak

        # === Alpha / Beta (if benchmark provided) ===
        alpha = 0
        beta = 0
        if benchmark_returns is not None:
            try:
                aligned = pd.concat([returns, benchmark_returns], axis=1, join='inner').dropna()
                if len(aligned) > 10:
                    aligned.columns = ['strategy', 'benchmark']
                    cov_matrix = np.cov(aligned['strategy'], aligned['benchmark'])
                    beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 0
                    alpha = (annualized_return - 0.02) - beta * (aligned['benchmark'].mean() * 252 - 0.02)
            except Exception:
                pass

        result = {
            "Total Return": f"{total_return:.2%}",
            "Annualized Return": f"{annualized_return:.2%}",
            "Max Drawdown": f"{max_drawdown:.2%}",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "Sortino Ratio": f"{sortino_ratio:.2f}",
            "Calmar Ratio": f"{calmar_ratio:.2f}",
            "Recovery Factor": f"{recovery_factor:.2f}",
            "Profit Factor": f"{profit_factor:.2f}",
            "Expectancy": f"{expectancy:.2%}",
            "Trade Count": len(self.trade_log),
            "Win Rate": f"{win_rate:.2%}",
            "Avg Win": f"{avg_win:.2%}",
            "Avg Loss": f"{avg_loss:.2%}",
            "Max Consecutive Wins": max_consecutive_wins,
            "Max Consecutive Losses": max_consecutive_losses,
        }

        if benchmark_returns is not None:
            result["Alpha"] = f"{alpha:.4f}"
            result["Beta"] = f"{beta:.2f}"

        return result


if __name__ == "__main__":
    # Quick sanity test
    pass
