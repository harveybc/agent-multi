import backtrader as bt
import pandas as pd
import datetime
import matplotlib.pyplot as plt

class HeuristicStrategy(bt.Strategy):
    """
    Forex Dynamic Volume Strategy using perfect (ideal) future predictions.

    This strategy uses pre‐computed future predictions (both hourly and daily)
    from a CSV file to decide on trade entry. When no position is open, it looks at the daily
    predictions (columns starting with 'Prediction_d_') to compute:
      - The ideal profit in pips for a long trade = (max(predictions) – current_price) / pip_cost.
      - The ideal drawdown in pips for a long trade = (current_price – min(predictions)) / pip_cost.
    (Analogous calculations are made for a short trade.)

    It then calculates a risk–reward ratio (RR) and, if the predicted profit meets a threshold,
    chooses the trade direction with the higher RR. The take‐profit (TP) and stop‐loss (SL)
    levels are set by applying configurable multipliers to the ideal profit and drawdown.

    Order size is computed by a linear interpolation between a minimum and maximum volume based
    on RR, capped by available cash * rel_volume * leverage.

    While in a trade, on every bar the strategy checks both the hourly and daily predictions
    (columns starting with 'Prediction_h_' and 'Prediction_d_') to decide if the trade should be
    closed early (if either the TP is hit or the predicted future price would breach the SL).

    Trade frequency is limited to a maximum number of trades per rolling 5‐day period.

    At the end of the simulation a summary is printed that includes:
      • Number of trades,
      • Average profit in USD,
      • Net average profit in pips,
      • Average absolute profit in pips,
      • Average trade duration (in bars),
      • Average max drawdown (in pips),
      • Minimum balance,
      • Initial and final balance.
    A balance versus date plot is also saved.
    """

    params = (
        # File paths for price and prediction data.
        ('price_file', '../trading-signal/output.csv'),
        ('pred_file', '../trading-signal/output.csv'),
        # Date range for filtering the data.
        ('date_start', datetime.datetime(2010, 1, 1)),
        ('date_end', datetime.datetime(2015, 1, 1)),
        # Trading parameters.
        ('pip_cost', 0.00001),          # 1 pip = 0.00001 for EURUSD.
        ('rel_volume', 0.05),           # Maximum fraction of cash to risk.
        ('min_order_volume', 10000),    # Minimum order volume (currency units).
        ('max_order_volume', 1000000),  # Maximum order volume (currency units).
        ('leverage', 1000),             # Leverage.
        ('profit_threshold', 5),        # Minimum ideal profit (in pips) required for entry.
        ('min_drawdown_pips', 10),      # Minimum drawdown (in pips) if predictions are too tight.
        # Multipliers for TP and SL.
        ('tp_multiplier', 0.9),         # TP = entry + 0.9 * ideal profit (pips) converted to price.
        ('sl_multiplier', 2.0),         # SL = entry - 2.0 * ideal drawdown (pips) (for long; inverse for short).
        # Risk–reward thresholds for order sizing.
        ('lower_rr_threshold', 0.5),    # RR below which use minimum volume.
        ('upper_rr_threshold', 2.0),    # RR above which use maximum volume.
        # Maximum trades allowed in any rolling 5-day period.
        ('max_trades_per_5days', 3),
    )

    def __init__(self):
        # Load prediction data from CSV and filter by the specified date range.
        self.pred_df = pd.read_csv(self.p.pred_file, parse_dates=['DATE_TIME'])
        self.pred_df = self.pred_df[(self.pred_df['DATE_TIME'] >= self.p.date_start) &
                                     (self.pred_df['DATE_TIME'] <= self.p.date_end)]
        # Floor DATE_TIME to the hour.
        self.pred_df['DATE_TIME'] = self.pred_df['DATE_TIME'].apply(
            lambda dt: dt.replace(minute=0, second=0, microsecond=0))
        self.pred_df.set_index('DATE_TIME', inplace=True)

        # Dynamically determine how many prediction columns exist.
        self.num_hourly_preds = len([col for col in self.pred_df.columns if col.startswith('Prediction_h_')])
        self.num_daily_preds = len([col for col in self.pred_df.columns if col.startswith('Prediction_d_')])

        # Get the price feed (assumed to be the first data feed).
        self.data0 = self.datas[0]

        # Save the initial balance.
        self.initial_balance = self.broker.getvalue()

        # Variables for managing the current trade.
        self.trade_entry_bar = None
        self.order_entry_price = None  # Set in notify_order.
        self.current_tp = None
        self.current_sl = None
        self.order_direction = None  # Used for custom attributes if needed.
        self.current_direction = None  # Will be set to 'long' or 'short' when an order is placed.

        # For tracking intra‐trade extreme prices.
        self.trade_low = None   # For long trades: the lowest price reached since entry.
        self.trade_high = None  # For short trades: the highest price reached since entry.

        # For enforcing trade frequency.
        self.trade_entry_dates = []

        # For plotting balance vs. date.
        self.balance_history = []
        self.date_history = []

        # For recording trade details.
        self.trades = []

    def next(self):
        dt = self.data0.datetime.datetime(0)
        dt_hour = dt.replace(minute=0, second=0, microsecond=0)
        current_price = self.data0.close[0]

        # Record current balance and date for plotting.
        self.balance_history.append(self.broker.getvalue())
        self.date_history.append(dt)

        # --- If a position is open, update intra‐trade extremes and check for exit ---
        if self.position:
            if self.current_direction == 'long':
                if self.trade_low is None or current_price < self.trade_low:
                    self.trade_low = current_price
                if dt_hour in self.pred_df.index:
                    preds_hourly = [self.pred_df.loc[dt_hour].get(f'Prediction_h_{i}', current_price)
                                    for i in range(1, self.num_hourly_preds+1)]
                    preds_daily = [self.pred_df.loc[dt_hour].get(f'Prediction_d_{i}', current_price)
                                   for i in range(1, self.num_daily_preds+1)]
                    predicted_min = min(preds_hourly + preds_daily)
                else:
                    predicted_min = current_price
                if current_price >= self.current_tp or predicted_min < self.current_sl:
                    self.close()
                    return
            elif self.current_direction == 'short':
                if self.trade_high is None or current_price > self.trade_high:
                    self.trade_high = current_price
                if dt_hour in self.pred_df.index:
                    preds_hourly = [self.pred_df.loc[dt_hour].get(f'Prediction_h_{i}', current_price)
                                    for i in range(1, self.num_hourly_preds+1)]
                    preds_daily = [self.pred_df.loc[dt_hour].get(f'Prediction_d_{i}', current_price)
                                   for i in range(1, self.num_daily_preds+1)]
                    predicted_max = max(preds_hourly + preds_daily)
                else:
                    predicted_max = current_price
                if current_price <= self.current_tp or predicted_max > self.current_sl:
                    self.close()
                    return
            return  # Do not attempt new entries if a position is open.
        else:
            # Reset intra‐trade extremes when no position is open.
            self.trade_low = current_price
            self.trade_high = current_price

        # --- Enforce trade frequency: no more than max_trades_per_5days in the last 5 days ---
        recent_trades = [d for d in self.trade_entry_dates if (dt - d).days < 5]
        if len(recent_trades) >= self.p.max_trades_per_5days:
            return

        # --- Check if prediction data exists for the current bar ---
        if dt_hour not in self.pred_df.index:
            return
        row = self.pred_df.loc[dt_hour]
        try:
            daily_preds = [row[f'Prediction_d_{i}'] for i in range(1, self.num_daily_preds+1)]
        except KeyError:
            return

        # --- Long (Buy) Calculations ---
        ideal_profit_pips_buy = (max(daily_preds) - current_price) / self.p.pip_cost
        if current_price > min(daily_preds):
            ideal_drawdown_pips_buy = (current_price - min(daily_preds)) / self.p.pip_cost
        else:
            ideal_drawdown_pips_buy = self.p.min_drawdown_pips
        rr_buy = ideal_profit_pips_buy / ideal_drawdown_pips_buy if ideal_drawdown_pips_buy > 0 else 0
        tp_buy = current_price + self.p.tp_multiplier * ideal_profit_pips_buy * self.p.pip_cost
        sl_buy = current_price - self.p.sl_multiplier * ideal_drawdown_pips_buy * self.p.pip_cost

        # --- Short (Sell) Calculations ---
        ideal_profit_pips_sell = (current_price - min(daily_preds)) / self.p.pip_cost
        if current_price < max(daily_preds):
            ideal_drawdown_pips_sell = (max(daily_preds) - current_price) / self.p.pip_cost
        else:
            ideal_drawdown_pips_sell = self.p.min_drawdown_pips
        rr_sell = ideal_profit_pips_sell / ideal_drawdown_pips_sell if ideal_drawdown_pips_sell > 0 else 0
        tp_sell = current_price - self.p.tp_multiplier * ideal_profit_pips_sell * self.p.pip_cost
        sl_sell = current_price + self.p.sl_multiplier * ideal_drawdown_pips_sell * self.p.pip_cost

        # --- Determine qualifying signals ---
        long_signal = (ideal_profit_pips_buy >= self.p.profit_threshold)
        short_signal = (ideal_profit_pips_sell >= self.p.profit_threshold)

        # Choose the signal with the higher risk–reward ratio.
        if long_signal and (rr_buy >= rr_sell):
            signal = 'long'
            chosen_tp = tp_buy
            chosen_sl = sl_buy
            chosen_rr = rr_buy
        elif short_signal and (rr_sell > rr_buy):
            signal = 'short'
            chosen_tp = tp_sell
            chosen_sl = sl_sell
            chosen_rr = rr_sell
        else:
            signal = None

        if signal is None:
            return

        # --- Compute order size based on RR ---
        order_size = self.compute_size(chosen_rr)
        if order_size <= 0:
            return

        # Record the trade entry details.
        self.trade_entry_dates.append(dt)
        self.trade_entry_bar = len(self)

        # Place the order and store the signal direction in self.current_direction.
        if signal == 'long':
            self.buy(size=order_size)
            self.current_direction = 'long'
        elif signal == 'short':
            self.sell(size=order_size)
            self.current_direction = 'short'
        # Save the chosen TP and SL.
        self.current_tp = chosen_tp
        self.current_sl = chosen_sl


    def compute_size(self, rr):
        """Compute order size by linear interpolation between min and max volumes based on RR."""
        min_vol = self.p.min_order_volume
        max_vol = self.p.max_order_volume
        if rr >= self.p.upper_rr_threshold:
            size = max_vol
        elif rr <= self.p.lower_rr_threshold:
            size = min_vol
        else:
            size = min_vol + ((rr - self.p.lower_rr_threshold) /
                              (self.p.upper_rr_threshold - self.p.lower_rr_threshold)) * (max_vol - min_vol)
        cash = self.broker.getcash()
        max_from_cash = cash * self.p.rel_volume * self.p.leverage
        return min(size, max_from_cash)

    def notify_order(self, order):
        """When an order is completed, record the execution price and capture its direction."""
        if order.status in [order.Completed]:
            self.order_entry_price = order.executed.price
            # Set the direction based on order type.
            self.order_direction = 'long' if order.isbuy() else 'short'
            
    def notify_trade(self, trade):
        """When a trade closes, record its results and print a summary.
           Profit (in pips) is computed using the stored order direction.
        """
        if trade.isclosed:
            # Determine trade duration.
            duration = len(self) - (self.trade_entry_bar if self.trade_entry_bar is not None else 0)
            dt = self.data0.datetime.datetime(0)
            entry_price = self.order_entry_price if self.order_entry_price is not None else 0
            exit_price = trade.price
            profit_usd = trade.pnlcomm

            # Use the stored order_direction to compute profit in pips.
            direction = self.order_direction
            if direction == 'long':
                profit_pips = (exit_price - entry_price) / self.p.pip_cost
            elif direction == 'short':
                profit_pips = (entry_price - exit_price) / self.p.pip_cost
            else:
                profit_pips = 0

            # Compute intra‐trade maximum drawdown (in pips) relative to the entry price.
            if self.order_direction == 'long':
                intra_dd = (entry_price - self.trade_low) / self.p.pip_cost if self.trade_low is not None else 0
            elif self.order_direction == 'short':
                intra_dd = (self.trade_high - entry_price) / self.p.pip_cost if self.trade_high is not None else 0
            else:
                intra_dd = 0

            current_balance = self.broker.getvalue()
            self.trades.append({
                'entry': entry_price,
                'exit': exit_price,
                'pnl': profit_usd,
                'pips': profit_pips,
                'duration': duration,
                'max_dd': intra_dd
            })

            print(f"TRADE CLOSED ({direction}): Date={dt}, Entry={entry_price:.5f}, Exit={exit_price:.5f}, "
                  f"Profit (pips)={profit_pips:.2f}, Profit (USD)={profit_usd:.2f}, "
                  f"Duration={duration} bars, Max DD (pips)={intra_dd:.2f}, Balance={current_balance:.2f}")

            # Reset trade-related variables.
            self.order_entry_price = None
            self.current_tp = None
            self.current_sl = None
            self.current_direction = None

    def stop(self):
        """At the end of the simulation, perform analysis, print summary statistics (including minimum balance),
        and plot the balance versus date."""
        # If a TradeAnalyzer was added, print its output.
        if hasattr(self, 'analyzers') and 'tradeanalyzer' in self.analyzers:
            trade_analyzer = self.analyzers.tradeanalyzer.get_analysis()
            print("\n==== Trade Analyzer Results ====")
            for key, value in trade_analyzer.items():
                print(f"{key}: {value}")

        # Compute the minimum balance encountered during the simulation.
        min_balance = min(self.balance_history) if self.balance_history else 0

        n_trades = len(self.trades)
        if n_trades > 0:
            avg_profit_usd = sum(t['pnl'] for t in self.trades) / n_trades
            avg_profit_pips = sum(t['pips'] for t in self.trades) / n_trades
            avg_profit_pips_abs = sum(abs(t['pips']) for t in self.trades) / n_trades
            avg_duration = sum(t['duration'] for t in self.trades) / n_trades
            avg_max_dd = sum(t['max_dd'] for t in self.trades) / n_trades
        else:
            avg_profit_usd = avg_profit_pips = avg_profit_pips_abs = avg_duration = avg_max_dd = 0
        final_balance = self.broker.getvalue()
        print("\n==== Summary ====")
        print(f"Initial Balance (USD): {self.initial_balance:.2f}")
        print(f"Final Balance (USD):   {final_balance:.2f}")
        print(f"Minimum Balance (USD): {min_balance:.2f}")
        print(f"Number of Trades: {n_trades}")
        print(f"Average Profit (USD): {avg_profit_usd:.2f}")
        print(f"Average Profit (pips): {avg_profit_pips:.2f}")
        print(f"Average Max Drawdown (pips): {avg_max_dd:.2f}")
        print(f"Average Trade Duration (bars): {avg_duration:.2f}")
        
        # Plot balance vs. date.
        plt.figure(figsize=(10, 5))
        plt.plot(self.date_history, self.balance_history, label="Balance")
        plt.xlabel("Date")
        plt.ylabel("Balance (USD)")
        plt.title("Balance vs Date")
        plt.legend()
        plt.savefig("balance_plot.png")
        plt.close()


if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        HeuristicStrategy,
        pred_file='../trading-signal/output.csv',
        pip_cost=0.00001,
        rel_volume=0.02,
        min_order_volume=10000,
        max_order_volume=1000000,
        leverage=1000,
        profit_threshold=5.08,
        date_start=datetime.datetime(2010, 1, 1),
        date_end=datetime.datetime(2015, 1, 1),
        min_drawdown_pips=10,
        tp_multiplier=0.92,
        sl_multiplier=6.3,
        lower_rr_threshold=-3.4,
        upper_rr_threshold=-4.95,
        max_trades_per_5days=3
    )

    data = bt.feeds.GenericCSVData(
        dataname='../trading-signal/output.csv',
        dtformat=('%Y-%m-%d %H:%M:%S'),
        datetime=0,
        time=-1,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=-1,
        openinterest=-1,
        timeframe=bt.TimeFrame.Minutes,
        compression=60,
        fromdate=datetime.datetime(2014, 1, 1),
        todate=datetime.datetime(2015, 1, 1)
    )

    cerebro.adddata(data)
    cerebro.broker.setcash(10000.0)
    # IMPORTANT: Do NOT add a sizer so that the strategy's computed order size is used.
    cerebro.run()
    print(f"Final Balance: {cerebro.broker.getvalue():.2f}")
