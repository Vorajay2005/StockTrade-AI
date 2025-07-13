"""
Technical indicators for stock analysis
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple
from config import INDICATORS

class TechnicalIndicators:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_sma(self, data: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """
        Calculate Simple Moving Average for multiple periods
        
        Args:
            data: DataFrame with price data
            periods: List of periods for SMA calculation
            
        Returns:
            DataFrame with SMA columns added
        """
        df = data.copy()
        
        for period in periods:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            
        return df
    
    def calculate_ema(self, data: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """
        Calculate Exponential Moving Average for multiple periods
        
        Args:
            data: DataFrame with price data
            periods: List of periods for EMA calculation
            
        Returns:
            DataFrame with EMA columns added
        """
        df = data.copy()
        
        for period in periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            
        return df
    
    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Relative Strength Index
        
        Args:
            data: DataFrame with price data
            period: Period for RSI calculation (default 14)
            
        Returns:
            DataFrame with RSI column added
        """
        df = data.copy()
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def calculate_macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            data: DataFrame with price data
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal line EMA period (default 9)
            
        Returns:
            DataFrame with MACD columns added
        """
        df = data.copy()
        
        ema_fast = df['close'].ewm(span=fast).mean()
        ema_slow = df['close'].ewm(span=slow).mean()
        
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        return df
    
    def calculate_bollinger_bands(self, data: pd.DataFrame, period: int = 20, std_dev: float = 2) -> pd.DataFrame:
        """
        Calculate Bollinger Bands
        
        Args:
            data: DataFrame with price data
            period: Period for moving average (default 20)
            std_dev: Standard deviation multiplier (default 2)
            
        Returns:
            DataFrame with Bollinger Bands columns added
        """
        df = data.copy()
        
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        
        df['bb_upper'] = sma + (std * std_dev)
        df['bb_middle'] = sma
        df['bb_lower'] = sma - (std * std_dev)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    def calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3, smooth_k: int = 3) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator
        
        Args:
            data: DataFrame with price data
            k_period: Period for %K calculation (default 14)
            d_period: Period for %D calculation (default 3)
            smooth_k: Period for %K smoothing (default 3)
            
        Returns:
            DataFrame with Stochastic columns added
        """
        df = data.copy()
        
        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()
        
        k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
        df['stoch_k'] = k_percent.rolling(window=smooth_k).mean()
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        
        return df
    
    def calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Average Directional Index (ADX)
        
        Args:
            data: DataFrame with price data
            period: Period for ADX calculation (default 14)
            
        Returns:
            DataFrame with ADX columns added
        """
        df = data.copy()
        
        # Calculate True Range
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift())
        df['tr3'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate directional movements
        df['dm_plus'] = np.where((df['high'] - df['high'].shift()) > (df['low'].shift() - df['low']), 
                                 np.maximum(df['high'] - df['high'].shift(), 0), 0)
        df['dm_minus'] = np.where((df['low'].shift() - df['low']) > (df['high'] - df['high'].shift()), 
                                  np.maximum(df['low'].shift() - df['low'], 0), 0)
        
        # Calculate smoothed values
        df['atr'] = df['tr'].rolling(window=period).mean()
        df['di_plus'] = 100 * (df['dm_plus'].rolling(window=period).mean() / df['atr'])
        df['di_minus'] = 100 * (df['dm_minus'].rolling(window=period).mean() / df['atr'])
        
        # Calculate ADX
        df['dx'] = 100 * abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus'])
        df['adx'] = df['dx'].rolling(window=period).mean()
        
        # Clean up temporary columns
        df.drop(['tr1', 'tr2', 'tr3', 'tr', 'dm_plus', 'dm_minus', 'atr', 'dx'], axis=1, inplace=True)
        
        return df
    
    def calculate_cci(self, data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Calculate Commodity Channel Index (CCI)
        
        Args:
            data: DataFrame with price data
            period: Period for CCI calculation (default 20)
            
        Returns:
            DataFrame with CCI column added
        """
        df = data.copy()
        
        tp = (df['high'] + df['low'] + df['close']) / 3
        ma = tp.rolling(window=period).mean()
        md = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        
        df['cci'] = (tp - ma) / (0.015 * md)
        
        return df
    
    def calculate_roc(self, data: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """
        Calculate Rate of Change (ROC)
        
        Args:
            data: DataFrame with price data
            period: Period for ROC calculation (default 10)
            
        Returns:
            DataFrame with ROC column added
        """
        df = data.copy()
        
        df['roc'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
        
        return df
    
    def calculate_williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Williams %R
        
        Args:
            data: DataFrame with price data
            period: Period for Williams %R calculation (default 14)
            
        Returns:
            DataFrame with Williams %R column added
        """
        df = data.copy()
        
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        
        df['williams_r'] = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
        
        return df
    
    def calculate_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-based indicators
        
        Args:
            data: DataFrame with price and volume data
            
        Returns:
            DataFrame with volume indicators added
        """
        df = data.copy()
        
        # Volume Moving Average
        df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        
        # Volume Rate of Change
        df['volume_roc'] = ((df['volume'] - df['volume'].shift(10)) / df['volume'].shift(10)) * 100
        
        # On-Balance Volume (OBV)
        df['obv'] = 0
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                df['obv'].iloc[i] = df['obv'].iloc[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                df['obv'].iloc[i] = df['obv'].iloc[i-1] - df['volume'].iloc[i]
            else:
                df['obv'].iloc[i] = df['obv'].iloc[i-1]
        
        # Volume Weighted Average Price (VWAP)
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        return df
    
    def calculate_price_action_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price action indicators
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with price action indicators added
        """
        df = data.copy()
        
        # Price change and percentage change
        df['price_change'] = df['close'].diff()
        df['price_change_pct'] = df['close'].pct_change() * 100
        
        # Volume change
        df['volume_change'] = df['volume'].diff()
        df['volume_change_pct'] = df['volume'].pct_change() * 100
        
        # Volatility (rolling standard deviation)
        df['volatility'] = df['close'].rolling(window=20).std()
        
        # Daily range
        df['daily_range'] = df['high'] - df['low']
        df['daily_range_pct'] = (df['daily_range'] / df['close']) * 100
        
        # Gap detection
        df['gap'] = df['open'] - df['close'].shift()
        df['gap_pct'] = (df['gap'] / df['close'].shift()) * 100
        
        # Support and resistance levels (pivot points)
        df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
        df['r1'] = 2 * df['pivot'] - df['low']
        df['s1'] = 2 * df['pivot'] - df['high']
        df['r2'] = df['pivot'] + (df['high'] - df['low'])
        df['s2'] = df['pivot'] - (df['high'] - df['low'])
        
        return df
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all indicators added
        """
        df = data.copy()
        
        try:
            # Moving averages
            df = self.calculate_sma(df, INDICATORS["SMA"])
            df = self.calculate_ema(df, INDICATORS["EMA"])
            
            # Momentum indicators
            df = self.calculate_rsi(df, INDICATORS["RSI"])
            df = self.calculate_macd(df, *INDICATORS["MACD"])
            df = self.calculate_roc(df, INDICATORS["ROC"])
            
            # Volatility indicators
            df = self.calculate_bollinger_bands(df, *INDICATORS["BB"])
            
            # Oscillators
            df = self.calculate_stochastic(df, *INDICATORS["STOCH"])
            df = self.calculate_williams_r(df, INDICATORS["WILLIAMS"])
            df = self.calculate_cci(df, INDICATORS["CCI"])
            
            # Trend indicators
            df = self.calculate_adx(df, INDICATORS["ADX"])
            
            # Volume indicators
            df = self.calculate_volume_indicators(df)
            
            # Price action indicators
            df = self.calculate_price_action_indicators(df)
            
            self.logger.info(f"Calculated all technical indicators for {len(df)} data points")
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            
        return df
    
    def get_trading_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on technical indicators
        
        Args:
            data: DataFrame with technical indicators
            
        Returns:
            DataFrame with trading signals added
        """
        df = data.copy()
        
        # Initialize signals
        df['signal'] = 0  # 0 = Hold, 1 = Buy, -1 = Sell
        df['signal_strength'] = 0.0  # Signal strength (0-1)
        
        # RSI signals
        rsi_buy = (df['rsi'] < 30)
        rsi_sell = (df['rsi'] > 70)
        
        # MACD signals
        macd_buy = (df['macd'] > df['macd_signal']) & (df['macd'].shift() <= df['macd_signal'].shift())
        macd_sell = (df['macd'] < df['macd_signal']) & (df['macd'].shift() >= df['macd_signal'].shift())
        
        # Moving average crossover signals
        ma_buy = (df['ema_12'] > df['ema_26']) & (df['ema_12'].shift() <= df['ema_26'].shift())
        ma_sell = (df['ema_12'] < df['ema_26']) & (df['ema_12'].shift() >= df['ema_26'].shift())
        
        # Bollinger Bands signals
        bb_buy = (df['close'] <= df['bb_lower'])
        bb_sell = (df['close'] >= df['bb_upper'])
        
        # Stochastic signals
        stoch_buy = (df['stoch_k'] < 20) & (df['stoch_d'] < 20)
        stoch_sell = (df['stoch_k'] > 80) & (df['stoch_d'] > 80)
        
        # Combine signals
        buy_signals = rsi_buy | macd_buy | ma_buy | bb_buy | stoch_buy
        sell_signals = rsi_sell | macd_sell | ma_sell | bb_sell | stoch_sell
        
        # Calculate signal strength
        buy_strength = (rsi_buy.astype(int) + macd_buy.astype(int) + ma_buy.astype(int) + 
                       bb_buy.astype(int) + stoch_buy.astype(int)) / 5
        sell_strength = (rsi_sell.astype(int) + macd_sell.astype(int) + ma_sell.astype(int) + 
                        bb_sell.astype(int) + stoch_sell.astype(int)) / 5
        
        # Set signals
        df.loc[buy_signals, 'signal'] = 1
        df.loc[sell_signals, 'signal'] = -1
        df.loc[buy_signals, 'signal_strength'] = buy_strength[buy_signals]
        df.loc[sell_signals, 'signal_strength'] = sell_strength[sell_signals]
        
        return df
    
    def analyze_trends(self, data: pd.DataFrame) -> Dict:
        """
        Analyze price trends
        
        Args:
            data: DataFrame with price data and indicators
            
        Returns:
            Dictionary with trend analysis
        """
        if len(data) < 50:
            return {}
        
        latest = data.iloc[-1]
        
        # Short-term trend (5-day)
        short_trend = "neutral"
        if latest['sma_5'] > latest['sma_10']:
            short_trend = "bullish"
        elif latest['sma_5'] < latest['sma_10']:
            short_trend = "bearish"
        
        # Medium-term trend (20-day)
        medium_trend = "neutral"
        if latest['sma_10'] > latest['sma_20']:
            medium_trend = "bullish"
        elif latest['sma_10'] < latest['sma_20']:
            medium_trend = "bearish"
        
        # Long-term trend (50-day)
        long_trend = "neutral"
        if latest['sma_20'] > latest['sma_50']:
            long_trend = "bullish"
        elif latest['sma_20'] < latest['sma_50']:
            long_trend = "bearish"
        
        # Overall trend strength
        trend_strength = 0
        if short_trend == medium_trend == long_trend:
            trend_strength = 3  # Strong trend
        elif short_trend == medium_trend or medium_trend == long_trend:
            trend_strength = 2  # Moderate trend
        else:
            trend_strength = 1  # Weak trend
        
        return {
            'short_term': short_trend,
            'medium_term': medium_trend,
            'long_term': long_trend,
            'strength': trend_strength,
            'current_price': latest['close'],
            'rsi': latest['rsi'],
            'macd': latest['macd'],
            'signal_line': latest['macd_signal'],
            'bb_position': latest['bb_position'],
            'volume_trend': 'high' if latest['volume'] > latest['volume_sma_20'] else 'low'
        }

# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Generate sample OHLCV data
    close = 100 + np.cumsum(np.random.randn(100) * 0.02)
    high = close + np.abs(np.random.randn(100) * 0.5)
    low = close - np.abs(np.random.randn(100) * 0.5)
    open_price = close.shift(1).fillna(close[0])
    volume = np.random.randint(1000, 10000, 100)
    
    sample_data = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    # Test technical indicators
    indicators = TechnicalIndicators()
    
    # Calculate all indicators
    data_with_indicators = indicators.calculate_all_indicators(sample_data)
    print("Technical indicators calculated:")
    print(data_with_indicators.columns.tolist())
    
    # Get trading signals
    data_with_signals = indicators.get_trading_signals(data_with_indicators)
    print("\nSample trading signals:")
    print(data_with_signals[['close', 'signal', 'signal_strength']].tail(10))
    
    # Analyze trends
    trend_analysis = indicators.analyze_trends(data_with_signals)
    print("\nTrend analysis:")
    print(trend_analysis)