"""
XAI Grok Deep Research API client using direct API calls.
"""
import asyncio
import json
import logging
import random
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from loguru import logger
from config import XAIConfig
from xai_utils import XAIClient, async_chat_completion_text


class XAIResearchClient:
    """XAI Grok-based client for deep research analysis."""

    def __init__(self, config: XAIConfig):
        """
        Initialize XAI Research Client.

        Args:
            config: XAI configuration with API key and model settings
        """
        if not config:
            raise ValueError("XAIConfig is required")

        self.config = config
        self.client = XAIClient(config.api_key, config.model)
        self.max_retries = 3
        self.base_delay = 1.0  # seconds

        logger.info(f"Initialized XAI Research Client with model: {config.model}")

    async def research_event(self, event: Dict[str, Any], markets: List[Dict[str, Any]]) -> str:
        """
        Research an event and its markets using XAI Grok Deep Research.

        Args:
            event: Event information (title, subtitle, category, etc.)
            markets: List of markets within the event (without odds)

        Returns:
            Research response as a string, or error message if failed.
        """
        if not event or not markets:
            logger.warning("Invalid input: event or markets is empty")
            return "Error: Invalid input data for research"

        try:
            # Validate and prepare event data
            event_info = self._prepare_event_info(event)
            markets_info = self._prepare_markets_info(markets)
            prompt = self._build_research_prompt(event_info, markets_info)

            event_ticker = event.get('event_ticker', 'UNKNOWN')
            logger.info(f"Starting XAI deep research for event {event_ticker} with {len(markets)} markets")

            # Attempt research with retry logic
            response = await self._perform_research_with_retry(prompt, event_ticker)

            if not response:
                logger.error(f"No response from XAI for event {event_ticker}")
                return f"Error: No research response for event {event_ticker}"

            logger.info(f"Completed XAI deep research for event {event_ticker}")
            return response

        except Exception as e:
            event_ticker = event.get('event_ticker', 'UNKNOWN') if event else 'UNKNOWN'
            logger.exception(f"Error researching event {event_ticker} with XAI: {e}")
            return f"Error researching event with XAI: {str(e)}"

    def _prepare_event_info(self, event: Dict[str, Any]) -> str:
        """Prepare formatted event information for the prompt."""
        title = event.get('title', 'Unknown Event')
        subtitle = event.get('subtitle', '')
        mutually_exclusive = event.get('mutually_exclusive', False)

        event_info = f"Event: {title}\n"
        if subtitle:
            event_info += f"Subtitle: {subtitle}\n"
        event_info += f"Mutually Exclusive: {mutually_exclusive}\n"

        return event_info

    def _prepare_markets_info(self, markets: List[Dict[str, Any]]) -> str:
        """Prepare formatted markets information for the prompt."""
        markets_info = "Markets:\n"

        for i, market in enumerate(markets, 1):
            # Always include all markets for comprehensive analysis
            title = market.get('title', 'Unknown Market')
            ticker = market.get('ticker', '')
            subtitle = market.get('subtitle', '')
            open_time = market.get('open_time', '')
            close_time = market.get('close_time', '')

            markets_info += f"{i}. {title}"
            if ticker:
                markets_info += f" (Ticker: {ticker})"
            markets_info += "\n"

            if subtitle:
                markets_info += f"   Subtitle: {subtitle}\n"
            if open_time:
                markets_info += f"   Open: {open_time}\n"
            if close_time:
                markets_info += f"   Close: {close_time}\n"
            markets_info += "\n"

        return markets_info

    def _build_research_prompt(self, event_info: str, markets_info: str) -> str:
        """Build the comprehensive research prompt."""
        mutually_exclusive_note = (
            "If the markets within this event are mutually exclusive (e.g., only one outcome can occur, such as in election winner markets), "
            "ensure that the predicted probabilities for YES outcomes across all markets sum to approximately 100%. "
            "Otherwise, evaluate each market independently without any summing constraints, allowing for overlapping or independent probabilities."
        )
        
        return f"""
You are a prediction market expert specializing in Kalshi markets. Your goal is to provide accurate, evidence-based probability estimates to maximize trading gains. 
Research the event thoroughly using real-time data, historical precedents, expert opinions, and market sentiment. Focus on optimizing for high-confidence trades with positive expected value.

Current UTC time: {datetime.utcnow().isoformat()}

Event Information:
{event_info}

Markets Information:
{markets_info}

{mutually_exclusive_note}

Provide a structured response with the following sections:
1. Overall Event Analysis: Summarize key factors, historical context, and potential outcomes. Include quantitative data where available (e.g., polls, economic indicators).
2. Current Sentiment and News Analysis: Evaluate recent news, social media trends from X (formerly Twitter), and expert commentary. Quantify sentiment (e.g., positive/negative ratios) if possible.
3. Probability Predictions: For each market, predict the probability of the YES outcome (0-100%). Evaluate independently unless mutually exclusive. 
   Format: "Market Ticker (Full Name): XX% probability" (e.g., "KXTHEOPEN-25-DMCC (Market Description): 15%").
   Base predictions on empirical evidence, adjusting for biases in sources.
4. Confidence Levels: For each prediction, provide a confidence score (1-10, where 10 is highest) based on data quality and consensus.
5. Key Risks and Catalysts: List potential upside/downside risks, upcoming events (e.g., data releases, announcements), and their estimated impact on probabilities.
6. Trading Recommendations: For each market, suggest actions (e.g., Buy YES, Sell NO, Hold) with position sizing based on confidence and edge. 
   Calculate expected value (EV) assuming Kalshi's fee structure (e.g., EV = (probability * (1 - fee)) - ((1 - probability) * fee)). 
   Prioritize trades with EV > 0 and high Sharpe ratio analogs. Consider arbitrage opportunities across related markets.

Optimization Focus:
- Cross-reference multiple sources for robustness.
- Incorporate quantitative models (e.g., Bayesian updating with priors from historical Kalshi resolutions).
- Identify mispricings by comparing your probabilities to current Kalshi prices (if available in context).
- For diverse markets (e.g., economic, political, weather), tailor analysis: use polls for politics, models for weather, indicators for economics.
- Be precise, data-driven, and avoid speculation; substantiate all claims with citations where possible.

Ensure response is parsable: Use bullet points or numbered lists for sections 3-6, with clear market identifiers.
"""

    async def _perform_research_with_retry(self, prompt: str, event_ticker: str) -> Optional[str]:
        """Perform research with exponential backoff retry logic, enhanced for broader data sourcing."""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                today = datetime.now()
                # Dynamically adjust lookback period: 7 days for fast-moving events, up to 30 for longer-term contexts
                lookback_days = 30 if "election" in event_ticker.lower() or "economic" in event_ticker.lower() else 7
                from_date = (today - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

                messages = [{"role": "user", "content": prompt}]

                logger.info(f"Sending XAI research request for {event_ticker} (attempt {attempt + 1}) with {lookback_days}-day lookback")
                
                response_text = await async_chat_completion_text(
                    self.client,
                    model=self.config.model,
                    messages=messages,
                    enable_search=self.config.enable_search,
                    search_from_date=from_date,
                    search_to_date=today.strftime('%Y-%m-%d'),
                )

                if response_text:
                    logger.info(f"XAI research response received for {event_ticker}")
                    return response_text
                else:
                    raise RuntimeError("Empty response from XAI API")

            except Exception as e:
                last_error = str(e)
                wait_time = self.base_delay * (2 ** attempt) + random.uniform(0, 1)  # Add jitter to retries

                logger.warning(f"XAI research attempt {attempt + 1}/{self.max_retries} failed for {event_ticker}: {e}")
                logger.info(f"Retrying in {wait_time:.2f} seconds...")

                if attempt < self.max_retries - 1:
                    await asyncio.sleep(wait_time)

        logger.error(f"All {self.max_retries} XAI research attempts failed for {event_ticker}: {last_error}")
        return None

    async def close(self):
        """Close any resources (XAI client doesn't need explicit closing)."""
        # No resources to close for XAIClient, but method kept for interface compatibility
        logger.debug("XAI Research Client closed")
        return