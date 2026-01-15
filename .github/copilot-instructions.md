# Kalshi Deep Trading Bot - AI Coding Guidelines

## Architecture Overview

This is a linear workflow trading bot with 9 distinct steps orchestrated by `SimpleTradingBot.run()`:

1. **Fetch Events** - Get top events by volume from Kalshi API
2. **Get Markets** - Retrieve markets for each event (top 10 by volume)
3. **Research Events** - Batch process with Octagon Deep Research API
4. **Extract Probabilities** - Parse research into structured probabilities
5. **Get Market Odds** - Fetch current bid/ask prices
6. **Generate Decisions** - Use XAI/Grok for betting decisions with risk metrics
7. **Save Decisions** - Export to CSV with full context
8. **Place Bets** - Execute orders via Kalshi API
9. **Cleanup** - Close API connections

**Data Flow**: Events → Markets → Research → Probabilities → Odds → Decisions → Bets

## Key Components

- **`SimpleTradingBot`** - Main orchestration class in `trading_bot.py`
- **`KalshiClient`** - RSA-signed API client for market data and orders
- **`OctagonClient`** - Research API for event analysis
- **`XAIClient`** - Structured decision making with Pydantic models
- **Configuration** - Pydantic-based settings in `config.py`

## Critical Patterns

### Configuration Management
Use Pydantic `BaseSettings` with environment variables. Always call `load_config()` for settings:

```python
from config import load_config
config = load_config()
# Access: config.kalshi.api_key, config.max_bet_amount, etc.
```

### API Client Initialization
Always initialize and cleanup clients properly:

```python
# Initialize
client = KalshiClient(config.kalshi, ...)
await client.login()

# Cleanup
await client.close()
```

### Structured Outputs
Use `async_chat_completion_parse_pydantic` for XAI/Grok with Pydantic models:

```python
from xai_utils import async_chat_completion_parse_pydantic
from betting_models import BettingDecision

result = await async_chat_completion_parse_pydantic(
    client=xai_client,
    messages=[...],
    response_format=BettingDecision
)
```

### Risk-Adjusted Trading
Calculate hedge-fund style metrics in `calculate_risk_adjusted_metrics()`:
- **Expected Return**: `(p - y) / y`
- **R-Score**: `(p - y) / sqrt(p*(1-p))` (z-score of edge)
- **Kelly Fraction**: Optimal position sizing

### Logging
Use loguru with structured logging. Key patterns:
- `logger.info()` for major steps
- `logger.debug()` for detailed data
- Separate log files: `trading_bot.log`, `errors.log`, `trades.log`

### Error Handling
Comprehensive try/catch with specific error types. Always log exceptions:

```python
try:
    # risky operation
except Exception as e:
    logger.exception(f"Operation failed: {e}")
    raise
```

## Development Workflow

### Package Management
Use `uv` for all Python operations:
```bash
uv sync                    # Install dependencies
uv run trading-bot         # Run in dry-run mode
uv run trading-bot --live  # Live trading
uv run pytest              # Run tests
uv run black .             # Format code
```

### Testing Strategy
- **Demo Environment**: `KALSHI_USE_DEMO=true` for safe testing
- **Dry Run Mode**: Default behavior, simulates all operations
- **Live Trading**: `--live` flag places real bets

### Environment Setup
Copy `env_template.txt` to `.env` and configure:
- `KALSHI_API_KEY` + `KALSHI_PRIVATE_KEY` (RSA PEM format)
- `OCTAGON_API_KEY`
- `XAI_API_KEY` (or `OPENAI_API_KEY`)

## Code Organization

### File Structure
- `trading_bot.py` - Main bot logic and CLI
- `config.py` - Pydantic configuration models
- `kalshi_client.py` - Kalshi API integration
- `research_client.py` - Octagon research client
- `betting_models.py` - Pydantic data models
- `xai_utils.py` - XAI/Grok utilities

### Model Patterns
Use Pydantic `BaseModel` with field descriptions:
```python
class BettingDecision(BaseModel):
    ticker: str = Field(..., description="Market ticker")
    action: Literal["buy_yes", "buy_no", "skip"] = Field(...)
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
```

### Async/Await
All API operations are async. Use `asyncio.run()` in CLI entry points.

## Risk Management

### Hedging
Automatic hedge generation when `config.enable_hedging=True`:
- Hedge ratio: `config.hedge_ratio` (default 0.25)
- Min confidence threshold: `config.min_confidence_for_hedging`
- Max hedge amount: `config.max_hedge_amount`

### Position Limits
- `MAX_BET_AMOUNT` per market
- `SKIP_EXISTING_POSITIONS` to avoid duplicates
- R-score filtering with `z_threshold`

### Safety Features
- Demo environment for testing
- Dry-run mode validation
- Comprehensive logging and error recovery

## External Dependencies

### Kalshi API
- RSA signature authentication
- Rate limiting: Respect API limits
- Market filtering: Use `max_close_ts` for expiration filtering

### Octagon Deep Research
- Batch processing with `research_batch_size`
- Timeout handling: `research_timeout_seconds`
- Event + market analysis without odds

### XAI/Grok API
- Structured outputs with Pydantic
- Web search integration when `enable_search=True`
- Model selection: `grok-4-latest` default

## Common Pitfalls

- **API Keys**: Always validate PEM format for Kalshi private keys
- **Rate Limits**: Use batching and delays for research requests
- **Market Data**: Verify close times and status before trading
- **Async Cleanup**: Always close client connections in finally blocks
- **Configuration**: Override `dry_run` via CLI, not config file

## Testing Commands

```bash
# Safe testing
uv run trading-bot

# With expiration filter
uv run trading-bot --max-expiration-hours 6

# Live trading (use caution)
uv run trading-bot --live --max-expiration-hours 12
```

Reference: `README.md` for detailed setup and `config.py` for all available settings.