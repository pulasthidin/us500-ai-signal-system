# Contributing

This is a personal trading system. The repository is **read-only** for public access.

## Reporting Issues

If you find a bug or have a suggestion:

1. Open an [Issue](https://github.com/pulasthidin/us500-ai-signal-system/issues)
2. Include:
   - What you expected to happen
   - What actually happened
   - Steps to reproduce (if applicable)
   - Python version and OS

## Code Standards

This project follows these conventions:

- **Python 3.11+** with type hints on all function signatures
- **Every function** wrapped in try/except — the app must never crash
- **Every exception** logged with `exc_info=True` and sent to Telegram system channel
- **No hardcoded API keys** — all secrets loaded from `.env`
- **No UI code** — Telegram-only output
- **No Google Drive** — all files saved locally
- Config values in `config.py` only — no magic numbers in src modules
- SQLite writes use transactions (`with conn:`)
- Signal saved to database BEFORE Telegram alert fires

## Testing

```bash
python -m pytest tests/ -v
```

All 443 tests must pass before any change is accepted.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
