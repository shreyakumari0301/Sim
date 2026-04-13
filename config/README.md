# Configuration reference

## Portal URLs (keys and accounts)

| Variable | Purpose |
|----------|---------|
| `OPEN_FDA_SITE` | OpenFDA: create or manage API keys. |
| `NCBI_SITE` | NCBI account settings (API key for E-utilities). |

These are **not** data endpoints. Ingestion code should use the API bases below.

## API bases (machine endpoints)

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENFDA_API_BASE` | `https://api.fda.gov` | OpenFDA REST API root. |
| `NCBI_EUTILS_BASE` | `https://eutils.ncbi.nlm.nih.gov/entrez/eutils` | NCBI E-utilities base URL. |

## Keys

| Variable | Required? | Notes |
|----------|-----------|--------|
| `OPENFDA_API_KEY` | No | Improves rate limits; can be empty for development. |
| `NCBI_API_KEY` | No | Improves E-utilities throughput; respect [usage guidelines](https://www.ncbi.nlm.nih.gov/books/NBK25497/). |

`OPEN_FDA_API_KEY` is accepted as an alias for `OPENFDA_API_KEY` (same value).

## Copying environment

```bash
cp .env.example .env
# Edit .env; never commit it.
```

Load settings in Python via `sim.settings.get_settings()`.
