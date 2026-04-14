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

## Data layout (Phase 2)

After `pip install -e .`, create `data/raw`, `data/processed`, and `data/manifests` under `DATA_DIR`:

```bash
sim init
sim info
```

## OpenFDA ingest (Phase 3)

Requires network access. Uses `OPENFDA_API_BASE` and optional `OPENFDA_API_KEY` / `OPEN_FDA_API_KEY`.

```bash
# One page (default), drug labels — writes under data/raw/openfda/… and a manifest
sim ingest openfda

# Narrow search; more pages (respect OpenFDA rate limits)
sim ingest openfda --search 'openfda.generic_name:"aspirin"' --limit-per-page 50 --max-pages 3
```

Other endpoints: `drug/event`, `drug/ndc`, etc. (see [open.fda.gov](https://open.fda.gov/apis/)).

If you see **ConnectTimeout** or **SSL handshake timed out**, the client cannot reach `api.fda.gov` (VPN, firewall, proxy, or WSL networking). Verify with:

`curl -vI 'https://api.fda.gov/drug/label.json?limit=1'`

If `curl` tries IPv6 first and prints **Network is unreachable** for those addresses, then connects on IPv4, that is normal on WSL. Set **`HTTP_FORCE_IPV4=1`** in `.env` so Python skips IPv6 for ingest.

Then try raising `HTTP_CONNECT_TIMEOUT` / `HTTP_READ_TIMEOUT` in `.env`, or fix network path.

If **`curl` times out with `0 bytes received`** after TLS (even with `curl -4`), your stack may be stalling on **HTTP/2**. Compare:

`curl -4 --http1.1 -vI --max-time 30 'https://api.fda.gov/drug/label.json?limit=1'`

The ingest client uses **HTTP/1.1 only** (`http2=False`), so it may succeed where `curl -I` hangs. Also try the same URL from **Windows** (PowerShell / `curl.exe`) to see if the issue is WSL-only.

### WSL2: `curl.exe` on Windows works, but WSL gets `0 bytes received` or hangs after TLS

That usually means **WSL’s virtual network path** is broken for large TLS/HTTP payloads (common with **VPN**, **strict firewall**, or **MTU black holes**), not your Python code.

Do these **in order** (stop when WSL `curl` returns `HTTP/1.1 200`):

1. **Mirrored networking (Windows 11 22H2+)** — makes WSL use a path closer to the Windows host (where `curl.exe` already works).

   - Copy the `[wsl2]` block from `wsl/wslconfig.example` into **`%USERPROFILE%\.wslconfig`** on Windows (file in your user folder, not inside this repo).
   - In **PowerShell**: `wsl --shutdown`, wait ~10s, open Ubuntu again.

2. **Lower MTU inside WSL** (classic fix when TCP+TLS succeed but no HTTP data):

   ```bash
   bash scripts/wsl_mtu.sh 1350
   ```

   If still bad, try `1280`. Re-run after each WSL reboot unless you automate the `ip link set … mtu` line (e.g. `sudo` + `/etc/rc.local` or a user systemd unit if you use systemd in WSL).

3. **VPN / firewall** — allow WSL or switch VPN to **full tunnel** / split-tunnel rules that include **vEthernet (WSL)**. Corporate proxies: set `HTTPS_PROXY` / `NO_PROXY` in `.env` if your Windows stack uses a proxy (httpx honors common proxy env vars when enabled).

4. **Keep ingest flags** — in `.env` use `HTTP_FORCE_IPV4=1` and reasonable `HTTP_CONNECT_TIMEOUT` / `HTTP_READ_TIMEOUT`.

After WSL `curl` works:

```bash
curl -4 --http1.1 -I --max-time 60 'https://api.fda.gov/drug/label.json?limit=1'
sim ingest openfda
```
