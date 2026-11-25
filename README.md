# LlamaIndex Tools Integration: Clawifi

`ClawifiToolSpec` gives a LlamaIndex agent web search and page reading through [Clawifi](https://clawifi.dev):

- `search(query, max_results=None, time_range=None)`: web search (`POST /search/free`); one `Document` per result, with the snippet as text and `url`, `title`, `rank` in metadata.
- `fetch(url, query=None)`: read a page or PDF (`POST /fetch`) as Markdown; with `query`, only the passages that answer it.

Both tools have async versions and wrap the official [`clawifi`](https://pypi.org/project/clawifi/) SDK.

Source: <https://github.com/ReARCLabs/clawifi-llamaindex>

## Install

```bash
pip install llama-index-tools-clawifi
```

## Credentials

Pass `api_key=` or set `CLAWIFI_API_KEY`. Without a key the tools still work, through the keyless trial below.

## Keyless trial

The Clawifi REST API needs a key. Without one, the tools call the trial tools of Clawifi's hosted MCP server instead ([Remote MCP](https://clawifi.dev/docs/remote-mcp)): `search_free` and `fetch_url` at `https://api.clawifi.dev/mcp`. The trial's limits, from the Clawifi docs:

- 12 calls a minute and 50 a day per address. On a shared host (serverless, CI) many apps can share one outbound address, so set `CLAWIFI_API_KEY` in production.
- Search returns at most 3 results, and domain and time filters are not applied.
- Fetched pages are cut at 20,000 characters; with `query`, up to 3 passages come back.
- Usually the day's limit comes back as HTTP 401 ("keyless limit reached") and the minute's limit as HTTP 429 with `Retry-After`.

A free account has 1,000 credits a month ([sign up](https://clawifi.dev/signup)). With a key, the tools use the REST API and your plan's limits.

Without a key, every search `Document` also carries a `keyless_note` in its metadata that says which limits applied. Fetch results have the same fields with or without a key. Trial errors are raised as `ClawifiKeylessError` (with `status` and `retry_after`), a subclass of `clawifi.ClawifiError`.

## Usage

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.anthropic import Anthropic
from llama_index.tools.clawifi import ClawifiToolSpec

tools = ClawifiToolSpec(max_results=5).to_tool_list()
agent = FunctionAgent(tools=tools, llm=Anthropic(model="claude-sonnet-5-5"))

response = await agent.run("What changed in the latest MCP specification?")
print(response)
```

Direct calls:

```python
spec = ClawifiToolSpec()
results = spec.search("Model Context Protocol", time_range="month")
page = spec.fetch(results[0].metadata["url"], query="who maintains it?")
print(page[0].text)
```

Options: `max_results` (default `5`), `max_chars` (default `20000`; longer pages are cut and marked `truncated` in metadata), `api_base`, and pre-built `client` / `async_client`.

Each call uses Clawifi credits; see [Pricing and credits](https://clawifi.dev/docs/pricing-and-credits). See `examples/clawifi_agent.py` for a runnable script.

## Development

```bash
pip install -e . pytest pytest-asyncio
pytest
```

The tests mock every HTTP call.

## License

MIT
