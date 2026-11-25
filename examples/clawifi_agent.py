"""Run a LlamaIndex agent with Clawifi search and fetch.

Needs: pip install llama-index-tools-clawifi llama-index-llms-anthropic
Env: ANTHROPIC_API_KEY, and optionally CLAWIFI_API_KEY (keyless trial without it).
"""

import asyncio

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.anthropic import Anthropic

from llama_index.tools.clawifi import ClawifiToolSpec


async def main() -> None:
    agent = FunctionAgent(
        tools=ClawifiToolSpec(max_results=5).to_tool_list(),
        llm=Anthropic(model="claude-sonnet-5-5"),
        system_prompt="Search the web, read the best source, and cite its URL.",
    )
    print(await agent.run("What is the Model Context Protocol?"))


if __name__ == "__main__":
    asyncio.run(main())
