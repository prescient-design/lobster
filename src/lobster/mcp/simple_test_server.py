#!/usr/bin/env python3
"""
Simple test MCP server to debug the connection issues
"""

import asyncio

from mcp.server import Server
from mcp.types import CallToolRequest, CallToolResult, ListToolsResult, TextContent, Tool


def main():
    """Main entry point"""
    server = Server("simple-test")

    @server.list_tools()
    async def list_tools():
        return ListToolsResult(
            tools=[
                Tool(
                    name="hello",
                    description="Say hello",
                    inputSchema={
                        "type": "object",
                        "properties": {"name": {"type": "string", "description": "Name to greet"}},
                        "required": ["name"],
                    },
                )
            ]
        )

    @server.call_tool()
    async def call_tool(request: CallToolRequest):
        if request.params.name == "hello":
            name = request.params.arguments.get("name", "World")
            return CallToolResult(content=[TextContent(type="text", text=f"Hello, {name}!")])
        else:
            return CallToolResult(content=[TextContent(type="text", text="Unknown tool")], isError=True)

    async def run_server():
        from mcp.server.stdio import stdio_server

        async with stdio_server() as streams:
            await server.run(*streams)

    asyncio.run(run_server())


if __name__ == "__main__":
    main()
