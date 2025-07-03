from fastmcp import FastMCP

from lobster.server import app

mcp = FastMCP.from_fastapi(app=app)


def serve():
    mcp.run()
