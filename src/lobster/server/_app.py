import os
from typing import Literal

from fastmcp import FastMCP
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.applications import Starlette
from starlette.routing import Mount
from pydantic import BaseModel, Field

from ._esm import get_esm_cached, esm_aa_naturalness

DEFAULT_LOBSTER_ALLOW_ORIGINS = "http://localhost"

ALLOW_ORIGINS = os.getenv("LOBSTER_ALLOW_ORIGINS", DEFAULT_LOBSTER_ALLOW_ORIGINS).split(",")


class NaturalnessInput(BaseModel):
    sequence: str
    model_name: Literal[
        "facebook/esm2_t6_8M_UR50D",
        "facebook/esm2_t30_150M_UR50D",
        "facebook/esm2_t33_650M_UR50D",
    ] = Field(default="facebook/esm2_t6_8M_UR50D")


class NaturalnessOutput(BaseModel):
    logp: list[list[float]]
    wt_logp: list[float]
    naturalness: float
    encoded: list[float]


fastapi_app = FastAPI()


@fastapi_app.post("/naturalness")
def naturalness(input: NaturalnessInput) -> NaturalnessOutput:
    L = len(input.sequence)
    if not L > 0:
        return {
            "logp": [],
            "wt_logp": [],
            "naturalness": None,
            "encoded": [],
        }

    model, tokenizer = get_esm_cached(input.model_name)

    out = esm_aa_naturalness(sequence=input.sequence, model=model, tokenizer=tokenizer, batch_size=64)

    return out


mcp = FastMCP.from_fastapi(app=fastapi_app)
mcp_app = mcp.http_app(path="/mcp")

app = Starlette(
    routes=[
        Mount("/mcp-server", app=mcp_app),
        Mount("/api", app=fastapi_app),
    ],
    lifespan=mcp_app.lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
