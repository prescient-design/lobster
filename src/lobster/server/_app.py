import os
from typing import Literal
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ._esm import get_esm_cached, esm_aa_naturalness
from ._utils import SHARE_LOBSTER_PATH

DEFAULT_LOBSTER_ALLOW_ORIGINS = "*"

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


app = FastAPI()

app.mount(
    "/ui",
    StaticFiles(directory=Path(SHARE_LOBSTER_PATH, "ui")),
    name="ui",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/naturalness")
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


def serve():
    import uvicorn

    host = "localhost"
    port = 8000

    url = f"http://{host}:{port}/ui/index.html"
    print(f"visit {url}")

    uvicorn.run(app, host=host, port=port)
