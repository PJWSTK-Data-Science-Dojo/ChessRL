"""WSGI entry point for production deployment."""

import os
from pathlib import Path

from src.web_app import app, LunaHtmlWrapper

# Initialize Luna engine
htmlWrap = LunaHtmlWrapper(
    verbose=os.getenv("VERBOSE", "true").lower() == "true",
    device=os.getenv("DEVICE", "cuda"),
    num_mcts_sims=int(os.getenv("MCTS_SIMS", "50")),
    checkpoint_dir=os.getenv("CHECKPOINT_DIR", "./temp/"),
    checkpoint_file=os.getenv("CHECKPOINT_FILE", "best.pth.tar"),
)

# Make htmlWrap available to routes
import src.web_app

src.web_app.htmlWrap = htmlWrap

if __name__ == "__main__":
    # For development, you can run this file directly
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("DEBUG", "false").lower() == "true",
    )
