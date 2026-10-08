"""Chay: python -m backend.api [--host ...] [--port ...] [--no-reload]."""
import argparse

import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chay RAG Chatbot API + Frontend")
    parser.add_argument("--host", default="0.0.0.0", help="Host (mac dinh 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (mac dinh 8000)")
    parser.add_argument("--no-reload", action="store_true", help="Tat reload")
    args = parser.parse_args()
    uvicorn.run("backend.api:app", host=args.host, port=args.port, reload=not args.no_reload)
