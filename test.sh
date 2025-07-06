#!/bin/bash
(cd tests && PYTHONPATH=:. uv run python -m unittest)
