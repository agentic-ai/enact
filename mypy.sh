#!/bin/bash
(cd src && uv run mypy enact --check-untyped-defs)
