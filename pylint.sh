#!/bin/bash
(cd src && uv run pylint --rcfile=../.vscode/pylintrc enact)
(cd tests && uv run pylint --rcfile=../.vscode/pylintrc tests)
