#!/bin/bash
(cd src && pylint --rcfile=../.vscode/pylintrc enact)
(cd tests && pylint --rcfile=../.vscode/pylintrc tests)
