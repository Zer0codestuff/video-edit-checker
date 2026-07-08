#!/usr/bin/env bash
cd "$(dirname "$0")"
if [ ! -x ".venv/bin/python" ]; then
    echo "Esegui prima l'installazione: python3 install.py"
    exit 1
fi
exec .venv/bin/python app.py
