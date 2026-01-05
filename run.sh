#!/bin/sh
uv run python -m lib.server.start -kd
uv run python -m lib.client.start -kd
