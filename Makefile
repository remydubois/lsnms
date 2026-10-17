.PHONY: install test test-python test-rust format format-check docs docs-rust

# Build the python package and the Rust extension
install:
	poetry install
	poetry run maturin develop --release

test: test-python test-Rust

test-python:
	poetry run pytest

test-rust:
	cargo test

format:
	poetry run ruff format .
	poetry run ruff check --fix .
	cargo fmt

format-check:
	poetry run ruff format --check .
	poetry run ruff check .
	cargo fmt --check

docs: docs-rust docs-python

docs-python:
	poetry run sphinx-build -b html docs docs/_build

docs-rust:
	cargo doc --no-deps