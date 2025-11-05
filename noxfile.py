# noxfile.py
import nox

import time
from functools import wraps


nox.options.default_venv_backend = "uv"
nox.options.reuse_existing_virtualenvs = True


def timed_session(fn):
    @wraps(fn)
    def wrapper(session, *a, **kw):
        start = time.perf_counter()
        try:
            return fn(session, *a, **kw)
        finally:
            elapsed = time.perf_counter() - start
            session.log(f"Session {session.name or fn.__name__} duration: {elapsed:.2f}s")
    return wrapper


@nox.session(python=["3.9", "3.10", "3.11", "3.12", "3.13"])
@timed_session
def tests(session):
    """Run tests with pytest."""
    session.install(".")
    session.run("uv", "run", "pytest", "./tests", "--import-mode=importlib")
    session.run("uv", "run", "pytest", "--cov=./tests")


@nox.session(python="3.9")
@timed_session
def type_check(session):
    """Run type checking on specified files or all files by default."""
    session.install(".")
    session.run("uv", "run", "mypy")


@nox.session(python="3.9")
@timed_session
def lint(session):
    session.install("ruff")
    session.run("ruff", "check", "./src", "./tests")


@nox.session(python="3.9")
@timed_session
def format(session):
    session.install("ruff")
    session.run("ruff", "format", "./src", "./tests")
    session.run("ruff", "check", "--fix", "./src", "./tests")


@nox.session(python="3.9")
@timed_session
def format_unsafe(session):
    session.install("ruff")
    session.run("ruff", "format", "./src", "./tests")
    session.run("ruff", "check", "--fix", "--unsafe-fixes", "./src", "./tests")


@nox.session(python="3.9")
@timed_session
def check(session):
    """Run all code quality checks (lint + format)."""
    session.install("ruff")
    session.run("ruff", "check", "./src", "./tests")
    session.run("ruff", "format", "--check", "./src", "./tests")


@nox.session(python="3.13")
@timed_session
def python_compatibility(session):
    """Run Python version compatibility checks."""
    session.install("vermin")
    session.run("vermin", "-t=3.9-", "--violations", "--eval-annotations", "--backport", "typing", "--no-parse-comments", ".")


@nox.session(python="3.9")
@timed_session
def dependency_check(session):
    """Run dependency checks."""
    session.install("deptry")
    session.run("uv", "run", "deptry", "src")
