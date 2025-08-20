# noxfile.py
import nox


@nox.session(python="3.12")
def tests(session):
    """Run tests with pytest."""
    session.run("pip", "install", "-e", ".")
    session.run("pytest", "./tests")
    session.run("pytest", "--cov=./tests")


@nox.session(python="3.12")
def type_check(session):
    """Run type checking on specified files or all files by default."""
    session.run("pip", "install", "-e", ".")    
    session.run("mypy")


@nox.session(python="3.12")
def lint(session):
    session.install("ruff")
    session.run("ruff", "check", "./src", "./tests")


@nox.session(python="3.12")
def format(session):
    session.install("ruff")
    session.run("ruff", "format", "./src", "./tests")
    session.run("ruff", "check", "--fix", "./src", "./tests")


@nox.session(python="3.12")
def format_unsafe(session):
    session.install("ruff")
    session.run("ruff", "format", "./src", "./tests")
    session.run("ruff", "check", "--fix", "--unsafe-fixes", "./src", "./tests")


@nox.session(python="3.12")
def check(session):
    """Run all code quality checks (lint + format)."""
    session.install("ruff")
    session.run("ruff", "check", "./src", "./tests")
    session.run("ruff", "format", "--check", "./src", "./tests")
