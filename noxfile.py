# noxfile.py
import nox


@nox.session(python="3.10")  # (python=["3.8", "3.9", "3.10", "3.11"])
def tests(session):
    # session.run("poetry", "install", "--all-extras", external=True)
    session.run("pip", "install", "-e", "./[dev]")
    session.run("pytest", "./tests")
    session.run("pytest", "--cov=./tests")


@nox.session(python="3.10")
def type_check(session):
    session.install("mypy")
    session.install("pandas-stubs")
    session.install("types-requests")
    session.install("pytest-stub")
    session.install("types-colorama")
    session.install("types-Pygments")
    session.install("types-setuptools")
    session.run("mypy", "--install-types", "--non-interactive")
    session.run("mypy", "./src", "./tests")


@nox.session(python="3.10")
def lint(session):
    session.install("flake8")
    session.run("flake8", "./src", "./tests", "--max-line-length=127")


@nox.session(python="3.10")
def format(session):
    session.install("isort", "black")
    session.run("isort", "./src", "./tests")
    session.run("black", "./src", "./tests")
