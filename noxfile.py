import nox

PYTHON_VERSIONS = [
    "3.7",
    "3.8",
    "3.9",
    "3.10",
    "3.11",
    "3.12",
    "3.13",
]

@nox.session(python=PYTHON_VERSIONS)
def tests(session):
    session.install("-r", "requirements.txt")
    session.install("pytest", "ruff")
    session.run("ruff", "check", ".")
    session.run("pytest")
