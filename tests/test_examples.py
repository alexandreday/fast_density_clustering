"""Integration tests — run each example script end-to-end and check exit code."""

import os
import subprocess
import sys

import pytest

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "example")


def _run(script_name: str, timeout: int = 120) -> subprocess.CompletedProcess:
    script_path = os.path.join(EXAMPLES_DIR, script_name)
    env = {**os.environ, "MPLBACKEND": "Agg"}
    return subprocess.run(
        [sys.executable, script_path],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=EXAMPLES_DIR,
    )


class TestExampleScripts:
    def test_example(self):
        result = _run("example.py")
        assert result.returncode == 0, (
            f"example.py exited with code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    def test_example3(self):
        result = _run("example3.py")
        assert result.returncode == 0, (
            f"example3.py exited with code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    def test_benchmark_sklearn(self):
        result = _run("benchmark_sklearn.py", timeout=300)
        assert result.returncode == 0, (
            f"benchmark_sklearn.py exited with code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        assert "ARI" in result.stdout
