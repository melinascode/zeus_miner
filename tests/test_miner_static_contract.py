from __future__ import annotations

import ast
from pathlib import Path


def test_miner_routes_by_variable_horizon_and_cycle() -> None:
    source = Path("neurons/miner.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "synapse.variable" in source
    assert "synapse.requested_hours" in source
    assert "synapse.start_time" in source
    assert "load_artifact" in source
    assert "expected_hotkey=self.hotkey" in source
    assert "compressed_forecast_49hours" not in source
    assert "compressed_forecast_15days" not in source
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "on_challenge_block"
        for node in ast.walk(tree)
    )
