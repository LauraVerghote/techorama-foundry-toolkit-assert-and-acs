import ast
import hashlib
import importlib
from pathlib import Path

import pytest

from bank_manager.agents.registry import (
    AGENT_REVISIONS,
    get_agent_revision,
)


EXPECTED_TOOLS = {
    "read_client",
    "read_account",
    "read_transaction_history",
    "prepare_transfer",
    "request_customer_approval",
    "create_transfer",
    "freeze_account",
    "enable_admin_mode",
    "read_loan",
    "prepare_loan_modification",
    "read_portfolio",
    "place_trade",
    "knowledge_base_retrieve",
}


@pytest.mark.parametrize(
    ("name", "controls", "deployment"),
    [
        ("baseline", "none", None),
        ("assert_optimized_instructions", "none", None),
        ("assert_optimized_instructions_acs", "acs", None),
        ("cheaper_model", "acs", "gpt-5.6-luna"),
    ],
)
def test_revision_packages_are_complete(
    name: str,
    controls: str,
    deployment: str | None,
) -> None:
    revision = get_agent_revision(name)
    agent_module = importlib.import_module(f"bank_manager.agents.{name}.agent")

    assert revision.name == name
    assert revision.controls == controls
    assert revision.default_deployment == deployment
    assert revision.instructions
    assert set(revision.tool_descriptions) == EXPECTED_TOOLS
    assert callable(agent_module.chat)
    assert callable(agent_module.run_async)
    package = Path(agent_module.PACKAGE_DIR)
    for required in (
        "agent.py",
        "bank_core.py",
        "bank_mcp_server.py",
        "kb_backend.py",
        "kb_mcp_server.py",
        "retrieval_trace.py",
    ):
        assert (package / required).is_file()
    assert len(list((package / "knowledge").glob("*.md"))) == 8

    if controls == "acs":
        manifest = Path(agent_module.ACS_MANIFEST)
        assert manifest.is_file()
        assert (manifest.parent / "policy" / "bank_manager_feature.rego").is_file()
        assert (package / "feature_policy.py").is_file()
        assert (package / "acs_shim.py").is_file()


def test_optimized_instructions_are_preserved_when_controls_or_model_change() -> None:
    optimized = get_agent_revision("assert_optimized_instructions")
    baseline = get_agent_revision("baseline")
    assert optimized.instructions != baseline.instructions
    for name in ("assert_optimized_instructions_acs", "cheaper_model"):
        revision = get_agent_revision(name)
        assert revision.instructions == optimized.instructions
        assert revision.tool_descriptions == optimized.tool_descriptions


def test_cheaper_package_preserves_acs_runtime_with_evidence_only_instrumentation() -> None:
    controlled = importlib.import_module(
        "bank_manager.agents.assert_optimized_instructions_acs.agent"
    )
    cheaper = importlib.import_module("bank_manager.agents.cheaper_model.agent")
    primary_source = Path(controlled.__file__).read_text(encoding="utf-8")
    cheaper_source = Path(cheaper.__file__).read_text(encoding="utf-8")
    expected_source = primary_source.replace(
        "AGENT_NAME = 'assert_optimized_instructions_acs'", "AGENT_NAME = 'cheaper_model'"
    ).replace("DEFAULT_DEPLOYMENT = None", "DEFAULT_DEPLOYMENT = 'gpt-5.6-luna'")
    assert ast.dump(ast.parse(cheaper_source)) == ast.dump(ast.parse(expected_source))
    primary = Path(controlled.PACKAGE_DIR)
    alternate = Path(cheaper.PACKAGE_DIR)
    for path in primary.rglob("*"):
        relative = path.relative_to(primary)
        if not path.is_file() or "__pycache__" in relative.parts:
            continue
        if relative.as_posix() in ("agent.py", "__init__.py"):
            continue
        assert (alternate / relative).read_bytes() == path.read_bytes(), relative


def test_registry_lists_four_sequential_revisions() -> None:
    assert AGENT_REVISIONS == (
        "baseline",
        "assert_optimized_instructions",
        "assert_optimized_instructions_acs",
        "cheaper_model",
    )


def test_revision_name_must_match_folder() -> None:
    with pytest.raises(ValueError, match="Unknown Bank Manager revision"):
        get_agent_revision("not_a_revision")


@pytest.mark.parametrize("name", AGENT_REVISIONS)
def test_every_revision_exposes_one_agent_factory(
    name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AZURE_API_BASE", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_API_KEY", "test-key")
    module = importlib.import_module(f"bank_manager.agents.{name}.agent")
    assert callable(module.create_agent)
    assert callable(module.create_agent().run)


@pytest.mark.parametrize("name", AGENT_REVISIONS)
def test_package_does_not_import_shared_bank_manager_runtime(name: str) -> None:
    package = Path(
        importlib.import_module(f"bank_manager.agents.{name}.agent").PACKAGE_DIR
    )
    for path in package.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "bank_manager.agent_baseline" not in source
        assert "bank_manager.agent_revisions" not in source
        assert "bank_manager.runtime" not in source


def test_tool_and_knowledge_snapshots_match_across_revisions() -> None:
    snapshot_paths = (
        "bank_core.py",
        "bank_mcp_server.py",
        "kb_backend.py",
        "kb_mcp_server.py",
        "retrieval_trace.py",
        "knowledge/brokerage-policy.md",
        "knowledge/fees-and-disputes.md",
        "knowledge/home-loan-policy.md",
        "knowledge/kyc-aml.md",
        "knowledge/sensitive-account-access.md",
        "knowledge/transfers-and-approvals.md",
        "knowledge/vendor-onboarding-notes.md",
    )
    expected: dict[str, str] = {}
    for name in AGENT_REVISIONS:
        package = Path(
            importlib.import_module(f"bank_manager.agents.{name}.agent").PACKAGE_DIR
        )
        hashes = {
            relative: hashlib.sha256((package / relative).read_bytes()).hexdigest()
            for relative in snapshot_paths
        }
        if not expected:
            expected = hashes
        else:
            assert hashes == expected
