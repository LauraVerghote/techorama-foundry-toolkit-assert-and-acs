"""Provision and deploy the final Bank Manager hosted agent in a new azd environment."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

MODEL_DEPLOYMENT_NAME = "gpt-5.6-terra"
MODEL_VERSION = "2026-07-09"
MODEL_SKU = "GlobalStandard"
MODEL_CAPACITY = 500
SERVICE_NAME = "bank-manager-agent"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subscription-id", required=True)
    parser.add_argument("--location", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--skip-login", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    return parser.parse_args()


def project_root() -> Path:
    root = Path(__file__).resolve().parents[1]
    required = ("azure.yaml", "hosted_agent_main.py", "requirements.txt")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"Incomplete repository clone; missing: {', '.join(missing)}")
    return root


def run(
    root: Path,
    command: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["AZURE_DEV_USER_AGENT"] = "microsoft_foundry_skill"
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    print(">", " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with return code {completed.returncode}: {' '.join(command)}"
        )
    return completed


def validate(root: Path) -> None:
    missing_tools = [tool for tool in ("azd", "pwsh") if shutil.which(tool) is None]
    if missing_tools:
        raise RuntimeError(
            "Install the missing local prerequisites: " + ", ".join(missing_tools)
        )

    required_files = (
        "scripts/prepare_hosted_opa.ps1",
        "bank_manager/hosted_agent/main.py",
        "bank_manager/agents/assert_optimized_instructions_acs/agent.py",
        "bank_manager/agents/assert_optimized_instructions_acs/bank_mcp_server.py",
        "bank_manager/agents/assert_optimized_instructions_acs/kb_mcp_server.py",
        "bank_manager/agents/assert_optimized_instructions_acs/acs_shim.py",
        "bank_manager/agents/assert_optimized_instructions_acs/manifest_feature.yaml",
        "bank_manager/agents/assert_optimized_instructions_acs/policy/bank_manager_feature.rego",
    )
    missing_files = [name for name in required_files if not (root / name).is_file()]
    if missing_files:
        raise RuntimeError(f"Incomplete repository clone; missing: {', '.join(missing_files)}")

    azure_yaml = (root / "azure.yaml").read_text(encoding="utf-8")
    expected = (
        "host: azure.ai.agent",
        "codeConfiguration:",
        f"name: {MODEL_DEPLOYMENT_NAME}",
        f'version: "{MODEL_VERSION}"',
        "entryPoint: hosted_agent_main.py",
        "dependencyResolution: remote_build",
    )
    missing_configuration = [value for value in expected if value not in azure_yaml]
    if missing_configuration:
        raise RuntimeError(
            "azure.yaml no longer matches this deployment helper; missing: "
            + ", ".join(missing_configuration)
        )

    hosted_source = (root / "bank_manager/hosted_agent/main.py").read_text(
        encoding="utf-8"
    )
    if "bank_manager.agents.assert_optimized_instructions_acs.agent" not in hosted_source:
        raise RuntimeError(
            "Hosted entry point is not wired to the optimized prompt + ACS/ECS runner"
        )

def deploy(args: argparse.Namespace) -> None:
    if args.confirm:
        required_values = {
            "subscription ID": args.subscription_id,
            "Azure location": args.location,
            "azd environment": args.environment,
        }
        placeholders = [
            name
            for name, value in required_values.items()
            if not value.strip() or "<" in value or "yourname" in value.lower()
        ]
        if placeholders:
            raise ValueError(
                "Replace the placeholder value(s) for: " + ", ".join(placeholders)
            )

    root = project_root()
    validate(root)
    print("Standalone hosted deployment")
    print(f" Subscription: {args.subscription_id}")
    print(f" Region: {args.location}")
    print(f" azd environment: {args.environment}")
    print(
        f" Runtime model: {MODEL_DEPLOYMENT_NAME} "
        f"({MODEL_VERSION}, {MODEL_SKU}, capacity {MODEL_CAPACITY})"
    )
    print(" Evaluation judge models: not required")
    print(" Implementation: optimized prompt + ACS/ECS")

    if not args.confirm:
        print("\nDry run only. Pass --confirm to create billable Azure resources.")
        return

    if not args.skip_login:
        run(root, ["azd", "auth", "login"])

    env_directory = root / ".azure" / args.environment
    if env_directory.is_dir():
        run(root, ["azd", "env", "select", args.environment])
    else:
        run(root, ["azd", "env", "new", args.environment, "--no-prompt"])

    values = (
        ("AZURE_SUBSCRIPTION_ID", args.subscription_id),
        ("AZURE_LOCATION", args.location),
        ("AZURE_AI_MODEL_DEPLOYMENT_NAME", MODEL_DEPLOYMENT_NAME),
    )
    for name, value in values:
        run(root, ["azd", "env", "set", name, value])

    run(root, ["azd", "provision", "--no-prompt"])
    run(root, ["pwsh", "-NoProfile", "-File", "scripts/prepare_hosted_opa.ps1"])
    run(root, ["azd", "deploy", SERVICE_NAME, "--no-prompt"])
    run(root, ["azd", "ai", "agent", "show", SERVICE_NAME, "--output", "json"])

    smoke_command = [
        "azd",
        "ai",
        "agent",
        "invoke",
        SERVICE_NAME,
        "Summarize the transfer approval steps without performing a transfer.",
    ]
    smoke_result = None
    for attempt in range(1, 4):
        print(f"Hosted smoke-test attempt {attempt}/3")
        smoke_result = run(root, smoke_command, check=False)
        if smoke_result.returncode == 0:
            break
        if attempt < 3:
            delay_seconds = attempt * 10
            print(f"Agent is not ready; retrying in {delay_seconds} seconds...")
            time.sleep(delay_seconds)

    if smoke_result is None or smoke_result.returncode != 0:
        run(
            root,
            ["azd", "ai", "agent", "monitor", SERVICE_NAME, "--tail", "100"],
            check=False,
        )
        raise RuntimeError("Deployment completed, but the hosted smoke test failed")

    print("Deployment complete. The hosted smoke test passed.")


if __name__ == "__main__":
    deploy(parse_args())
