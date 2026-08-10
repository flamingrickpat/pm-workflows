"""Workflow kernel.

The workflow manifest is the contract; this package executes it and nothing
more. Roles run one at a time in fresh sessions, declared checks decide the
routing, and every attempt is journalled outside the target repository.
"""
from .checkpoint import GitCheckpoint
from .drivers import (
    ClaudeDriver,
    CodexDriver,
    MinimalAgentDriver,
    PmCoderDriver,
    PiDriver,
    PythonDriver,
    build_driver,
)
from .journal import Journal
from .kernel import Kernel
from .manifest import ManifestError, StatePolicyConfig, Workflow, parse_workflow
from .extensions import PhaseExtensionRegistry, PhaseKindExtension
from .protocol import (
    AgentResult,
    GateResult,
    JournalEntry,
    PhaseConfig,
    RoleConfig,
    StepResult,
    WorkflowResolution,
)
from .python_role import RoleContext

__all__ = [
    "AgentResult",
    "ClaudeDriver",
    "CodexDriver",
    "GateResult",
    "GitCheckpoint",
    "Journal",
    "JournalEntry",
    "Kernel",
    "ManifestError",
    "MinimalAgentDriver",
    "PmCoderDriver",
    "PiDriver",
    "PhaseConfig",
    "PhaseExtensionRegistry",
    "PhaseKindExtension",
    "PythonDriver",
    "RoleConfig",
    "RoleContext",
    "StatePolicyConfig",
    "StepResult",
    "Workflow",
    "WorkflowResolution",
    "build_driver",
    "parse_workflow",
]
