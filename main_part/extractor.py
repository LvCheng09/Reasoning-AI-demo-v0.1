"""Compatibility exports for the main model launcher and scheduler types."""

from .data_structure import WorkingMemoryState
from .model_launcher import MainPartModelLauncher
from .scheduler_new import (
    InstructionExecutor,
    InstructionResult,
    RuleStrategyCompiler,
    SchedulerInstruction,
    SchedulerNew,
    SchedulerNewState,
    SchedulerProgram,
)

MainPartEngine = MainPartModelLauncher
MainPartExtractor = MainPartModelLauncher


def demo_main_part_extraction(input_text: str = "") -> dict:
    launcher = MainPartModelLauncher(auto_run=False)
    if input_text:
        launcher.submit_input({"input": input_text, "frontend_state": {"spans": [], "events": []}})
    return {
        "scheduler": launcher.scheduler.state.to_dict(),
        "working_memory": launcher.working_memory.to_dict(),
    }

__all__ = [
    "InstructionExecutor",
    "InstructionResult",
    "MainPartEngine",
    "MainPartExtractor",
    "RuleStrategyCompiler",
    "SchedulerInstruction",
    "SchedulerNew",
    "SchedulerNewState",
    "SchedulerProgram",
    "WorkingMemoryState",
    "demo_main_part_extraction",
]
