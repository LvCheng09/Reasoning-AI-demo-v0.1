"""Public package interface for the main-part reasoning system."""

from .extractor import (
    InstructionExecutor,
    InstructionResult,
    MainPartEngine,
    MainPartExtractor,
    RuleStrategyCompiler,
    SchedulerInstruction,
    SchedulerNew,
    SchedulerNewState,
    SchedulerProgram,
    WorkingMemoryState,
    demo_main_part_extraction,
)
from .memory import *  # noqa: F401,F403
from .memory import __all__ as _memory_all
from .data_structure import *  # noqa: F401,F403
from .data_structure import __all__ as _model_all
from .frontend_working_memory_bridge import submit_frontend_state_to_working_memory
from .frontend_event_builder import FrontendEventBuilder
from .model_launcher import MainPartModelLauncher
from .rule_vector_encoder import RuleVectorEncoder
from .runtime_launcher import RuntimeLauncher
from .output_decoder import SemanticOutputDecoder, SemanticPrototypeIndex
from .output_service import SemanticAnswerService


def main() -> None:
    from .cli import main as cli_main

    cli_main()

__all__ = [
    *_model_all,
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
    "RuntimeLauncher",
    "SemanticOutputDecoder",
    "SemanticPrototypeIndex",
    "SemanticAnswerService",
    "MainPartModelLauncher",
    "FrontendEventBuilder",
    "RuleVectorEncoder",
    "submit_frontend_state_to_working_memory",
    "demo_main_part_extraction",
    "main",
    *_memory_all,
]
