"""
Astra Experiment Lifecycle & State Machine (Phase 6)
Enforces valid transitions and immutability boundaries across the lifecycle:
  DRAFT -> VALIDATED -> RUNNING -> COMPLETED / FAILED / CANCELLED
"""

from enum import Enum
from typing import Set


class ExperimentState(str, Enum):
    DRAFT = "DRAFT"
    VALIDATED = "VALIDATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    @property
    def is_locked(self) -> bool:
        """Returns True if the scientific identity of the experiment is locked and immutable."""
        return self in (
            ExperimentState.VALIDATED,
            ExperimentState.RUNNING,
            ExperimentState.COMPLETED,
            ExperimentState.FAILED,
            ExperimentState.CANCELLED,
        )

    @property
    def is_terminal(self) -> bool:
        """Returns True if the experiment has concluded and cannot resume execution."""
        return self in (
            ExperimentState.COMPLETED,
            ExperimentState.FAILED,
            ExperimentState.CANCELLED,
        )

    @property
    def is_training_eligible(self) -> bool:
        """Returns True if the experiment can pass the training gate to start/resume execution."""
        return self in (
            ExperimentState.VALIDATED,
            ExperimentState.RUNNING,
        )


class StateTransitionError(Exception):
    """Raised when an invalid lifecycle state transition is attempted."""
    pass


ALLOWED_TRANSITIONS: dict[ExperimentState, Set[ExperimentState]] = {
    ExperimentState.DRAFT: {
        ExperimentState.VALIDATED,
        ExperimentState.CANCELLED,
    },
    ExperimentState.VALIDATED: {
        ExperimentState.RUNNING,
        ExperimentState.CANCELLED,
    },
    ExperimentState.RUNNING: {
        ExperimentState.COMPLETED,
        ExperimentState.FAILED,
        ExperimentState.CANCELLED,
    },
    ExperimentState.COMPLETED: set(),
    ExperimentState.FAILED: set(),
    ExperimentState.CANCELLED: set(),
}


def validate_transition(current_state: ExperimentState, new_state: ExperimentState) -> None:
    """
    Validates that a transition from current_state to new_state is scientifically permitted.
    Raises StateTransitionError if the transition violates the state machine.
    """
    if current_state == new_state:
        return

    allowed = ALLOWED_TRANSITIONS.get(current_state, set())
    if new_state not in allowed:
        raise StateTransitionError(
            f"Illegal experiment state transition: cannot move from {current_state.value} to {new_state.value}. "
            f"Allowed transitions from {current_state.value}: {[s.value for s in allowed]}"
        )
