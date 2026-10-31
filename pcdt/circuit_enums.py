"""Enumerations for the circuit optimization."""
import enum


class SamplingEnum(enum.Enum):
    """Enum for the different sampling classes."""

    meshgrid = "meshgrid"
    latin_hypercube = "latin_hypercube"
    dessca = "dessca"

class TopologyEnum(enum.Enum):
    """Enum for available topologies."""

    # Dual active bridge
    dab = "dab"
    # Synchronous buck converter
    sbc = "sbc"


class TemplateType(enum.Enum):
    """Enum for available topologies."""

    # Template for circuit configuration
    circuit = "circuit"
    # Template for general configuration
    general = "general"

class CalcModeEnum(enum.Enum):
    """Enum for calculation mode."""

    # Calculate new
    new_mode = "new"
    # Continue calculation
    continue_mode = "continue"
    # skip calculation and use actual data
    skip_mode = "skip"
    # Copy the data from special folder (Only for heat sink)
    disc_mode = "disc"
    # Skip the process, but delete the output to keep the data integrity (Only for data generation)
    skip_purge_mode = "skip_purge"
