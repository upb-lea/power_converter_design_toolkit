"""Component requirements from the circuit class."""

# python libraries
import dataclasses

# 3rd party libraries
import numpy as np

@dataclasses.dataclass(init=False)
class CapacitorRequirements:
    """
    Requirements for a single capacitor design from the circuit results.

    The current waveform needs to be given for a full period:
     - time_vec (time based)
     - the current_vec gives the amplitude of the current at given time
    """

    # exact a single current waveform to optimize the capacitor
    time_vec: np.ndarray
    current_vec: np.ndarray

    # all current waveforms for calculation the capacitor loss for a single (optimized) capacitor
    time_array: np.ndarray
    current_array: np.ndarray

    # maximum dc link voltage
    v_dc_max: float

    # misc
    study_name: str
    circuit_id: str
    capacitor_number_in_circuit: int  # e.g. 0 is input capacitor, 1 is output capacitor, ...

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

@dataclasses.dataclass(init=False)
class InductorRequirements:
    """
    Requirements for a single inductor design from the circuit results.

    The current waveform needs to be given for a full period:
     - time_vec (time based)
     - the current_vec gives the amplitude of the current at given time
    """

    target_inductance: float

    # exact a single current waveform to optimize the inductor. Add tuple with operating point index
    time_vec: np.ndarray
    current_vec: np.ndarray
    vvp_index: tuple[int, int, int]

    # all current waveforms for calculation the inductor loss for a single (optimized) inductor
    time_array: np.ndarray
    current_array: np.ndarray

    # misc
    study_name: str
    circuit_id: str
    inductor_number_in_circuit: int  # e.g. 0 is series inductance, 1 is parallel inductance

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)


@dataclasses.dataclass(init=False)
class TransformerRequirements:
    """
    Requirements for a single transformer design from the circuit results.

    The current waveform needs to be given for a full period:
     - time_vec (time based)
     - the current_vec gives the amplitude of the current at given time
    """

    l_s12_target: float
    l_h_target: float
    n_target: float
    temperature: float

    # exact a single current waveform to optimize the transformer
    time_vec: np.ndarray
    current_1_vec: np.ndarray
    current_2_vec: np.ndarray
    vvp_index: tuple[int, int, int]

    # all current waveforms for calculation the transformer loss for a single (optimized) transformer
    time_array: np.ndarray
    current_1_array: np.ndarray
    current_2_array: np.ndarray

    # misc
    study_name: str
    circuit_id: str
    transformer_number_in_circuit: int  # e.g. 0 is input transformer, 1 is output transformer

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)


@dataclasses.dataclass(init=False)
class ComponentRequirements:
    """DTO contains the requirements for all the additional DAB components."""

    capacitor_requirements: list[CapacitorRequirements]
    inductor_requirements: list[InductorRequirements]
    transformer_requirements: list[TransformerRequirements]

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)


@dataclasses.dataclass
class ComponentCooling:
    """Fix parameters for the transistor, inductor and transformer cooling."""

    tim_thickness: float
    tim_conductivity: float

@dataclasses.dataclass(init=False)
class CircuitThermal:
    """DTO contains the thermal circuit parameters for summarized [transistor_1, transistor_2]."""

    t_j_max: list[float]
    r_th_jhs: list[float]
    area: list[float]
    loss_array: list[np.ndarray]
    temperature_heat_sink_max_array: list[np.ndarray]

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

@dataclasses.dataclass(init=False)
class InductorResults:
    """DTO contains the inductor losses."""

    # identification
    circuit_id: str
    inductor_id: int
    inductor_number_in_circuit: int

    # pareto
    loss_array: np.ndarray
    winding_loss_array: np.ndarray
    core_loss_array: np.ndarray
    volume: float
    area_to_heat_sink: float
    r_th_ind_heat_sink: float
    inductance: float

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

@dataclasses.dataclass(init=False)
class StackedTransformerResults:
    """DTO contains the stacked transformer losses."""

    # identification
    circuit_id: str
    transformer_id: int
    transformer_number_in_circuit: int

    # pareto
    loss_array: np.ndarray
    core_loss_array: np.ndarray
    winding_1_loss_array: np.ndarray
    winding_2_loss_array: np.ndarray
    volume: float
    area_to_heat_sink: float
    r_th_xfmr_heat_sink: float
    l_s: float
    l_h: float
    n: float

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)
