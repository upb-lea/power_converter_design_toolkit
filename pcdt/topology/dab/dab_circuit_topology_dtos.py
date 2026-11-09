"""Data transfer objects (DTOs) for the Pareto optimization."""
# Python libraries
import dataclasses

# 3rd party libraries

# own libraries
from pcdt.circuit_enums import SamplingEnum

@dataclasses.dataclass
class CircuitParetoDesignSpace:
    """Definition of the hardware design space for electronic components."""

    # DAB optimization parameters
    f_s_min_max_list: list[int]
    l_s_min_max_list: list[float]
    l_1_min_max_list: list[float]
    l_2__min_max_list: list[float]
    n_min_max_list: list[float]
    transistor_1_name_list: list[str]
    transistor_2_name_list: list[str]
    c_par_1: float
    c_par_2: float
    t_dead_1_max: float
    t_dead_2_max: float

@dataclasses.dataclass
class CircuitOutputRange:
    """Definition of the DAB operating area."""

    v1_min_max_list: list
    v2_min_max_list: list
    p_min_max_list: list

@dataclasses.dataclass
class CircuitSampling:
    """Definition of the sampling method."""

    sampling_method: SamplingEnum
    sampling_points: int
    sampling_random_seed: int | None
    v1_additional_user_point_list: list[float]
    v2_additional_user_point_list: list[float]
    p_additional_user_point_list: list[float]
    additional_user_weighting_point_list: list[float]

@dataclasses.dataclass
class CircuitFilter:
    """Filter the results."""

    filter_algorithm: str
    number_filtered_designs: int
    difference_percentage: float

@dataclasses.dataclass
class CircuitParetoDabDesign:
    """Config to optimize the Dual-Active Bridge (DAB) converter."""

    circuit_study_name: str
    project_directory: str

    design_space: CircuitParetoDesignSpace
    output_range: CircuitOutputRange
    sampling: CircuitSampling
    filter: CircuitFilter
