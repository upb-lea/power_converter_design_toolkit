"""Classes and toml checker for the flow control."""
# python libraries
from typing import Literal

# 3rd party libraries

# own libraries
from pcdt.topology.circuit_optimization_base import TomlGData, TomlCData
from pcdt.circuit_enums import SamplingEnum
from pcdt.topology.dab.dab_enums import DabEnums

# ######################################################
# dab general
# ######################################################
class TomlDabOutputRange(TomlGData):
    """Definition of the DAB operating area."""

    v1_min_max_list: list[float]
    v2_min_max_list: list[float]
    p_min_max_list: list[float]

class TomlDabSampling(TomlGData):
    """Definition of the sampling method."""

    sampling_method: SamplingEnum
    sampling_points: int
    sampling_random_seed: int | Literal["random"]
    v1_additional_user_point_list: list[float]
    v2_additional_user_point_list: list[float]
    p_additional_user_point_list: list[float]
    additional_user_weighting_point_list: list[float]

class TomlDabGeneral(TomlGData):
    """Definition of the general parameters affecting mostly all kind of calculations."""

    output_range: TomlDabOutputRange
    sampling: TomlDabSampling

# ######################################################
# dab circuit
# ######################################################

class TomlDabCircuitParetoDesignSpace(TomlCData):
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

class TomlDabThermalResistanceData(TomlCData):
    """Toml checker for HeatSinkThermalResistanceData."""

    # [tim_thickness, tim_conductivity]
    transistor_b1_cooling: list[float]
    transistor_b2_cooling: list[float]

class TomlDabCircuitFilterDistance(TomlCData):
    """Toml checker class for CircuitFilterDistance."""

    filter_algorithm: Literal[
        DabEnums.sort_pareto_3_dimensional.value, DabEnums.sort_pareto_2_dimensional.value, DabEnums.sort_frequency.value, DabEnums.sort_lowest.value]
    number_filtered_designs: int
    difference_percentage: float

class TomlDabCircuitParetoDesign(TomlCData):
    """Config to optimize the Dual-Active Bridge (DAB) converter."""

    design_space: TomlDabCircuitParetoDesignSpace
    filter_distance: TomlDabCircuitFilterDistance
    thermal_data: TomlDabThermalResistanceData
