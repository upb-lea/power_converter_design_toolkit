"""Classes and toml checker for the flow control."""
# python libraries
from typing import Literal

# 3rd party libraries
from pydantic import BaseModel

# own libraries
from materialdatabase.meta.data_enums import Material, DataSource

# ######################################################
# debug
# ######################################################

class DebugGeneral(BaseModel):
    """Debug mode general information."""

    is_debug: bool

class DebugCircuit(BaseModel):
    """Debug information for the circuit."""

    is_waveform_validation: bool

class DebugCapacitor(BaseModel):
    """Debug information for the capacitor."""

    number_working_point_max: int

class DebugInductor(BaseModel):
    """Debug information for the inductor."""

    number_reluctance_working_point_max: int
    number_fem_working_point_max: int

class DebugTransformer(BaseModel):
    """Debug information for the transformer."""

    number_reluctance_working_point_max: int
    number_fem_working_point_max: int

class DebugDataGeneration(BaseModel):
    """Debug information for the data generation."""

    number_combinations_max: int

class Debug(BaseModel):
    """General information in debug configuration."""

    general: DebugGeneral
    circuit: DebugCircuit
    capacitor: DebugCapacitor
    inductor: DebugInductor
    transformer: DebugTransformer
    data_generation: DebugDataGeneration

# ######################################################
# flow control
# ######################################################

class General(BaseModel):
    """General flow control information."""

    project_directory: str
    topology: Literal['dab', 'sbc']

class Breakpoints(BaseModel):
    """Flow control breakpoints."""

    circuit_pareto: Literal['no', 'pause', 'stop']
    circuit_filtered: Literal['no', 'pause', 'stop']
    capacitor: Literal['no', 'pause', 'stop']
    inductor: Literal['no', 'pause', 'stop']
    transformer: Literal['no', 'pause', 'stop']
    heat_sink: Literal['no', 'pause', 'stop']
    pre_summary: Literal['no', 'pause', 'stop']
    summary: Literal['no', 'pause', 'stop']

class CondBreakpoints(BaseModel):
    """Flow control conditional breakpoints."""

    circuit: int
    inductor: int
    transformer: int
    heat_sink: int

class Circuit(BaseModel):
    """Flow control for the circuit."""

    number_of_trials: int
    calculation_mode: Literal['new', 'continue', 'skip']
    subdirectory: str

class Capacitor(BaseModel):
    """Flow control for the capacitor 1."""

    calculation_modes: list[Literal['new', 'skip']]
    subdirectory: str

class Inductor(BaseModel):
    """Flow control for the inductor."""

    numbers_of_trials: list[int]
    calculation_modes: list[Literal['new', 'continue', 'skip']]
    subdirectory: str

class Transformer(BaseModel):
    """Flow control for the transformer."""

    numbers_of_trials: list[int]
    calculation_modes: list[Literal['new', 'continue', 'skip']]
    subdirectory: str

class HeatSink(BaseModel):
    """Flow control for the heat sink."""

    number_of_trials: int
    calculation_mode: Literal['new', 'continue', 'skip', 'disc']
    subdirectory: str

class PreSummary(BaseModel):
    """Flow control for the pre-summary."""

    calculation_mode: Literal['new', 'skip']
    subdirectory: str

class Summary(BaseModel):
    """Flow control for the summary."""

    calculation_mode: Literal['new', 'skip']
    subdirectory: str

class DataGeneration(BaseModel):
    """Flow control for the data generation."""

    calculation_mode_manufacture: Literal['new', 'continue', 'skip_purge']
    calculation_mode_visualization: Literal['new', 'skip_purge']
    subdirectory: str

class ConfigurationDataFiles(BaseModel):
    """File paths to the configuration files."""

    topology_files: list[str]
    capacitor_configuration_files: list[str]
    inductor_configuration_files: list[str]
    transformer_configuration_files: list[str]
    heat_sink_configuration_file: str
    summary_configuration_file: str

class FlowControl(BaseModel):
    """General flow control class."""

    general: General
    breakpoints: Breakpoints
    conditional_breakpoints: CondBreakpoints
    circuit: Circuit
    capacitor: Capacitor
    inductor: Inductor
    transformer: Transformer
    heat_sink: HeatSink
    pre_summary: PreSummary
    summary: Summary
    data_generation: DataGeneration
    configuration_data_files: ConfigurationDataFiles

# ######################################################
# used by multiple components
# ######################################################

class TomlFilterDistance(BaseModel):
    """Toml checker class for FilterDistance."""

    factor_dc_losses_min_max_list: list[float]

# ######################################################
# capacitor
# ######################################################

class TomlCapacitorSelection(BaseModel):
    """Capacitor selection details."""

    maximum_peak_to_peak_voltage_ripple: float
    temperature_ambient: float
    voltage_safety_margin_percentage: float
    maximum_number_series_capacitors: int
    lifetime_h: float
    filter_distance: TomlFilterDistance

# ######################################################
# inductor and transformer
# ######################################################

class TomlThermalData(BaseModel):
    """Toml checker for HeatSinkThermalResistanceData."""

    # [tim_thickness, tim_conductivity]
    thermal_cooling: list[float]

# ######################################################
# inductor
# ######################################################

class TomlInductorDesignSpace(BaseModel):
    """Toml checker class for InductorDesignSpace."""

    core_name_list: list[str]
    material_name_list: list[Material]
    litz_wire_name_list: list[str]
    core_inner_diameter_min_max_list: list[float]
    window_h_min_max_list: list[float]
    window_w_min_max_list: list[float]

class TomlInductorSettings(BaseModel):
    """Toml checker class for InductorSettings."""

    fft_filter_value_factor: float
    mesh_accuracy: float

class TomlInductorInsulation(BaseModel):
    """Toml checker class for InductorInsulation."""

    primary_to_primary: float
    core_bot: float
    core_top: float
    core_right: float
    core_left: float

class TomlMaterialDataSources(BaseModel):
    """Toml checker class for MaterialDataSources."""

    permeability_datasource: DataSource
    permittivity_datasource: DataSource

class TomlInductorBoundaryConditions(BaseModel):
    """Toml checker class for InductorBoundaryConditions."""

    # Temperature
    temperature: float

class TomlInductor(BaseModel):
    """Toml checker class for Inductor."""

    design_space: TomlInductorDesignSpace
    insulations: TomlInductorInsulation
    boundary_conditions: TomlInductorBoundaryConditions
    thermal_data: TomlThermalData
    filter_distance: TomlFilterDistance
    settings: TomlInductorSettings
    material_data_sources: TomlMaterialDataSources

# ######################################################
# transformer
# ######################################################
class TomlTransformerDesignSpace(BaseModel):
    """Toml checker class for TransformerDesignSpace."""

    material_name_list: list[Material]
    core_name_list: list[str]
    core_inner_diameter_min_max_list: list[float]
    window_w_min_max_list: list[float]
    window_h_bot_min_max_list: list[float]
    primary_litz_wire_list: list[str]
    secondary_litz_wire_list: list[str]
    n_p_top_min_max_list: list[int]
    n_p_bot_min_max_list: list[int]

class TomlTransformerSettings(BaseModel):
    """Toml checker class for TransformerSettings."""

    fft_filter_value_factor: float
    mesh_accuracy: float

class TomlTransformerBoundaryConditions(BaseModel):
    """Toml checker class for TransformerBoundaryConditions."""

    max_transformer_total_height: float
    max_core_volume: float
    temperature: float

class TomlTransformerInsulation(BaseModel):
    """Toml checker class for TransformerInsulation."""

    # insulation for top core window
    iso_window_top_core_top: float
    iso_window_top_core_bot: float
    iso_window_top_core_left: float
    iso_window_top_core_right: float
    # insulation for bottom core window
    iso_window_bot_core_top: float
    iso_window_bot_core_bot: float
    iso_window_bot_core_left: float
    iso_window_bot_core_right: float
    # winding-to-winding insulation
    iso_primary_to_primary: float
    iso_secondary_to_secondary: float
    iso_primary_to_secondary: float

class TomlTransformerFilterDistance(BaseModel):
    """Toml checker class for TransformerFilterDistance."""

    factor_dc_losses_min_max_list: list[float]

class TomlTransformer(BaseModel):
    """Toml checker class for Transformer."""

    design_space: TomlTransformerDesignSpace
    insulation: TomlTransformerInsulation
    boundary_conditions: TomlTransformerBoundaryConditions
    thermal_data: TomlThermalData
    filter_distance: TomlTransformerFilterDistance
    settings: TomlTransformerSettings
    material_data_sources: TomlMaterialDataSources

# ######################################################
# heat sink
# ######################################################

class TomlHeatSinkBoundaryConditions(BaseModel):
    """Toml checker for HeatSinkBoundaryConditions."""

    t_ambient: float
    t_hs_max: float
    area_min: float

class TomlHeatSinkSettings(BaseModel):
    """Toml checker for HeatSinkSettings."""

    number_directions: int
    factor_pcb_area_copper_coin: float
    factor_bottom_area_copper_coin: float
    thermal_conductivity_copper: float

class TomlHeatSinkDesignSpace(BaseModel):
    """Toml checker for HeatSinkDesignSpace."""

    height_c_min_max_list: list[float]
    width_b_min_max_list: list[float]
    length_l_min_max_list: list[float]
    height_d_min_max_list: list[float]
    number_cooling_channels_n_min_max_list: list[int]
    thickness_fin_t_min_max_list: list[float]

class TomlHeatSink(BaseModel):
    """Toml checker for HeatSink."""

    design_space: TomlHeatSinkDesignSpace
    settings: TomlHeatSinkSettings
    boundary_conditions: TomlHeatSinkBoundaryConditions

# ######################################################
# misc
# ######################################################

class TomlMisc(BaseModel):
    """Data structure for Misc data."""

    min_efficiency_percent: float
    control_board_volume: float
    control_board_loss: float

# ######################################################
# summary
# ######################################################

class TomlSummaryDetail(BaseModel):
    """Toml checker class for SummaryDetail."""

    filter_distance_losses: list[float]
    filter_distance_efficiency: list[float]

class TomlSummary(BaseModel):
    """Toml checker class for Summary."""

    summary: TomlSummaryDetail
    pre_summary: TomlSummaryDetail
