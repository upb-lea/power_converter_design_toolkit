"""Inductor optimization class."""
# python libraries
import os
import copy
import threading
import logging

# 3rd party libraries

# own libraries
import hct
import pcdt
from pcdt.boundary_check import CheckCondition as c_flag
from pcdt.server_ctl_dtos import ProgressData
from pcdt.server_ctl_dtos import ProgressStatus
from pcdt.server_ctl_dtos import RunTimeMeasurement as RunTime
from pcdt.components.component_dtos import ComponentCooling
from pcdt.components.heat_sink_dtos import *
from pcdt.constant_path import HEAT_SINK_CONF_TOML

logger = logging.getLogger(__name__)

class HeatSinkOptimization:
    """Optimization support class for heat sink optimization."""

    # Simulation configuration list
    _hct_config: hct.OptimizationParameters
    _progress_data: ProgressData
    _progress_run_time: RunTime

    def __init__(self) -> None:
        """Initialize the configuration list for the heat sink optimizations."""
        self._hct_config: hct.OptimizationParameters | None = None
        self._progress_data: ProgressData = ProgressData(run_time=0.0, number_of_filtered_points=0,
                                                         progress_status=ProgressStatus.Idle)
        self._progress_run_time: RunTime = RunTime()
        self._h_lock_stat: threading.Lock = threading.Lock()

    @staticmethod
    def verify_optimization_parameter(toml_heat_sink: pcdt.TomlHeatSink) -> tuple[bool, str]:
        """Verify the input parameter ranges.

        :param toml_heat_sink: toml inductor configuration
        :type toml_heat_sink: pcdt.TomlInductor
        :return: True, if the configuration was consistent
        :rtype: bool
        """
        # Variable declaration
        inconsistency_report: str = ""
        is_consistent: bool = True
        toml_check_value_list: list[tuple[float, str]]
        toml_check_min_max_values_list: list[tuple[list[float], str]]

        # Design space parameter check
        group_name = "design_space"

        # Setup check value list
        toml_check_min_max_values_list = (
            [(toml_heat_sink.design_space.height_c_min_max_list, f"{group_name}: height_c_min_max_list"),
             (toml_heat_sink.design_space.width_b_min_max_list, f"{group_name}: width_b_min_max_list"),
             (toml_heat_sink.design_space.length_l_min_max_list, f"{group_name}: length_l_min_max_list"),
             (toml_heat_sink.design_space.height_d_min_max_list, f"{group_name}: height_d_min_max_list")])

        # Perform the boundary check
        is_check_passed, issue_report = pcdt.BoundaryCheck.check_float_min_max_values_list(
            0, 5, toml_check_min_max_values_list, c_flag.check_exclusive, c_flag.check_exclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Perform the boundary check for thickness_fin_t_min_max_list
        is_check_passed, issue_report = pcdt.BoundaryCheck.check_float_min_max_values(
            0, 0.1, toml_heat_sink.design_space.thickness_fin_t_min_max_list,
            f"{group_name}: thickness_fin_t_min_max_list", c_flag.check_exclusive, c_flag.check_exclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Convert min_max-list from integer to float values
        float_number_cooling_channels_n_min_max_list = pcdt.BoundaryCheck.convert_int_list_to_float_list(
            toml_heat_sink.design_space.number_cooling_channels_n_min_max_list)
        # Perform the boundary check for number_fins_n_min_max_list
        is_check_passed, issue_report = pcdt.BoundaryCheck.check_float_min_max_values(
            3, 100, float_number_cooling_channels_n_min_max_list, f"{group_name}: number_cooling_channels_n_min_max_list",
            c_flag.check_inclusive, c_flag.check_exclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Perform boundary condition check
        group_name = "boundary_condition"

        # Create a list of parameter to check
        toml_check_value_list = (
            [(toml_heat_sink.boundary_conditions.t_ambient, f"{group_name}: t_ambient"),
             (toml_heat_sink.boundary_conditions.t_hs_max, f"{group_name}: t_hs_max")])

        # Perform the boundary check for temperatures
        is_check_passed, issue_report = pcdt.BoundaryCheck.check_float_value_list(
            -40, 125, toml_check_value_list, c_flag.check_inclusive, c_flag.check_inclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Perform the boundary check for area_min
        is_check_passed, issue_report = pcdt.BoundaryCheck.check_float_value(
            0, 25, toml_heat_sink.boundary_conditions.area_min, f"{group_name}: area_min",
            c_flag.check_exclusive, c_flag.check_exclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Perform setting check
        group_name = "setting"
        # Perform the boundary check for number_directions
        is_check_passed, issue_report = pcdt.BoundaryCheck.check_float_value(
            2, 3, float(toml_heat_sink.settings.number_directions), f"{group_name}: number_directions",
            c_flag.check_inclusive, c_flag.check_inclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Create a list of parameter to check
        toml_check_value_list = (
            [(toml_heat_sink.settings.factor_pcb_area_copper_coin, f"{group_name}: factor_pcb_area_copper_coin"),
             (toml_heat_sink.settings.factor_bottom_area_copper_coin, f"{group_name}: factor_bottom_area_copper_coin")])

        # Perform the boundary check for temperatures
        is_check_passed, issue_report = pcdt.BoundaryCheck.check_float_value_list(
            0, 10, toml_check_value_list, c_flag.check_exclusive, c_flag.check_exclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Perform the boundary check for number_directions
        is_check_passed, issue_report = pcdt.BoundaryCheck.check_float_value(
            80, 200, toml_heat_sink.settings.thermal_conductivity_copper, f"{group_name}: thermal_conductivity_copper",
            c_flag.check_inclusive, c_flag.check_inclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        return is_consistent, inconsistency_report

    def initialize_heat_sink_optimization(self, toml_heat_sink: pcdt.TomlHeatSink, toml_prog_flow: pcdt.FlowControl) -> bool:
        """
        Initialize the configuration.

        :param toml_heat_sink: toml heat sink class
        :type toml_heat_sink: pcdt.TomlHeatSink
        :param toml_prog_flow: toml program flow class
        :type toml_prog_flow: pcdt.FlowControl
        :return: True, if the configuration was successful initialized
        :rtype: bool
        """
        is_list_generation_successful = False

        # Verify optimization parameter
        is_consistent, issue_report = pcdt.HeatSinkOptimization.verify_optimization_parameter(toml_heat_sink)
        if not is_consistent:
            raise ValueError(
                "Heat sink optimization parameter are inconsistent!\n",
                issue_report)

        # Create file path
        heat_sink_fan_datapath = os.path.join(os.path.dirname(hct.__file__), "data")

        # Check if path exists
        if not os.path.exists(heat_sink_fan_datapath):
            logger.info(f"Fan data path {heat_sink_fan_datapath} does not exists!")
        # Generate the fan-list
        for (_, _, file_name_list) in os.walk(heat_sink_fan_datapath):
            fan_list = file_name_list

        if not fan_list:
            logger.info(f"No fan design data found in path {heat_sink_fan_datapath}!")

        heat_sink_study_name = HEAT_SINK_CONF_TOML.replace(".toml", "")

        self._hct_config = hct.OptimizationParameters(

            # general parameters
            heat_sink_study_name=heat_sink_study_name,
            heat_sink_optimization_directory=os.path.join(toml_prog_flow.general.project_directory, toml_prog_flow.heat_sink.subdirectory,
                                                          HEAT_SINK_CONF_TOML.replace(".toml", "")),

            # geometry parameters
            height_c_min_max_list=toml_heat_sink.design_space.height_c_min_max_list,
            width_b_min_max_list=toml_heat_sink.design_space.width_b_min_max_list,
            length_l_min_max_list=toml_heat_sink.design_space.length_l_min_max_list,
            height_d_min_max_list=toml_heat_sink.design_space.height_d_min_max_list,
            number_cooling_channels_n_min_max_list=toml_heat_sink.design_space.number_cooling_channels_n_min_max_list,
            thickness_fin_t_min_max_list=toml_heat_sink.design_space.thickness_fin_t_min_max_list,
            fan_list=fan_list,

            # boundary conditions
            t_ambient=toml_heat_sink.boundary_conditions.t_ambient,
            area_min=toml_heat_sink.boundary_conditions.area_min,

            # constraints
            number_directions=toml_heat_sink.settings.number_directions
        )

        is_list_generation_successful = True

        return is_list_generation_successful

    def get_progress_data(self) -> ProgressData:
        """Provide the progress data of the optimization.

        :return: Progress data: Processing start time, actual processing time and status.
                 number of filtered heat sink Pareto front points are obsolete
        :rtype: ProgressData
        """
        # Lock statistical performance data access
        with self._h_lock_stat:
            # Check if list is in progress
            if self._progress_data.progress_status == ProgressStatus.InProgress:
                # Update statistical data if optimization is running
                self._progress_data.run_time = self._progress_run_time.get_runtime()

        return copy.deepcopy(self._progress_data)

    # Simulation handler. Later the simulation handler starts a process per list entry.
    @staticmethod
    def _optimize(act_hct_config: hct.OptimizationParameters, target_number_trials: int, debug: bool) -> None:
        """
        Perform the simulation.

        :param target_number_trials: Number of trials for the optimization
        :type  target_number_trials: int
        :param debug: Debug mode flag
        :type debug: bool
        """
        # Check number of trials
        if target_number_trials > 0:
            hct.Optimization.start_proceed_study(config=act_hct_config, number_trials=target_number_trials)
        else:
            logger.info(f"Target number of trials = {target_number_trials} which are less equal 0!. No simulation is performed")

    # Optimization handler
    def optimization_handler(self, target_number_trials: int, debug: bool = False) -> None:
        """
        Control the multi simulation processes.

        :param target_number_trials: Number of trials for the optimization
        :type  target_number_trials: int
        :param debug: Debug mode flag
        :type  debug: bool
        """
        # Start the progress time measurement
        with self._h_lock_stat:
            self._progress_run_time.reset_start_trigger()
            self._progress_data.run_time = self._progress_run_time.get_runtime()
            self._progress_data.progress_status = ProgressStatus.InProgress

        # Perform optimization
        # Debug switch
        if target_number_trials != 0:
            if debug:
                # overwrite input number of trials with 100 for short simulation times
                if target_number_trials > 100:
                    target_number_trials = 100

        if self._hct_config is not None:
            HeatSinkOptimization._optimize(self._hct_config, target_number_trials, debug)
        else:
            logger.warning("Method 'initialize_heat_sink_optimization' is not called!\n"
                           "    No list is generated so that no optimization can be performed!")

        # Update statistical data
        with self._h_lock_stat:
            self._progress_run_time.stop_trigger()
            self._progress_data.run_time = self._progress_run_time.get_runtime()
            self._progress_data.progress_status = ProgressStatus.Done

class ThermalCalcSupport:
    """Provides functions to calculate the thermal resistance."""

    @staticmethod
    def calculate_r_th_copper_coin(cooling_area: float, height_pcb: float = 1.55e-3,
                                   height_pcb_heat_sink: float = 3.0e-3) -> tuple[float, float]:
        """
        Calculate the thermal resistance of the copper coin.

        Assumptions are made with some geometry factors from a real copper coin for TO263 housing.
        :param cooling_area: cooling area in m²
        :type cooling_area: float
        :param height_pcb: PCB thickness, e.g. 1.55 mm
        :type height_pcb: float
        :param height_pcb_heat_sink: Distance from PCB to heat sink in m
        :type height_pcb_heat_sink: float
        :return: r_th_copper_coin, effective_copper_coin_cooling_area
        :rtype: tuple[float, float]
        """
        factor_pcb_area_copper_coin = 1.42
        factor_bottom_area_copper_coin = 0.39
        thermal_conductivity_copper = 136  # W/(m*K)

        effective_pcb_cooling_area = cooling_area / factor_pcb_area_copper_coin
        effective_bottom_cooling_area = effective_pcb_cooling_area / factor_bottom_area_copper_coin

        r_pcb = 1 / thermal_conductivity_copper * height_pcb / effective_pcb_cooling_area
        r_bottom = 1 / thermal_conductivity_copper * height_pcb_heat_sink / effective_bottom_cooling_area

        r_copper_coin = r_pcb + r_bottom

        return r_copper_coin, effective_bottom_cooling_area

    @staticmethod
    def calculate_r_th_tim(copper_coin_bot_area: float, component_cooling: ComponentCooling) -> float:
        """
        Calculate the thermal resistance of the thermal interface material (TIM).

        :param copper_coin_bot_area: bottom area in m²
        :type copper_coin_bot_area: float
        :param component_cooling: Thermal material data: Specific thermal conductance and thickness
        :type component_cooling: ComponentCooling
        :return: r_th of TIM material
        :rtype: float
        """
        r_th_tim = 1 / component_cooling.tim_conductivity * component_cooling.tim_thickness / copper_coin_bot_area

        return r_th_tim
