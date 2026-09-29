"""Stacked transformer optimization class."""

# python libraries
import os
import copy
import logging
import os.path
import pickle
import threading
from multiprocessing import Pool, cpu_count, current_process
import traceback
import shutil

# 3rd party libraries
import numpy as np
import tqdm

# own libraries
import pcdt
from pcdt.circuit_enums import CalcModeEnum
from pcdt.components.transformer_optimization_dtos import TransformerOptimizationDto
import femmt as fmt
from pcdt.datasets_dtos import TransformerConfiguration
from pcdt.boundary_check import CheckCondition as c_flag
from pcdt.server_ctl_dtos import ProgressStatus, ProgressData
from pcdt.server_ctl_dtos import RunTimeMeasurement as RunTime
from pcdt.constant_path import CIRCUIT_TRANSFORMER_RELUCTANCE_LOSSES_FOLDER, CIRCUIT_TRANSFORMER_FEM_LOSSES_FOLDER
from pcdt.components.component_dtos import TransformerRequirements, StackedTransformerResults, ComponentCooling
from pcdt.components.heat_sink_optimization import ThermalCalcSupport

logger = logging.getLogger(__name__)

class TransformerOptimization:
    """Optimization of the transformer."""

    # List with configurations to optimize and lock variable
    _optimization_config_list: list[list[TransformerOptimizationDto]]
    _t_lock_stat: threading.Lock
    _progress_run_time: RunTime

    def __init__(self) -> None:
        """Initialize the configuration list for the transformer optimizations."""
        self._optimization_config_list = []
        self._t_lock_stat: threading.Lock = threading.Lock()
        self._number_performed_calculations: int = 0
        self._progress_run_time: RunTime = RunTime()

    @staticmethod
    def verify_optimization_parameter(toml_transformer: pcdt.TomlTransformer) -> tuple[bool, str]:
        """Verify the input parameter ranges.

        :param toml_transformer: toml transformer configuration
        :type toml_transformer: pcdt.TomlTransformer
        :return: True, if the configuration was consistent
        :rtype: bool
        """
        # Variable declaration
        inconsistency_report: str = ""
        is_consistent: bool = True
        toml_check_min_max_values_list: list[tuple[list[float], str]]
        toml_check_value_list: list[tuple[float, str]]

        # Design space parameter check
        group_name = "design_space"
        # Check core_name_list
        if len(toml_transformer.design_space.core_name_list) != 0:
            # Get available keywords
            keyword_dictionary: dict = fmt.core_database()
            # Perform dictionary check
            for keyword_entry in toml_transformer.design_space.core_name_list:
                is_check_passed, issue_report = pcdt.BoundaryCheck.check_dictionary(
                    keyword_dictionary, keyword_entry, f"{group_name}: core_name_list")
                # Check if boundary check fails
                if not is_check_passed:
                    inconsistency_report = inconsistency_report + issue_report
                    is_consistent = False
        else:
            toml_check_min_max_values_list = (
                [(toml_transformer.design_space.core_inner_diameter_min_max_list, f"{group_name}: core_inner_diameter_min_max_list"),
                 (toml_transformer.design_space.window_h_bot_min_max_list, f"{group_name}: window_h_bot_min_max_list"),
                 (toml_transformer.design_space.window_w_min_max_list, f"{group_name}: window_w_min_max_list")])

            # Perform the boundary check
            is_check_passed, issue_report = pcdt.BoundaryCheck.check_float_min_max_values_list(
                0, 5, toml_check_min_max_values_list, c_flag.check_exclusive, c_flag.check_exclusive)
            if not is_check_passed:
                inconsistency_report = inconsistency_report + issue_report
                is_consistent = False

        # Convert min_max-list from integer to float values
        float_n_p_top_min_max_list = pcdt.BoundaryCheck.convert_int_list_to_float_list(toml_transformer.design_space.n_p_top_min_max_list)
        float_n_p_bot_min_max_list = pcdt.BoundaryCheck.convert_int_list_to_float_list(toml_transformer.design_space.n_p_bot_min_max_list)
        # Setup check value list
        toml_check_min_max_values_list = (
            [(float_n_p_top_min_max_list, f"{group_name}: n_p_top_min_max_list"),
             (float_n_p_bot_min_max_list, f"{group_name}: n_p_bot_min_max_list")])

        # Perform the boundary check
        is_check_passed, issue_report = pcdt.BoundaryCheck.check_float_min_max_values_list(
            0, 10000, toml_check_min_max_values_list, c_flag.check_exclusive, c_flag.check_inclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Check litz_wire_list
        # Get available keywords
        keyword_dictionary = fmt.litz_database()
        # Perform dictionary check for primary litz
        for keyword_entry in toml_transformer.design_space.primary_litz_wire_list:
            is_check_passed, issue_report = pcdt.BoundaryCheck.check_dictionary(keyword_dictionary, keyword_entry, f"{group_name}: litz_wire_name_list")
            # Check if boundary check fails
            if not is_check_passed:
                inconsistency_report = inconsistency_report + issue_report
                is_consistent = False
        # Perform dictionary check for primary litz
        for keyword_entry in toml_transformer.design_space.secondary_litz_wire_list:
            is_check_passed, issue_report = pcdt.BoundaryCheck.check_dictionary(keyword_dictionary, keyword_entry, f"{group_name}: litz_wire_name_list")
            # Check if boundary check fails
            if not is_check_passed:
                inconsistency_report = inconsistency_report + issue_report
                is_consistent = False

        # Insulation parameter check
        group_name = "insulation"
        # Create a list of parameter to check
        toml_check_value_list = (
            [(toml_transformer.insulation.iso_primary_to_primary, f"{group_name}: iso_primary_to_primary"),
             (toml_transformer.insulation.iso_secondary_to_secondary, f"{group_name}: iso_secondary_to_secondary"),
             (toml_transformer.insulation.iso_primary_to_secondary, f"{group_name}: iso_primary_to_secondary"),
             (toml_transformer.insulation.iso_window_top_core_top, f"{group_name}: iso_window_top_core_top"),
             (toml_transformer.insulation.iso_window_top_core_bot, f"{group_name}: iso_window_top_core_bot"),
             (toml_transformer.insulation.iso_window_top_core_left, f"{group_name}: iso_window_top_core_left"),
             (toml_transformer.insulation.iso_window_top_core_right, f"{group_name}: iso_window_top_core_right"),
             (toml_transformer.insulation.iso_window_bot_core_top, f"{group_name}: iso_window_bot_core_top"),
             (toml_transformer.insulation.iso_window_bot_core_bot, f"{group_name}: iso_window_bot_core_bot"),
             (toml_transformer.insulation.iso_window_bot_core_left, f"{group_name}: iso_window_bot_core_left"),
             (toml_transformer.insulation.iso_window_bot_core_right, f"{group_name}: iso_window_bot_core_right")])

        # Perform insulation value check
        # Perform the boundary check
        is_check_passed, issue_report = pcdt.BoundaryCheck.check_float_value_list(
            0, 0.1, toml_check_value_list, c_flag.check_exclusive, c_flag.check_exclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Thermal data parameter check
        group_name = "thermal_data"
        # Perform list length check for thermal_cooling
        if len(toml_transformer.thermal_data.thermal_cooling) != 2:
            issue_report = f"    Number of values in parameter '{group_name}: thermal_cooling' is not equal 2!\n"
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False
        else:
            # Perform the boundary check for tim-thickness
            is_check_passed, issue_report = pcdt.BoundaryCheck.check_float_value(
                0, 0.01, toml_transformer.thermal_data.thermal_cooling[0],
                f"'{group_name}: thermal_cooling[0]-tim-thickness",
                c_flag.check_exclusive, c_flag.check_inclusive)
            if not is_check_passed:
                inconsistency_report = inconsistency_report + issue_report
                is_consistent = False

            # Perform the boundary check for tim-conductivity
            is_check_passed, issue_report = pcdt.BoundaryCheck.check_float_value(
                0, 100, toml_transformer.thermal_data.thermal_cooling[1],
                f"'{group_name}: thermal_cooling[1]-tim-conductivity",
                c_flag.check_exclusive, c_flag.check_inclusive)
            if not is_check_passed:
                inconsistency_report = inconsistency_report + issue_report
                is_consistent = False

        # Perform boundary condition check
        group_name = "boundary_condition"
        # Perform the boundary check
        is_check_passed, issue_report = pcdt.BoundaryCheck.check_float_value(
            -40, 175, toml_transformer.boundary_conditions.temperature, f"{group_name}: temperature", c_flag.check_inclusive, c_flag.check_inclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        toml_check_value_list = (
            [(toml_transformer.boundary_conditions.max_transformer_total_height, f"{group_name}: max_transformer_total_height"),
             (toml_transformer.boundary_conditions.max_core_volume, f"{group_name}: max_core_volume")])
        # Perform the boundary check
        is_check_passed, issue_report = pcdt.BoundaryCheck.check_float_value_list(
            0, 5, toml_check_value_list, c_flag.check_exclusive, c_flag.check_exclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Perform setting check
        group_name = "setting"
        # Perform the boundary check
        is_check_passed, issue_report = pcdt.BoundaryCheck.check_float_value(
            0, 1, toml_transformer.settings.fft_filter_value_factor, f"{group_name}: fft_filter_value_factor", c_flag.check_inclusive, c_flag.check_inclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False
        is_check_passed, issue_report = pcdt.BoundaryCheck.check_float_value(
            0, 1, toml_transformer.settings.mesh_accuracy, f"{group_name}: mesh_accuracy", c_flag.check_exclusive, c_flag.check_exclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Perform filter_distance value check
        group_name = "filter_distance"
        # Perform the boundary check
        is_check_passed, issue_report = pcdt.BoundaryCheck.check_float_min_max_values(
            0, 100, toml_transformer.filter_distance.factor_dc_losses_min_max_list,
            f"{group_name}: factor_dc_losses_min_max_list", c_flag.check_inclusive, c_flag.check_inclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        return is_consistent, inconsistency_report

    def initialize_transformer_optimization_list(self, configuration_data_list: list[TransformerConfiguration],
                                                 transformer_requirements_list: list[TransformerRequirements]) -> None:
        """Initialize the transformer optimization.

        :param configuration_data_list: List of transformer configuration data including study data
        :type  configuration_data_list: list[TransformerConfiguration]
        :param transformer_requirements_list: list with transformer requirements
        :type transformer_requirements_list: list[TransformerRequirements]
        """
        # Create the io_config_list for all trials
        for transformer_requirements in transformer_requirements_list:
            # Set index
            transformer_number_in_circuit = transformer_requirements.transformer_number_in_circuit

            # Check, if transformer optimization is not to skip
            if not configuration_data_list[transformer_number_in_circuit].study_data.calculation_mode == CalcModeEnum.skip_mode\
                    or not configuration_data_list[transformer_number_in_circuit].simulation_calculation_mode == CalcModeEnum.skip_mode:

                circuit_id = transformer_requirements.circuit_id
                configuration_data = configuration_data_list[transformer_number_in_circuit]
                trial_directory = os.path.join(configuration_data.study_data.optimization_directory,
                                               circuit_id, configuration_data.study_data.study_name)

                # Check if transformer_toml_data not initializes
                if configuration_data.transformer_toml_data is None:
                    raise ValueError("Serious programming error in transformer optimization. toml-data are not initialized.",
                                     "Please write an issue!")

                transformer_toml_data = configuration_data.transformer_toml_data

                # common parameters for all types of transformers
                act_insulations = fmt.StoInsulation(
                    # insulation for top core window
                    iso_window_top_core_top=transformer_toml_data.insulation.iso_window_top_core_top,
                    iso_window_top_core_bot=transformer_toml_data.insulation.iso_window_top_core_bot,
                    iso_window_top_core_left=transformer_toml_data.insulation.iso_window_top_core_left,
                    iso_window_top_core_right=transformer_toml_data.insulation.iso_window_top_core_right,
                    # insulation for bottom core window
                    iso_window_bot_core_top=transformer_toml_data.insulation.iso_window_bot_core_top,
                    iso_window_bot_core_bot=transformer_toml_data.insulation.iso_window_bot_core_bot,
                    iso_window_bot_core_left=transformer_toml_data.insulation.iso_window_bot_core_left,
                    iso_window_bot_core_right=transformer_toml_data.insulation.iso_window_bot_core_right,
                    # winding-to-winding insulation
                    iso_primary_to_primary=transformer_toml_data.insulation.iso_primary_to_primary,
                    iso_secondary_to_secondary=transformer_toml_data.insulation.iso_secondary_to_secondary,
                    iso_primary_to_secondary=transformer_toml_data.insulation.iso_primary_to_secondary
                )

                # Initialize the material data source
                material_data_sources = fmt.MaterialDataSources(
                    permeability_datasource=transformer_toml_data.material_data_sources.permeability_datasource,
                    permittivity_datasource=transformer_toml_data.material_data_sources.permittivity_datasource,
                )

                # Create fix part of io_config
                sto_config = fmt.StoSingleInputConfig(
                    stacked_transformer_study_name=configuration_data.study_data.study_name,
                    # target parameters  initialized with default values
                    l_s12_target=0,
                    l_h_target=0,
                    n_target=0,
                    # operating point: current waveforms and temperature initialized with default values
                    time_current_1_vec=np.ndarray([]),
                    time_current_2_vec=np.ndarray([]),
                    temperature=transformer_toml_data.boundary_conditions.temperature,
                    # sweep parameters: geometry and materials
                    n_p_top_min_max_list=transformer_toml_data.design_space.n_p_top_min_max_list,
                    n_p_bot_min_max_list=transformer_toml_data.design_space.n_p_bot_min_max_list,
                    material_list=transformer_toml_data.design_space.material_name_list,
                    core_name_list=transformer_toml_data.design_space.core_name_list,
                    core_inner_diameter_min_max_list=transformer_toml_data.design_space.core_inner_diameter_min_max_list,
                    window_w_min_max_list=transformer_toml_data.design_space.window_w_min_max_list,
                    window_h_bot_min_max_list=transformer_toml_data.design_space.window_h_bot_min_max_list,
                    primary_litz_wire_list=transformer_toml_data.design_space.primary_litz_wire_list,
                    secondary_litz_wire_list=transformer_toml_data.design_space.secondary_litz_wire_list,
                    # maximum limitation for transformer total height and core volume
                    max_transformer_total_height=transformer_toml_data.boundary_conditions.max_transformer_total_height,
                    max_core_volume=transformer_toml_data.boundary_conditions.max_core_volume,
                    # fix parameters: insulation
                    insulations=act_insulations,
                    # misc
                    stacked_transformer_optimization_directory=configuration_data.study_data.optimization_directory,

                    fft_filter_value_factor=transformer_toml_data.settings.fft_filter_value_factor,
                    mesh_accuracy=transformer_toml_data.settings.mesh_accuracy,

                    # data sources
                    material_data_sources=material_data_sources
                )

                # Initialize the statistical data
                stat_data_init: ProgressData = ProgressData(run_time=0, number_of_filtered_points=0, progress_status=ProgressStatus.Idle)

                # Generate new sto_config
                next_io_config = copy.deepcopy(sto_config)
                # target parameters
                next_io_config.l_s12_target = float(transformer_requirements.l_s12_target)
                next_io_config.l_h_target = float(transformer_requirements.l_h_target)
                next_io_config.n_target = float(transformer_requirements.n_target)
                # operating point: current waveforms and temperature initialized with default values
                next_io_config.time_current_1_vec = np.array([transformer_requirements.time_vec, transformer_requirements.current_1_vec])
                next_io_config.time_current_2_vec = np.array([transformer_requirements.time_vec, transformer_requirements.current_2_vec])
                # misc
                # Get thermal data
                thermal_data: ComponentCooling = ComponentCooling(
                    tim_thickness=configuration_data.transformer_toml_data.thermal_data.thermal_cooling[0],
                    tim_conductivity=configuration_data.transformer_toml_data.thermal_data.thermal_cooling[1])

                next_io_config.stacked_transformer_optimization_directory = trial_directory
                transformer_dto = TransformerOptimizationDto(
                    trial_directory=trial_directory,
                    circuit_id=circuit_id,
                    transformer_number_in_circuit=transformer_number_in_circuit,
                    progress_data=copy.deepcopy(stat_data_init),
                    number_of_trails=configuration_data.study_data.number_of_trials,
                    fmt_transformer_optimization_dto=next_io_config,
                    thermal_data=thermal_data,
                    factor_dc_losses_min_max_list=transformer_toml_data.filter_distance.factor_dc_losses_min_max_list,
                    transformer_requirements=transformer_requirements)

                # Check list size
                while len(self._optimization_config_list) <= transformer_number_in_circuit:
                    self._optimization_config_list.append([])

                # Add transformer dto to the sub-list of assigned number in circuit
                self._optimization_config_list[transformer_number_in_circuit].append(transformer_dto)

    def get_progress_data(self, index: int, filtered_list_id: int) -> ProgressData:
        """Provide the progress data of the optimization.

        :param index: Index within the list of component configurations
        :type  index: int
        :param filtered_list_id: List index of the filtered operation point from circuit
        :type  filtered_list_id: int
        :return: Progress data: Processing start time, actual processing time, number of filtered transformer Pareto front points and status.
        :rtype: ProgressData
        """
        # Variable declaration and default initialization
        ret_progress_data: ProgressData = ProgressData(
            run_time=0, number_of_filtered_points=0,
            progress_status=ProgressStatus.Idle)

        # Check for valid filtered_list_id
        if len(self._optimization_config_list[index]) > filtered_list_id:
            # Lock statistical performance data access (ASA: Possible Bug)
            with self._t_lock_stat:
                # Check if list is in progress
                if self._optimization_config_list[index][filtered_list_id].progress_data.progress_status == ProgressStatus.InProgress:
                    # Update statistical data
                    self._optimization_config_list[index][filtered_list_id].progress_data.run_time = self._progress_run_time.get_runtime()

                # Create a copy of actual data
                ret_progress_data = copy.deepcopy(self._optimization_config_list[index][filtered_list_id].progress_data)

        return ret_progress_data

    def get_number_of_performed_calculations(self) -> int:
        """Provide the number of performed calculations.

        :return: int: Number of performed calculations
        :rtype: int
        """
        return self._number_performed_calculations

    @staticmethod
    def _optimize_reluctance_model(circuit_id: str, act_sto_config: fmt.StoSingleInputConfig, circuit_study_name: str,
                                   transformer_requirements: TransformerRequirements, thermal_data: ComponentCooling,
                                   target_number_trials: int, factor_dc_losses_min_max_list: list[float], debug: pcdt.Debug) -> int:
        """
        Perform the optimization.

        :param circuit_id: Filename of the filtered optimal electrical circuit
        :type circuit_id: str
        :param act_sto_config: stacked transformer optimization configuration
        :type act_sto_config: fmt.StoSingleInputConfig
        :param circuit_study_name: Name of the circuit study
        :type  circuit_study_name: str
        :param transformer_requirements: Requirements for the transformer
        :type  transformer_requirements: TransformerRequirements
        :param thermal_data: Thermal data of the connection to heat sink
        :type  thermal_data: ComponentCooling
        :param target_number_trials: Number of trials for the reluctance model optimization
        :type target_number_trials: int
        :param factor_dc_losses_min_max_list: Pareto filter, tolerance band = Multiplication of minimum/maximum losses
        :type factor_dc_losses_min_max_list: float
        :param debug: Debug DTO
        :type debug: pcdt.Debug
        """
        # Number of filtered operating points
        quantity_transformer_id_pareto = 0

        # Check number of trials
        if target_number_trials > 0:
            fmt.optimization.StackedTransformerOptimization.ReluctanceModel.start_proceed_study(
                act_sto_config, target_number_trials=target_number_trials)
        else:
            logger.info(f"Target number of trials = {target_number_trials} which are less equal 0!. No simulation is performed")
            return 0

        # Filter reluctance model results
        df_transformer = fmt.optimization.StackedTransformerOptimization.ReluctanceModel.study_to_df(act_sto_config)
        df_transformer_pareto = fmt.optimization.StackedTransformerOptimization.ReluctanceModel.filter_loss_list_df(
            df_transformer, factor_min_dc_losses=factor_dc_losses_min_max_list[0], factor_max_dc_losses=factor_dc_losses_min_max_list[1])
        if debug.general.is_debug:
            # reduce dataset to the given number from the debug configuration
            df_transformer_pareto = df_transformer_pareto.iloc[:debug.transformer.number_reluctance_working_point_max]

        # Assemble configuration path
        config_filepath = os.path.join(act_sto_config.stacked_transformer_optimization_directory,
                                       f"{act_sto_config.stacked_transformer_study_name}.pkl")

        transformer_id_list_pareto = df_transformer_pareto["number"].to_numpy()

        # Overtake the filtered operation points
        quantity_transformer_id_pareto = len(transformer_id_list_pareto)

        logger.info(f"Full-operating point simulation list with {len(transformer_id_list_pareto)} designs: {transformer_id_list_pareto}")

        for transformer_id in tqdm.tqdm(transformer_id_list_pareto):
            df_transformer_id = df_transformer_pareto[df_transformer_pareto["number"] == transformer_id]

            result_array = np.full_like(transformer_requirements.time_array[..., 0], np.nan)
            winding_1_loss_array = np.full_like(transformer_requirements.time_array[..., 0], np.nan)
            winding_2_loss_array = np.full_like(transformer_requirements.time_array[..., 0], np.nan)
            core_loss_array = np.full_like(transformer_requirements.time_array[..., 0], np.nan)

            new_circuit_dto_directory = os.path.join(act_sto_config.stacked_transformer_optimization_directory,
                                                     CIRCUIT_TRANSFORMER_RELUCTANCE_LOSSES_FOLDER)
            if not os.path.exists(new_circuit_dto_directory):
                os.makedirs(new_circuit_dto_directory)

            if os.path.exists(os.path.join(new_circuit_dto_directory, f"{transformer_id}.pkl")):
                logger.info(f"All-operating point simulation of circuit {circuit_id} with transformer {transformer_id} already exists. Skip.")
            else:
                for vec_vvp in np.ndindex(result_array.shape):
                    time, unique_indices = np.unique(transformer_requirements.time_array[vec_vvp], return_index=True)
                    current_1 = transformer_requirements.current_1_array[vec_vvp][unique_indices]
                    current_2 = transformer_requirements.current_2_array[vec_vvp][unique_indices]

                    current_waveform_1 = np.array([time, current_1])
                    current_waveform_2 = np.array([time, current_2])

                    logger.debug(f"{current_waveform_1=}")
                    logger.debug("----------------------")
                    logger.debug("Re-simulation of:")
                    logger.debug(f"   * Circuit study: {circuit_study_name}")
                    logger.debug(f"   * Circuit ID: {circuit_id}")
                    logger.debug(f"   * Transformer study: {act_sto_config.stacked_transformer_study_name}")
                    logger.debug(f"   * Transformer ID: {transformer_id}")

                    volume, combined_losses, area_to_heat_sink, winding_1_loss, winding_2_loss, core_loss = (
                        fmt.StackedTransformerOptimization.ReluctanceModel.full_simulation(
                            df_transformer_id, current_waveform_1, current_waveform_2, config_filepath))
                    result_array[vec_vvp] = combined_losses
                    winding_1_loss_array[vec_vvp] = winding_1_loss
                    winding_2_loss_array[vec_vvp] = winding_2_loss
                    core_loss_array[vec_vvp] = core_loss

                # Calculate thermal resistance
                r_th_xfmr_heat_sink = ThermalCalcSupport.calculate_r_th_tim(
                    area_to_heat_sink, thermal_data)

                transformer_results = StackedTransformerResults(
                    loss_array=result_array,
                    winding_1_loss_array=winding_1_loss_array,
                    winding_2_loss_array=winding_2_loss_array,
                    core_loss_array=core_loss_array,
                    volume=volume,
                    area_to_heat_sink=area_to_heat_sink,
                    r_th_xfmr_heat_sink=r_th_xfmr_heat_sink,
                    circuit_id=circuit_id,
                    transformer_id=transformer_id,
                    transformer_number_in_circuit=transformer_requirements.transformer_number_in_circuit,
                    l_s=transformer_requirements.l_s12_target,
                    l_h=transformer_requirements.l_h_target,
                    n=transformer_requirements.n_target,
                )

                pickle_file = os.path.join(new_circuit_dto_directory, f"{int(transformer_id)}.pkl")
                with open(pickle_file, 'wb') as output:
                    pickle.dump(transformer_results, output, pickle.HIGHEST_PROTOCOL)

        # returns the number of filtered results
        return quantity_transformer_id_pareto

    def _generate_optimization_parameter(self, circuit_study_name: str, transformer_in_circuit: int, debug: pcdt.Debug) -> (
            list[tuple[str, fmt.fmt.StoSingleInputConfig, str, TransformerRequirements, ComponentCooling, int, list[float], pcdt.Debug]]):
        """
        Generate the list of parameter sets for analytic and simulation optimization.

        :param circuit_study_name: Name of the circuit study
        :type  circuit_study_name: str
        :param transformer_in_circuit: Number of transformer to optimize
        :type  transformer_in_circuit: int
        :param debug: True to use debug mode which stops earlier
        :type debug: bool
        :return: List of parameter sets for multi simulation processes
        :rtype:  list[tuple[str, fmt.TransformerOptimizationDTO, FilterData, TransformerRequirements,
                 ComponentCooling, int, list[float], pcdt.Debug]]
        """
        parameter_set_list = []
        for act_optimization_configuration in self._optimization_config_list[transformer_in_circuit]:
            parameter_set = (
                act_optimization_configuration.circuit_id, act_optimization_configuration.fmt_transformer_optimization_dto,
                circuit_study_name, act_optimization_configuration.transformer_requirements,
                act_optimization_configuration.thermal_data,
                act_optimization_configuration.number_of_trails,
                act_optimization_configuration.factor_dc_losses_min_max_list,
                debug)
            # Add set to list
            parameter_set_list.append(parameter_set)

        return parameter_set_list

    # Simulation handler. Later the simulation handler starts a process per list entry.
    def optimization_handler_reluctance_model(self, circuit_study_name: str, transformer_in_circuit: int, debug: pcdt.Debug) -> None:
        """
        Control the multi simulation processes.

        :param circuit_study_name: Name of the circuit study
        :type  circuit_study_name: str
        :param transformer_in_circuit: Number of transformer component in circuit
        :type  transformer_in_circuit: int
        :param debug: True to use debug mode which stops earlier
        :type debug: bool
        """
        # Check if class is initialized and transformer_in_circuit is valid
        if len(self._optimization_config_list) == 0:
            raise ValueError("Transformer reluctance handler:Transformer selection class is no initialized")
        elif len(self._optimization_config_list) <= transformer_in_circuit or transformer_in_circuit < 0:
            raise ValueError(f"Transformer reluctance handler: Invalid parameter value 'transformer_in_circuit'={transformer_in_circuit}.\n"
                             f"Value has to be between 0 and {len(self._optimization_config_list)-1}.")

        number_cpus = cpu_count()

        parameter_set_list = self._generate_optimization_parameter(circuit_study_name, transformer_in_circuit, debug)

        with Pool(processes=number_cpus) as pool:
            # Perform parallel calculation
            pool.starmap(func=TransformerOptimization._optimize_reluctance_model, iterable=parameter_set_list)

    def fem_simulation_handler(self, circuit_study_name: str, transformer_in_circuit: int, debug: pcdt.Debug) -> None:
        """
        Control the multi simulation processes.

        :param circuit_study_name: Name of the circuit study
        :type  circuit_study_name: str
        :param transformer_in_circuit: Number of transformer component in circuit
        :type  transformer_in_circuit: int
        :param debug: Debug DTO
        :type debug: pcdt.Debug
        """
        # Check if class is initialized and transformer_in_circuit is valid
        if len(self._optimization_config_list) == 0:
            raise ValueError("Transformer simulation handler:Transformer selection class is no initialized")
        elif len(self._optimization_config_list) <= transformer_in_circuit or transformer_in_circuit < 0:
            raise ValueError(f"Transformer simulation handler: Invalid parameter value 'transformer_in_circuit'={transformer_in_circuit}.\n"
                             f"Value has to be between 0 and {len(self._optimization_config_list)-1}.")

        number_cpus = cpu_count()

        parameter_set_list = self._generate_optimization_parameter(circuit_study_name, transformer_in_circuit, debug)

        with Pool(processes=number_cpus) as pool:
            # Perform parallel calculation
            pool.starmap(func=TransformerOptimization._fem_simulation, iterable=parameter_set_list)

    @staticmethod
    def _fem_simulation(circuit_id: str, act_sto_config: fmt.StoSingleInputConfig, circuit_study_name: str,
                        transformer_requirements: TransformerRequirements, thermal_data: ComponentCooling,
                        target_number_trials: int, factor_dc_losses_min_max_list: list[float], debug: pcdt.Debug) -> None:
        """
        Perform the optimization.

        :param circuit_id: Filename of the filtered optimal electrical circuit
        :type  circuit_id: str
        :param act_sto_config: stacked transformer configuration for the optimization
        :type  act_sto_config: fmt.StackedTransformerOptimizationDTO
        :param circuit_study_name: Name of the circuit study
        :type  circuit_study_name: str
        :param transformer_requirements: Requirements for the transformer
        :type  transformer_requirements: TransformerRequirements
        :param thermal_data: Thermal data of the connection to heat sink
        :type  thermal_data: ComponentCooling
        :param target_number_trials: Number of trials for the reluctance model optimization (not used)
        :type target_number_trials: int (not used)
        :param factor_dc_losses_min_max_list: Filter factor to use filter the results min and max values
        :type  factor_dc_losses_min_max_list: list[float]
        :param debug: Debug DTO
        :type debug: pcdt.Debug
        """
        # Number of filtered operating points
        number_of_filtered_points = 0

        process_number = current_process().name

        # Filter study. Use same filter as in the reluctance model
        df = fmt.optimization.StackedTransformerOptimization.ReluctanceModel.study_to_df(act_sto_config)
        df_transformer_pareto = fmt.optimization.StackedTransformerOptimization.ReluctanceModel.filter_loss_list_df(
            df, factor_min_dc_losses=factor_dc_losses_min_max_list[0], factor_max_dc_losses=factor_dc_losses_min_max_list[1])
        if debug.general.is_debug:
            # reduce dataset to the given number from the debug configuration
            df_transformer_pareto = df_transformer_pareto.iloc[:debug.transformer.number_fem_working_point_max]

        # Assemble configuration path
        config_filepath = os.path.join(act_sto_config.stacked_transformer_optimization_directory,
                                       f"{act_sto_config.stacked_transformer_study_name}.pkl")

        transformer_id_list_pareto = df_transformer_pareto["number"].to_numpy()

        logger.info(f"Full-operating point simulation list with {len(transformer_id_list_pareto)} designs: {transformer_id_list_pareto}")

        for transformer_id in tqdm.tqdm(transformer_id_list_pareto):
            df_geometry_re_simulation_number = df_transformer_pareto[df_transformer_pareto["number"] == transformer_id]

            new_circuit_dto_directory = os.path.join(act_sto_config.stacked_transformer_optimization_directory,
                                                     CIRCUIT_TRANSFORMER_FEM_LOSSES_FOLDER)

            result_array = np.full_like(transformer_requirements.time_array[..., 0], np.nan)
            winding_1_loss_array = np.full_like(transformer_requirements.time_array[..., 0], np.nan)
            winding_2_loss_array = np.full_like(transformer_requirements.time_array[..., 0], np.nan)
            core_loss_array = np.full_like(transformer_requirements.time_array[..., 0], np.nan)

            if not os.path.exists(new_circuit_dto_directory):
                os.makedirs(new_circuit_dto_directory)

            # check if simulation already exists or has already failed in the past
            if (os.path.exists(os.path.join(new_circuit_dto_directory, f"{transformer_id}.pkl")) \
                    or os.path.exists(os.path.join(new_circuit_dto_directory, f"{transformer_id}_failed.txt"))):
                logger.info(f"FEM-simulation of circuit {circuit_id}, inductor {transformer_id} already exists or has already failed in the past. Skip.")
            else:
                # The femmt simulation (full_simulation()) can raise different errors, most of them are geometry errors
                # e.g. winding is not fitting in the winding window
                try:
                    for vec_vvp in np.ndindex(transformer_requirements.time_array[..., 0].shape):
                        time, unique_indices = np.unique(transformer_requirements.time_array[vec_vvp], return_index=True)
                        current_1 = transformer_requirements.current_1_array[vec_vvp][unique_indices]
                        current_2 = transformer_requirements.current_2_array[vec_vvp][unique_indices]

                        current_1_waveform = np.array([time, current_1])
                        current_2_waveform = np.array([time, current_2])

                        logger.debug(f"{current_1_waveform=}")
                        logger.debug("----------------------")
                        logger.debug("Re-simulation of:")
                        logger.debug(f"   * Circuit study: {circuit_study_name}")
                        logger.debug(f"   * Circuit ID: {circuit_id}")
                        logger.debug(f"   * Transformer study: {act_sto_config.stacked_transformer_study_name}")
                        logger.debug(f"   * Transformer ID: {transformer_id}")

                        fem_output, p_total, p_core, area_to_heat_sink = fmt.StackedTransformerOptimization.FemSimulation.full_simulation(
                            df_geometry_re_simulation_number, current_1_waveform, current_2_waveform, config_filepath, show_visual_outputs=False,
                            process_number=process_number)

                        result_array[vec_vvp] = p_total
                        winding_1_loss_array[vec_vvp] = fem_output.p_loss_winding_1
                        winding_2_loss_array[vec_vvp] = fem_output.p_loss_winding_2
                        core_loss_array[vec_vvp] = p_core

                    # Calculate thermal resistance
                    r_th_xfmr_heat_sink = ThermalCalcSupport.calculate_r_th_tim(
                        area_to_heat_sink, thermal_data)

                    transformer_results = StackedTransformerResults(
                        loss_array=result_array,
                        winding_1_loss_array=winding_1_loss_array,
                        winding_2_loss_array=winding_2_loss_array,
                        core_loss_array=core_loss_array,
                        volume=fem_output.volume,
                        area_to_heat_sink=area_to_heat_sink,
                        r_th_xfmr_heat_sink=r_th_xfmr_heat_sink,
                        circuit_id=circuit_id,
                        transformer_id=transformer_id,
                        transformer_number_in_circuit=transformer_requirements.transformer_number_in_circuit,
                        l_s=transformer_requirements.l_s12_target,
                        l_h=transformer_requirements.l_h_target,
                        n=transformer_requirements.n_target,
                    )

                    shutil.copy(fem_output.geometry_figure_path, os.path.join(new_circuit_dto_directory, f"{int(transformer_id)}.png"))

                    pickle_file = os.path.join(new_circuit_dto_directory, f"{int(transformer_id)}.pkl")
                    with open(pickle_file, 'wb') as output:
                        pickle.dump(transformer_results, output, pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    logger.info(f"FEM-simulation of transformer geometry {transformer_id} not possible due to non-possible geometry.")
                    failed_file = os.path.join(new_circuit_dto_directory, f"{int(transformer_id)}_failed.txt")
                    traceback_text = traceback.format_exc()
                    with open(failed_file, "a") as f:
                        f.write(f"Error message: {e}\n\n\n\n\n\n")
                        f.write(f"Traceback: {traceback_text}")
