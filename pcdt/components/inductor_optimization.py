"""Inductor optimization class."""
# python libraries
import os
import pickle
import logging
import copy
import shutil
import threading
from multiprocessing import Pool, cpu_count, current_process
import traceback

# 3rd party libraries
import numpy as np
import tqdm

# own libraries
import femmt as fmt
import pcdt
from pcdt.circuit_enums import CalcModeEnum
from pcdt.boundary_check import CheckCondition as c_flag
from pcdt.components.inductor_optimization_dtos import InductorOptimizationDto
from pcdt.components.heat_sink_optimization import ThermalCalcSupport
from pcdt.server_ctl_dtos import ProgressData, ProgressStatus
from pcdt.server_ctl_dtos import RunTimeMeasurement as RunTime
from pcdt.datasets_dtos import InductorConfiguration
from pcdt.constant_path import CIRCUIT_INDUCTOR_RELUCTANCE_LOSSES_FOLDER, CIRCUIT_INDUCTOR_FEM_LOSSES_FOLDER
from pcdt.components.component_dtos import InductorRequirements, InductorResults, ComponentCooling

# configure root logger
logger = logging.getLogger(__name__)

class InductorOptimization:
    """Optimization of the inductor."""

    # Declaration of member types
    _optimization_config_list: list[list[InductorOptimizationDto]]
    _i_lock_stat: threading.Lock
    _progress_run_time: RunTime

    def __init__(self) -> None:
        """Initialize the configuration list for the inductor optimizations."""
        self._i_lock_stat: threading.Lock = threading.Lock()
        self._optimization_config_list = []
        self._number_performed_calculations: int = 0
        self._progress_run_time: RunTime = RunTime()

    @staticmethod
    def verify_optimization_parameter(toml_inductor: pcdt.TomlInductor) -> tuple[bool, str]:
        """Verify the input parameter ranges from the inductor toml configuration file.

        :param toml_inductor: toml inductor configuration
        :type toml_inductor: pcdt.TomlInductor
        :return: True, if the configuration was consistent
        :rtype: bool
        """
        # Variable declaration
        inconsistency_report: str = ""
        is_consistent: bool = True
        keyword_dictionary: dict
        toml_check_min_max_values_list: list[tuple[list[float], str]]
        toml_check_value_list: list[tuple[float, str]]

        # Design space parameter check
        group_name = "design_space"
        # Check core_name_list
        if len(toml_inductor.design_space.core_name_list) != 0:
            # Get available keywords
            keyword_dictionary = fmt.core_database()
            # Perform dictionary check
            for keyword_entry in toml_inductor.design_space.core_name_list:
                is_check_passed, issue_report = pcdt.BoundaryCheck.check_dictionary(keyword_dictionary, keyword_entry, "core_name_list")
                # Check if boundary check fails
                if not is_check_passed:
                    inconsistency_report = inconsistency_report + issue_report
                    is_consistent = False
        else:

            toml_check_min_max_values_list = (
                [(toml_inductor.design_space.core_inner_diameter_min_max_list, f"{group_name}: core_inner_diameter_min_max_list"),
                 (toml_inductor.design_space.window_h_min_max_list, f"{group_name}: window_h_min_max_list"),
                 (toml_inductor.design_space.window_w_min_max_list, f"{group_name}: window_w_min_max_list")])

            # Perform the boundary check
            is_check_passed, issue_report = pcdt.BoundaryCheck.check_float_min_max_values_list(
                0, 5, toml_check_min_max_values_list, c_flag.check_exclusive, c_flag.check_exclusive)
            if not is_check_passed:
                inconsistency_report = inconsistency_report + issue_report
                is_consistent = False

        # Check litz_wire_list
        # Get available keywords
        keyword_dictionary = fmt.litz_database()
        # Perform dictionary check
        for keyword_entry in toml_inductor.design_space.litz_wire_name_list:
            is_check_passed, issue_report = pcdt.BoundaryCheck.check_dictionary(keyword_dictionary, keyword_entry, "litz_wire_name_list")
            # Check if boundary check fails
            if not is_check_passed:
                inconsistency_report = inconsistency_report + issue_report
                is_consistent = False

        # Perform temperature value check
        # Perform the boundary check
        is_check_passed, issue_report = pcdt.BoundaryCheck.check_float_value(
            -40, 175, toml_inductor.boundary_conditions.temperature,
            "boundary_conditions: temperature", c_flag.check_inclusive, c_flag.check_inclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Insulation parameter check
        group_name = "insulations"
        toml_check_value_list = (
            [(toml_inductor.insulations.primary_to_primary, f"{group_name}: primary_to_primary"),
             (toml_inductor.insulations.core_bot, f"{group_name}: core_bot"),
             (toml_inductor.insulations.core_top, f"{group_name}: core_top"),
             (toml_inductor.insulations.core_right, f"{group_name}: core_right"),
             (toml_inductor.insulations.core_left, f"{group_name}: core_left")])

        # Perform the boundary check
        is_check_passed, issue_report = pcdt.BoundaryCheck.check_float_value_list(
            0, 0.1, toml_check_value_list, c_flag.check_exclusive, c_flag.check_exclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Thermal data parameter check
        group_name = "thermal_data"
        # Perform list length check for thermal_cooling
        if len(toml_inductor.thermal_data.thermal_cooling) != 2:
            issue_report = f"    Number of values in parameter '{group_name}: thermal_cooling' is not equal 2!\n"
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False
        else:
            # Perform the boundary check for tim-thickness
            is_check_passed, issue_report = pcdt.BoundaryCheck.check_float_value(
                0, 0.01, toml_inductor.thermal_data.thermal_cooling[0],
                f"'{group_name}: thermal_cooling[0]-tim-thickness",
                c_flag.check_exclusive, c_flag.check_inclusive)
            if not is_check_passed:
                inconsistency_report = inconsistency_report + issue_report
                is_consistent = False

            # Perform the boundary check for tim-conductivity
            is_check_passed, issue_report = pcdt.BoundaryCheck.check_float_value(
                0, 100, toml_inductor.thermal_data.thermal_cooling[1],
                f"'{group_name}: thermal_cooling[1]-tim-conductivity",
                c_flag.check_exclusive, c_flag.check_inclusive)
            if not is_check_passed:
                inconsistency_report = inconsistency_report + issue_report
                is_consistent = False

        # Perform setting check
        group_name = "setting"
        # Perform the boundary check
        is_check_passed, issue_report = pcdt.BoundaryCheck.check_float_value(
            0, 1, toml_inductor.settings.fft_filter_value_factor, f"{group_name}: fft_filter_value_factor", c_flag.check_inclusive, c_flag.check_inclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False
        is_check_passed, issue_report = pcdt.BoundaryCheck.check_float_value(
            0, 1, toml_inductor.settings.mesh_accuracy, f"{group_name}: mesh_accuracy", c_flag.check_exclusive, c_flag.check_exclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Perform filter_distance value check
        group_name = "filter_distance"
        # Perform the boundary check
        is_check_passed, issue_report = pcdt.BoundaryCheck.check_float_min_max_values(
            0, 100, toml_inductor.filter_distance.factor_dc_losses_min_max_list,
            f"{group_name}: factor_dc_losses_min_max_list", c_flag.check_exclusive, c_flag.check_inclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        return is_consistent, inconsistency_report

    def initialize_inductor_optimization_list(self, configuration_data_list: list[InductorConfiguration],
                                              inductor_requirements_list: list[InductorRequirements]) -> None:
        """
        Initialize the inductor optimization.

        :param configuration_data_list: List of inductor configuration data including study data
        :type  configuration_data_list: list[InductorConfiguration]
        :param inductor_requirements_list: list with inductor requirements
        :type inductor_requirements_list: list[InductorRequirements]
        """
        # Create the io_config_list for all trials
        for inductor_requirements in inductor_requirements_list:
            # Set index
            inductor_number_in_circuit = inductor_requirements.inductor_number_in_circuit

            # Check, if inductor optimization is not to skip
            if not configuration_data_list[inductor_number_in_circuit].study_data.calculation_mode == CalcModeEnum.skip_mode\
                    or not configuration_data_list[inductor_number_in_circuit].simulation_calculation_mode == CalcModeEnum.skip_mode:

                circuit_id = inductor_requirements.circuit_id
                configuration_data = configuration_data_list[inductor_number_in_circuit]
                trial_directory = os.path.join(configuration_data.study_data.optimization_directory,
                                               circuit_id, configuration_data.study_data.study_name)

                # Catch mypy issue
                if configuration_data.inductor_toml_data is None:
                    raise ValueError("Serious programming error in inductor optimization. toml-data are not initialized.",
                                     "Please write an issue!")

                inductor_toml_data = configuration_data.inductor_toml_data

                # Insulation parameter
                act_insulations = fmt.InductorInsulationDTO(primary_to_primary=inductor_toml_data.insulations.primary_to_primary,
                                                            core_bot=inductor_toml_data.insulations.core_bot,
                                                            core_top=inductor_toml_data.insulations.core_top,
                                                            core_right=inductor_toml_data.insulations.core_right,
                                                            core_left=inductor_toml_data.insulations.core_left)

                # Initialize the material data source
                act_material_data_sources = fmt.MaterialDataSources(
                    permeability_datasource=inductor_toml_data.material_data_sources.permeability_datasource,
                    permittivity_datasource=inductor_toml_data.material_data_sources.permittivity_datasource,
                )

                # Create fix part of io_config
                fmt_inductor_optimization_dto = fmt.InductorOptimizationDTO(
                    inductor_study_name=configuration_data.study_data.study_name,
                    core_name_list=inductor_toml_data.design_space.core_name_list,
                    material_name_list=inductor_toml_data.design_space.material_name_list,
                    core_inner_diameter_min_max_list=inductor_toml_data.design_space.core_inner_diameter_min_max_list,
                    window_h_min_max_list=inductor_toml_data.design_space.window_h_min_max_list,
                    window_w_min_max_list=inductor_toml_data.design_space.window_w_min_max_list,
                    litz_wire_name_list=inductor_toml_data.design_space.litz_wire_name_list,
                    insulations=act_insulations,
                    target_inductance=inductor_requirements.target_inductance,
                    temperature=inductor_toml_data.boundary_conditions.temperature,
                    time_current_vec=[inductor_requirements.time_vec, inductor_requirements.current_vec],
                    inductor_optimization_directory=os.path.join(
                        configuration_data.study_data.optimization_directory,
                        circuit_id,
                        configuration_data.study_data.study_name),
                    material_data_sources=act_material_data_sources,
                    fft_filter_value_factor=inductor_toml_data.settings.fft_filter_value_factor,
                    mesh_accuracy=inductor_toml_data.settings.mesh_accuracy)

                # Initialize the statistical data
                stat_data_init: ProgressData = ProgressData(run_time=0, number_of_filtered_points=0,
                                                            progress_status=ProgressStatus.Idle)

                # Get thermal data
                thermal_data: ComponentCooling = ComponentCooling(
                    tim_thickness=configuration_data.inductor_toml_data.thermal_data.thermal_cooling[0],
                    tim_conductivity=configuration_data.inductor_toml_data.thermal_data.thermal_cooling[1])

                inductor_optimization_dto = InductorOptimizationDto(
                    trial_directory=trial_directory,
                    circuit_id=circuit_id,
                    inductor_number_in_circuit=inductor_number_in_circuit,
                    progress_data=copy.deepcopy(stat_data_init),
                    fmt_inductor_optimization_dto=fmt_inductor_optimization_dto,
                    number_of_trails=configuration_data.study_data.number_of_trials,
                    thermal_data=thermal_data,
                    factor_dc_losses_min_max_list=inductor_toml_data.filter_distance.factor_dc_losses_min_max_list,
                    inductor_requirements=inductor_requirements
                )

                # Check list size
                while len(self._optimization_config_list) <= inductor_number_in_circuit:
                    self._optimization_config_list.append([])

                # Add inductor dto to the sub-list of assigned number in circuit
                self._optimization_config_list[inductor_number_in_circuit].append(inductor_optimization_dto)

    def get_progress_data(self, index: int, filtered_list_id: int) -> ProgressData:
        """Provide the progress data of the optimization.

        :param index: Index within the list of component configurations
        :type  index: int
        :param filtered_list_id: List index of the filtered operation point from circuit
        :type  filtered_list_id: int

        :return: Progress data: Processing start time, actual processing time, number of filtered inductor Pareto front points and status.
        :rtype: ProgressData
        """
        # Variable declaration and default initialization
        ret_progress_data: ProgressData = ProgressData(run_time=0, number_of_filtered_points=0,
                                                       progress_status=ProgressStatus.Idle)

        # Check for valid filtered_list_id
        if len(self._optimization_config_list[index]) > filtered_list_id:
            # Lock statistical performance data access
            with self._i_lock_stat:
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
        with self._i_lock_stat:
            act_number_performed_calculations = self._number_performed_calculations

        return act_number_performed_calculations

    # Simulation handler. Later the simulation handler starts a process per list entry.
    @staticmethod
    def _optimize_reluctance_model(circuit_id: str, act_io_config: fmt.InductorOptimizationDTO, circuit_study_name: str,
                                   inductor_requirements: InductorRequirements, thermal_data: ComponentCooling,
                                   target_number_trials: int, factor_dc_losses_min_max_list: list[float], debug: pcdt.Debug) -> int:
        """
        Perform the optimization.

        :param circuit_id: Filename of the filtered optimal electrical circuit
        :type  circuit_id: str
        :param act_io_config: inductor configuration for the optimization
        :type  act_io_config: fmt.InductorOptimizationDTO
        :param circuit_study_name: Name of the circuit study
        :type  circuit_study_name: str
        :param inductor_requirements: Requirements for the inductor
        :type  inductor_requirements: InductorRequirements
        :param thermal_data: Thermal data of the connection to heat sink
        :type  thermal_data: ComponentCooling
        :param target_number_trials: Number of trials for the optimization
        :type  target_number_trials: int
        :param factor_dc_losses_min_max_list: Filter factor to use filter the results min and max values
        :type  factor_dc_losses_min_max_list: list[float]
        :param debug: Debug mode flag
        :type debug: bool

        :return: Number of calculated Pareto fronts
        :rtype:  int
        """
        quantity_of_inductor_id_pareto = 0

        # Check number of trials
        if target_number_trials > 0:
            fmt.optimization.InductorOptimization.ReluctanceModel.start_proceed_study(act_io_config, target_number_trials=target_number_trials)
        else:
            logger.info(f"Target number of trials = {target_number_trials} which are less equal 0!. No simulation is performed")
            return 0

        # filter the reluctance model data Pareto front
        df_inductor = fmt.optimization.InductorOptimization.ReluctanceModel.study_to_df(act_io_config)
        df_inductor_pareto = fmt.optimization.InductorOptimization.ReluctanceModel.filter_loss_list_df(
            df_inductor, factor_min_dc_losses=factor_dc_losses_min_max_list[0], factor_max_dc_losses=factor_dc_losses_min_max_list[1])
        if debug.general.is_debug:
            # reduce dataset to the given number from the debug configuration
            df_inductor_pareto = df_inductor_pareto.iloc[:debug.inductor.number_reluctance_working_point_max]

        config_filepath = os.path.join(act_io_config.inductor_optimization_directory, f"{act_io_config.inductor_study_name}.pkl")

        inductor_id_list_pareto = df_inductor_pareto["number"].to_numpy()

        # Overtake the filtered operation points
        quantity_of_inductor_id_pareto = len(inductor_id_list_pareto)

        logger.info(f"Full-operating point simulation list with {len(inductor_id_list_pareto)} designs: {inductor_id_list_pareto}")

        # simulate all operating points
        for inductor_id in tqdm.tqdm(inductor_id_list_pareto):
            df_inductor_id = df_inductor_pareto[df_inductor_pareto["number"] == float(inductor_id)]

            logger.debug(f"single_geometry_number: \n"
                         f"    {df_inductor_id.head()}")

            # Fill dimensional matrix and remove last dimension (which is the exact time/current value)
            combined_loss_array = np.full_like(inductor_requirements.time_array[..., 0], np.nan)
            core_loss_array = np.full_like(inductor_requirements.time_array[..., 0], np.nan)
            winding_loss_array = np.full_like(inductor_requirements.time_array[..., 0], np.nan)

            new_circuit_dto_directory = os.path.join(act_io_config.inductor_optimization_directory, CIRCUIT_INDUCTOR_RELUCTANCE_LOSSES_FOLDER)
            if not os.path.exists(new_circuit_dto_directory):
                os.makedirs(new_circuit_dto_directory)

            if os.path.exists(os.path.join(new_circuit_dto_directory, f"{inductor_id}.pkl")):
                logger.info(f"All-operating point simulation of circuit {circuit_id} with inductor {inductor_id} already exists. Skip.")
            else:
                for vec_vvp in np.ndindex(inductor_requirements.time_array[..., 0].shape):
                    time, unique_indices = np.unique(inductor_requirements.time_array[vec_vvp], return_index=True)
                    current = inductor_requirements.current_array[vec_vvp][unique_indices]

                    current_waveform = np.array([time, current])
                    logger.debug(f"{current_waveform=}")
                    logger.debug("All operating point simulation of:")
                    logger.debug(f"   * Circuit study: {circuit_study_name}")
                    logger.debug(f"   * Circuit ID: {circuit_id}")
                    logger.debug(f"   * Inductor study: {act_io_config.inductor_study_name}")
                    logger.debug(f"   * Inductor ID: {inductor_id}")

                    inductor_volume, combined_losses, area_to_heat_sink, winding_loss, core_loss = fmt.InductorOptimization.ReluctanceModel.full_simulation(
                        df_inductor_id, current_waveform=current_waveform,
                        inductor_config_filepath=config_filepath)
                    combined_loss_array[vec_vvp] = combined_losses
                    core_loss_array[vec_vvp] = core_loss
                    winding_loss_array[vec_vvp] = winding_loss

                # Calculate thermal resistance
                r_th_ind_heat_sink = ThermalCalcSupport.calculate_r_th_tim(
                    area_to_heat_sink, thermal_data)

                reluctance_inductor_results = InductorResults(
                    loss_array=combined_loss_array,
                    winding_loss_array=winding_loss_array,
                    core_loss_array=core_loss_array,
                    volume=inductor_volume,
                    area_to_heat_sink=area_to_heat_sink,
                    r_th_ind_heat_sink=r_th_ind_heat_sink,
                    inductance=inductor_requirements.target_inductance,
                    circuit_id=circuit_id,
                    inductor_number_in_circuit=inductor_requirements.inductor_number_in_circuit,
                    inductor_id=inductor_id
                )

                pickle_file = os.path.join(new_circuit_dto_directory, f"{int(inductor_id)}.pkl")
                with open(pickle_file, 'wb') as output:
                    pickle.dump(reluctance_inductor_results, output, pickle.HIGHEST_PROTOCOL)

        # returns the number of filtered results
        return quantity_of_inductor_id_pareto

    def _generate_optimization_parameter(self, circuit_study_name: str, inductor_in_circuit: int, debug: pcdt.Debug) -> (
            list[tuple[str, fmt.InductorOptimizationDTO, str, InductorRequirements, ComponentCooling, int, list[float], pcdt.Debug]]):
        """
        Generate the list of parameter sets for analytic and simulation optimization.

        :param circuit_study_name: Name of the circuit study
        :type  circuit_study_name: str
        :param inductor_in_circuit: Number of inductor to optimize
        :type  inductor_in_circuit: int
        :param debug: True to use debug mode which stops earlier
        :type debug: bool
        :return: List of parameter sets for multi simulation processes
        :rtype:  list[tuple[str, fmt.InductorOptimizationDTO, FilterData, InductorRequirements,
                 ComponentCooling, int, list[float], pcdt.Debug]]
        """
        parameter_set_list = []
        for act_optimization_configuration in self._optimization_config_list[inductor_in_circuit]:
            parameter_set = (act_optimization_configuration.circuit_id, act_optimization_configuration.fmt_inductor_optimization_dto,
                             circuit_study_name, act_optimization_configuration.inductor_requirements,
                             act_optimization_configuration.thermal_data,
                             act_optimization_configuration.number_of_trails, act_optimization_configuration.factor_dc_losses_min_max_list,
                             debug)
            # Add set to list
            parameter_set_list.append(parameter_set)

        return parameter_set_list

    def optimization_handler_reluctance_model(self, circuit_study_name: str, inductor_in_circuit: int, debug: pcdt.Debug) -> None:
        """
        Control the multi simulation processes.

        :param circuit_study_name: Name of the circuit study
        :type  circuit_study_name: str
        :param inductor_in_circuit: Number of inductor to optimize
        :type  inductor_in_circuit: int
        :param debug: True to use debug mode which stops earlier
        :type debug: bool
        """
        # Check if class is initialized and inductor_in_circuit is valid
        if len(self._optimization_config_list) == 0:
            raise ValueError("Inductor reluctance handler: Inductor selection class is no initialized")
        elif len(self._optimization_config_list) <= inductor_in_circuit or inductor_in_circuit < 0:
            raise ValueError(f"Inductor reluctance handler: Invalid parameter value 'inductor_in_circuit'={inductor_in_circuit}.\n"
                             f"Value has to be between 0 and {len(self._optimization_config_list)-1}.")

        number_cpus = cpu_count()

        parameter_set_list = self._generate_optimization_parameter(circuit_study_name, inductor_in_circuit, debug)

        with Pool(processes=number_cpus) as pool:
            # Perform parallel calculation
            pool.starmap(func=InductorOptimization._optimize_reluctance_model, iterable=parameter_set_list)

    def fem_simulation_handler(self, circuit_study_name: str, inductor_in_circuit: int, debug: pcdt.Debug) -> None:
        """
        Control the multi simulation processes.

        :param circuit_study_name: Name of the circuit study
        :type  circuit_study_name: str
        :param inductor_in_circuit: Number of inductor to optimize
        :type  inductor_in_circuit: int
        :param debug: True to use debug mode which stops earlier
        :type debug: bool
        """
        # Check if class is initialized and inductor_in_circuit is valid
        if len(self._optimization_config_list) == 0:
            raise ValueError("Inductor simulation handler: Inductor optimization class is no initialized")
        elif len(self._optimization_config_list) <= inductor_in_circuit or inductor_in_circuit < 0:
            raise ValueError(f"Inductor simulation handler:: Invalid parameter value 'inductor_in_circuit'={inductor_in_circuit}.\n"
                             f"Value has to be between 0 and {len(self._optimization_config_list)-1}.")

        number_cpus = cpu_count()

        parameter_set_list = self._generate_optimization_parameter(circuit_study_name, inductor_in_circuit, debug)

        with Pool(processes=number_cpus) as pool:
            # Perform parallel calculation
            pool.starmap(func=InductorOptimization._fem_simulation, iterable=parameter_set_list)

    # Simulation handler. Later the simulation handler starts a process per list entry.
    @staticmethod
    def _fem_simulation(circuit_id: str, act_io_config: fmt.InductorOptimizationDTO, circuit_study_name: str,
                        inductor_requirements: InductorRequirements, thermal_data: ComponentCooling,
                        target_number_trials: int, factor_dc_losses_min_max_list: list[float], debug: pcdt.Debug) -> None:
        """
        Perform the optimization.

        :param circuit_id: Filename of the filtered optimal electrical circuit
        :type  circuit_id: str
        :param act_io_config: inductor configuration for the optimization
        :type  act_io_config: fmt.InductorOptimizationDTO
        :param circuit_study_name: Name of the circuit study
        :type  circuit_study_name: str
        :param inductor_requirements: Requirements for the inductor
        :type  inductor_requirements: InductorRequirements
        :param thermal_data: Thermal data of the connection to heat sink
        :type  thermal_data: ComponentCooling
        :param target_number_trials: Number of trials for the optimization (Not used)
        :type  target_number_trials: int (Not used)
        :param factor_dc_losses_min_max_list: Filter factor to use filter the results min and max values
        :type  factor_dc_losses_min_max_list: list[float]
        :param debug: Debug DTO
        :type debug: pcdt.Debug
        """
        process_number = current_process().name

        df_inductor = fmt.optimization.InductorOptimization.ReluctanceModel.study_to_df(act_io_config)
        df_inductor_pareto = fmt.optimization.InductorOptimization.ReluctanceModel.filter_loss_list_df(
            df_inductor, factor_min_dc_losses=factor_dc_losses_min_max_list[0], factor_max_dc_losses=factor_dc_losses_min_max_list[1])
        if debug.general.is_debug:
            # reduce dataset to the fist given number from the debug configuration file
            df_inductor_pareto = df_inductor_pareto.iloc[:debug.inductor.number_fem_working_point_max]

        config_filepath = os.path.join(act_io_config.inductor_optimization_directory, f"{act_io_config.inductor_study_name}.pkl")

        inductor_id_list_pareto = df_inductor_pareto["number"].to_numpy()

        logger.info(f"Full-operating point simulation list with {len(inductor_id_list_pareto)} designs: {inductor_id_list_pareto}")

        # simulate all operating points
        for inductor_id in tqdm.tqdm(inductor_id_list_pareto):
            new_circuit_dto_directory = os.path.join(act_io_config.inductor_optimization_directory,
                                                     CIRCUIT_INDUCTOR_FEM_LOSSES_FOLDER)
            try:
                # generate a new df with only a single entry (with the geometry data)
                df_geometry_re_simulation_number = df_inductor_pareto[df_inductor_pareto["number"] == float(inductor_id)]

                logger.debug(f"single_geometry_number: \n"
                             f"    {df_geometry_re_simulation_number.head()}")

                combined_loss_array = np.full_like(inductor_requirements.time_array[..., 0], np.nan)
                winding_loss_array = np.full_like(inductor_requirements.time_array[..., 0], np.nan)
                core_loss_array = np.full_like(inductor_requirements.time_array[..., 0], np.nan)

                if not os.path.exists(new_circuit_dto_directory):
                    os.makedirs(new_circuit_dto_directory)

                # check if simulation already exists or has already failed in the past
                if (os.path.exists(os.path.join(new_circuit_dto_directory, f"{inductor_id}.pkl")) \
                        or os.path.exists(os.path.join(new_circuit_dto_directory, f"{inductor_id}_failed.txt"))):
                    logger.info(f"FEM-simulation of circuit {circuit_id}, inductor {inductor_id} already exists or has already failed in the past. Skip.")
                else:
                    logger.info(f"Start FEM-simulation of circuit {circuit_id}, inductor {inductor_id}.")
                    for vec_vvp in np.ndindex(combined_loss_array.shape):
                        time, unique_indices = np.unique(inductor_requirements.time_array[vec_vvp], return_index=True)
                        current = inductor_requirements.current_array[vec_vvp][unique_indices]

                        current_waveform = np.array([time, current])
                        logger.debug(f"{current_waveform=}")
                        logger.debug("All operating point simulation of:")
                        logger.debug(f"   * Circuit study: {circuit_study_name}")
                        logger.debug(f"   * Circuit ID: {circuit_id}")
                        logger.debug(f"   * Inductor study: {act_io_config.inductor_study_name}")
                        logger.debug(f"   * Inductor ID: {inductor_id}")

                        fem_output, p_total, area_to_heat_sink = fmt.InductorOptimization.FemSimulation.full_simulation(
                            df_geometry_re_simulation_number, current_waveform=current_waveform,
                            inductor_config_filepath=config_filepath, process_number=process_number)
                        combined_loss_array[vec_vvp] = p_total
                        winding_loss_array[vec_vvp] = fem_output.p_loss_winding
                        core_loss_array[vec_vvp] = fem_output.p_core_magnet

                    # Calculate thermal resistance
                    r_th_ind_heat_sink = ThermalCalcSupport.calculate_r_th_tim(area_to_heat_sink, thermal_data)

                    fem_inductor_results = InductorResults(
                        loss_array=combined_loss_array,
                        winding_loss_array=winding_loss_array,
                        core_loss_array=core_loss_array,
                        volume=fem_output.volume,
                        area_to_heat_sink=area_to_heat_sink,
                        r_th_ind_heat_sink=r_th_ind_heat_sink,
                        inductance=fem_output.inductance,
                        circuit_id=circuit_id,
                        inductor_id=inductor_id,
                        inductor_number_in_circuit=inductor_requirements.inductor_number_in_circuit,
                    )

                    shutil.copy(fem_output.geometry_figure_path, os.path.join(new_circuit_dto_directory, f"{int(inductor_id)}.png"))

                    pickle_file = os.path.join(new_circuit_dto_directory, f"{int(inductor_id)}.pkl")
                    with open(pickle_file, 'wb') as output:
                        pickle.dump(fem_inductor_results, output, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                logger.warning(f"circuit {circuit_id} with inductor {inductor_id}: an operation point exceeds the boundary!")
                failed_file = os.path.join(new_circuit_dto_directory, f"{int(inductor_id)}_failed.txt")
                traceback_text = traceback.format_exc()
                with open(failed_file, "a") as f:
                    f.write(f"Error message: {e}\n\n\n\n\n\n")
                    f.write(f"Traceback: {traceback_text}")
