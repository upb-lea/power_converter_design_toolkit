"""Capacitor selection."""

# python libraries
import logging
import os
from multiprocessing import Pool, cpu_count
import copy
import pickle

# 3rd party libraries
import numpy as np
import pandas as pd
import tqdm

# own libraries
import pecst
from pcdt.components.capacitor_optimization_dtos import CapacitorOptimizationDto
from pcdt.toml_checker import TomlCapacitorSelection, Debug
from pcdt.circuit_enums import CalcModeEnum
from pcdt.datasets_dtos import FilterData, CapacitorConfiguration
from pcdt.server_ctl_dtos import ProgressData, ProgressStatus
from pcdt.components.component_dtos import CapacitorRequirements
from pcdt.components.capacitor_optimization_dtos import CapacitorResults
from pcdt.constant_path import CIRCUIT_CAPACITOR_LOSS_FOLDER, CAPACITOR_RESULTS, CAPACITOR_RESULTS_FILTERED

logger = logging.getLogger(__name__)

class CapacitorSelection:
    """Select suitable capacitors."""

    c_df: pd.DataFrame
    _optimization_config_list: list[list[CapacitorOptimizationDto]]

    def __init__(self):
        self._optimization_config_list = []

    @staticmethod
    def verify_optimization_parameter(toml_capacitor: TomlCapacitorSelection) -> tuple[bool, str]:
        """
        Verify the parameters from toml file for the capacitor optimization.

        Dummy method so far.

        :param toml_capacitor: capacitor toml file to check
        :type toml_capacitor: TomlCapacitorSelection
        :return: is_consistent, issue_report
        :rtype: tuple[bool, str]
        """
        if toml_capacitor:
            pass
        return True, ""

    def initialize_capacitor_selection(self, configuration_data_list: list[CapacitorConfiguration],
                                       capacitor_requirements_list: list[CapacitorRequirements]) -> None:
        """
        Initialize the capacitor selection.

        The initialization initialize the optimization config list, which contains lists separated by
        number of capacitor component in circuit. After performing this method, the optimization handler
        can be used to optimize the selected capacitor. The selection capacitor is defined by the number
        of capacitor component.

        :param configuration_data_list: List of capacitor configuration data including study data
        :type  configuration_data_list: list[CapacitorConfiguration]
        :param capacitor_requirements_list: list with capacitor requirements
        :type capacitor_requirements_list: list[CapacitorRequirements]
        """
        pecst.download_esr_csv_files()

        # Create the io_config_list for all trials
        for capacitor_requirements in capacitor_requirements_list:
            # Set index
            capacitor_number_in_circuit = capacitor_requirements.capacitor_number_in_circuit

            # Check, if capacitor optimization is not to skip
            if not configuration_data_list[capacitor_number_in_circuit].study_data.calculation_mode == CalcModeEnum.skip_mode:

                circuit_id = capacitor_requirements.circuit_id
                trial_directory = os.path.join(configuration_data_list[capacitor_number_in_circuit].study_data.optimization_directory,
                                               circuit_id, configuration_data_list[capacitor_number_in_circuit].study_data.study_name)
                capacitor_toml_data = configuration_data_list[capacitor_number_in_circuit].capacitor_toml_data

                # catch mypy type issue
                if not isinstance(capacitor_requirements, CapacitorRequirements):
                    raise TypeError("circuit DTO file is incomplete.")
                # Catch mypy issue
                if capacitor_toml_data is None:
                    raise ValueError("Serious programming error in capacitor selection. toml-data are not initialized.",
                                     "Please write an issue!")

                # generate capacitor requirements from circuit simulation data
                capacitor_requirements_dto = pecst.CapacitorRequirements(
                    maximum_peak_to_peak_voltage_ripple=capacitor_toml_data.capacitor_requirements.maximum_peak_to_peak_voltage_ripple,
                    current_waveform_for_op_max_current=np.array([capacitor_requirements.time_vec,
                                                                  capacitor_requirements.current_vec]),
                    v_dc_for_op_max_voltage=capacitor_requirements.v_dc_max,
                    temperature_ambient=capacitor_toml_data.capacitor_requirements.temperature_ambient,
                    voltage_safety_margin_percentage=capacitor_toml_data.capacitor_requirements.voltage_safety_margin_percentage,
                    capacitor_type_list=[pecst.CapacitorType.FilmCapacitor],
                    maximum_number_series_capacitors=capacitor_toml_data.capacitor_requirements.maximum_number_series_capacitors,
                    capacitor_tolerance_percent=pecst.CapacitanceTolerance.TenPercent,
                    lifetime_h=capacitor_toml_data.capacitor_requirements.lifetime_h,
                    results_directory=trial_directory,
                    balancing_discharging_resistors=False)

                # Initialize the statistical data
                stat_data_init: ProgressData = ProgressData(run_time=0, number_of_filtered_points=0, progress_status=ProgressStatus.Idle)

                capacitor_optimization_dto = CapacitorOptimizationDto(
                    circuit_id=circuit_id,
                    progress_data=copy.deepcopy(stat_data_init),
                    capacitor_optimization_dto=capacitor_requirements_dto,
                    capacitor_number_in_circuit=capacitor_requirements.capacitor_number_in_circuit,
                    current_array=capacitor_requirements.current_array,
                    time_array=capacitor_requirements.time_array,
                    factor_dc_losses_min_max_list=capacitor_toml_data.filter_distance.factor_dc_losses_min_max_list)

                # Check list size
                while len(self._optimization_config_list) <= capacitor_number_in_circuit:
                    self._optimization_config_list.append([])

                # Add capacitor dto to the sub-list of assigned number in circuit
                self._optimization_config_list[capacitor_number_in_circuit].append(capacitor_optimization_dto)

    @staticmethod
    def _start_optimization(capacitor_number_in_circuit: int, act_config: CapacitorOptimizationDto, filter_data: FilterData,
                            factor_dc_losses_min_max_list: list[float], debug: Debug) -> int:
        """
        Start the capacitor optimization / selection.

        :param capacitor_number_in_circuit: functional capacitor number in the circuit. Commonly 0 for the input capacitor, 1 for the output capacitor
        :type capacitor_number_in_circuit: int
        :param act_config: capacitor optimization DTO
        :type act_config: CapacitorOptimizationDto
        :param filter_data: filtered circuit data
        :type filter_data: FilterData
        :param factor_dc_losses_min_max_list: minimum and maximum filter factor to filter suitable designs for the Pareto front
        :type factor_dc_losses_min_max_list: list[float]
        :param debug: Debug class
        :type debug: Debug
        """
        # capacitor requirements
        _, c_db_df_list = pecst.select_capacitors(act_config.capacitor_optimization_dto)

        c_db_df = pd.concat(c_db_df_list)

        if not os.path.exists(act_config.capacitor_optimization_dto.results_directory):
            os.makedirs(act_config.capacitor_optimization_dto.results_directory)
        c_db_df.to_csv(f"{act_config.capacitor_optimization_dto.results_directory}/{CAPACITOR_RESULTS}")

        df_filtered = pecst.filter_df(c_db_df, factor_min_dc_losses=factor_dc_losses_min_max_list[0], factor_max_dc_losses=factor_dc_losses_min_max_list[1])
        if debug.general.is_debug:
            # reduce dataset to the given number from the debug configuration
            df_filtered = df_filtered.iloc[:debug.capacitor.number_working_point_max]
        # save Pareto front designs (reduced in case of active debugging)
        df_filtered.to_csv(f"{act_config.capacitor_optimization_dto.results_directory}/{CAPACITOR_RESULTS_FILTERED}")

        all_operation_point_ordering_codes_list = df_filtered["ordering code"].to_numpy()
        all_operation_point_volume_list = df_filtered["volume_total"].to_numpy()
        all_operation_point_area_list = df_filtered["area_total"].to_numpy()
        all_operation_point_n_series_list = df_filtered["in_series_needed"].to_numpy()
        all_operation_point_n_parallel_list = df_filtered["in_parallel_needed"].to_numpy()

        # Overtake the filtered operation points
        number_of_filtered_points = len(all_operation_point_ordering_codes_list)

        logger.info(f"Full-operating point simulation list with {len(all_operation_point_ordering_codes_list)} "
                    f"designs: {all_operation_point_ordering_codes_list}")

        # Initialize volume_total_minimum
        volume_total_minimum = all_operation_point_volume_list[0]

        # simulate all operating points
        for count, ordering_code in enumerate(tqdm.tqdm(all_operation_point_ordering_codes_list)):

            volume_total = all_operation_point_volume_list[count]
            area_total = all_operation_point_area_list[count]
            df_geometry_re_simulation_number = df_filtered[df_filtered["ordering code"] == ordering_code]
            n_series = all_operation_point_n_series_list[count]
            n_parallel = all_operation_point_n_parallel_list[count]

            logger.debug(f"ordering_code: \n"
                         f"    {df_geometry_re_simulation_number.head()}")

            loss_total_array = np.full_like(act_config.current_array[:, :, :, 0], np.nan)

            new_circuit_dto_directory = os.path.join(act_config.capacitor_optimization_dto.results_directory, CIRCUIT_CAPACITOR_LOSS_FOLDER)
            if not os.path.exists(new_circuit_dto_directory):
                os.makedirs(new_circuit_dto_directory)

            if os.path.exists(os.path.join(new_circuit_dto_directory, f"{ordering_code}.pkl")):
                logger.info(f"Re-simulation of {ordering_code} already exists. Skip.")
            else:
                for vec_vvp in np.ndindex(act_config.current_array[:, :, :, 0].shape):
                    time = act_config.time_array[vec_vvp]
                    current = act_config.current_array[vec_vvp]

                    current_waveform = np.array([time, current])
                    logger.debug(f"{current_waveform=}")
                    logger.debug("All operating point simulation of:")
                    logger.debug(f"   * Circuit study: {filter_data.circuit_study_name}")
                    logger.debug(f"   * Circuit trial: {act_config.circuit_id}")
                    logger.debug(f"   * Inductor re-simulation trial: {ordering_code}")

                    [frequency_list, current_amplitude_list, _] = pecst.fft(current_waveform, plot='no', mode='time', title='fft input current')

                    loss_per_capacitor = pecst.power_loss_film_capacitor(ordering_code, frequency_list, current_amplitude_list,
                                                                         number_parallel_capacitors=n_parallel)

                    loss_total_array[vec_vvp] = loss_per_capacitor * n_series * n_parallel

                capacitor_results = CapacitorResults(
                    loss_total_array=loss_total_array,
                    volume_total=volume_total,
                    area_total=area_total,
                    circuit_id=act_config.circuit_id,
                    capacitor_id=df_geometry_re_simulation_number['ordering code'].values[0],
                    n_series=n_series,
                    n_parallel=n_parallel,
                    capacitor_number_in_circuit=capacitor_number_in_circuit
                )

                # Get capacitor with the smallest volume (power loss can be ignored due to very small impact)
                if volume_total <= volume_total_minimum:
                    ordering_code_final = ordering_code
                    capacitor_final_result = copy.deepcopy(capacitor_results)
                    volume_total_minimum = volume_total

                pickle_file = os.path.join(new_circuit_dto_directory, f"{ordering_code_final}.pkl")
        with open(pickle_file, 'wb') as output:
            pickle.dump(capacitor_final_result, output, pickle.HIGHEST_PROTOCOL)

        # returns the number of filtered results
        return number_of_filtered_points

    def optimization_handler(self, filter_data: FilterData, capacitor_in_circuit: int,
                             debug: Debug) -> None:
        """
        Control the multi simulation processes.

        :param filter_data: Information about the filtered designs
        :type  filter_data: pcdt.FilterData
        :param capacitor_in_circuit: Number of capacitor within topology
        :type  capacitor_in_circuit: int
        :param debug: True to use debug mode which stops earlier
        :type debug: bool
        """
        number_cpus = cpu_count()

        # Check if class is initialized and capacitor_in_circuit is valid
        if len(self._optimization_config_list) == 0:
            raise ValueError("Capacitor selection class is no initialized")
        elif len(self._optimization_config_list) <= capacitor_in_circuit or capacitor_in_circuit < 0:
            raise ValueError(f"Invalid parameter value 'capacitor_in_circuit'={capacitor_in_circuit}.\n"
                             f"Value has to be between 0 and {len(self._optimization_config_list)-1}.")

        with Pool(processes=number_cpus) as pool:
            parameters = []
            for act_optimization_configuration in self._optimization_config_list[capacitor_in_circuit]:
                parameters.append((
                    capacitor_in_circuit,
                    act_optimization_configuration,
                    filter_data,
                    act_optimization_configuration.factor_dc_losses_min_max_list,
                    debug
                ))

            pool.starmap(func=CapacitorSelection._start_optimization, iterable=parameters)
