"""Summary pareto optimizations."""
# python libraries
import os
import pickle
import logging
import threading
import copy

# 3rd party libraries
import pandas as pd
import numpy as np

# own libraries
from pcdt.server_ctl_dtos import ProgressStatus
from pcdt.components.heat_sink_optimization import ThermalCalcSupport
from pcdt.components.capacitor_optimization_dtos import CapacitorResults
from pcdt.components.heat_sink_dtos import HeatSinkBoundaryConditions
from pcdt.toml_checker import TomlHeatSink
from pcdt.datasets_dtos import (FilterData, StudyData, CapacitorConfiguration, InductorConfiguration,
                                TransformerConfiguration, SummaryConfiguration)
from pcdt.topology.circuit_optimization_base import CircuitOptimizationBase
import hct
from pcdt.server_ctl_dtos import ProgressData
from pcdt.server_ctl_dtos import RunTimeMeasurement as RunTime
from pcdt.constant_path import (CIRCUIT_INDUCTOR_RELUCTANCE_LOSSES_FOLDER, CIRCUIT_TRANSFORMER_RELUCTANCE_LOSSES_FOLDER,
                                CIRCUIT_INDUCTOR_FEM_LOSSES_FOLDER, CIRCUIT_TRANSFORMER_FEM_LOSSES_FOLDER,
                                CIRCUIT_CAPACITOR_LOSS_FOLDER, DF_SUMMARY_WITHOUT_HEAT_SINK_WITHOUT_OFFSET,
                                DF_SUMMARY_WITH_HEAT_SINK_WITHOUT_OFFSET, DF_SUMMARY_FINAL, DF_SUMMARY_FINAL_FILTERED_MEAN_LOSS,
                                DF_SUMMARY_FINAL_FILTERED_WEIGHTED_EFFICIENCY)

logger = logging.getLogger(__name__)

class SummaryProcessing:
    """Perform the summary calculation based on optimization results."""

    _s_lock_stat: threading.Lock
    _progress_run_time: RunTime
    _progress_data: ProgressData
    _summary_study_data: StudyData

    # Heat sink boundary condition parameter
    heat_sink_boundary_conditions: HeatSinkBoundaryConditions

    # Thermal calculation support class
    thr_sup: ThermalCalcSupport

    # Dictionaries and list of dictionaries for circuit and components to cool by heat sink
    _summary_configuration_list: list[SummaryConfiguration]

    def __init__(self):
        """Initialize the configuration list for the circuit optimizations."""
        self._s_lock_stat = threading.Lock()
        # Initialize the statistical data (For more configuration it needs to become instance instead of static
        self._progress_run_time = RunTime()
        self._progress_data = ProgressData(run_time=0, number_of_filtered_points=0, progress_status=ProgressStatus.Idle)

        # Heat sink boundary condition parameter
        self.heat_sink_boundary_conditions = HeatSinkBoundaryConditions(0, 0)
        # Thermal calculation support class
        self.thr_sup = ThermalCalcSupport()

        # Dictionary of circuit data and actual study data information
        self._summary_configuration_list = []

    def init_thermal_configuration(self, act_heat_sink_data: TomlHeatSink) -> bool:
        """Initialize the thermal parameter of the connection points for the inductor and transformer.

        :param act_heat_sink_data: toml file with configuration data
        :type act_heat_sink_data: TomlHeatSink

        :return: True, if the thermal parameter of the connection points was successful initialized
        :rtype: bool
        """
        # Variable declaration
        # Return variable initialized to True
        successful_init = True

        self.heat_sink_boundary_conditions = HeatSinkBoundaryConditions(t_ambient=act_heat_sink_data.boundary_conditions.t_ambient,
                                                                        t_hs_max=act_heat_sink_data.boundary_conditions.t_hs_max)
        # Return if initialization was successful performed (True)
        return successful_init

    def get_progress_data(self) -> ProgressData:
        """Provide the progress data of the optimization.

        :return: Progress data: Processing start time, actual processing time, number of filtered operation points and status.
        :rtype: ProgressData
        """
        # Lock statistical performance data access
        with self._s_lock_stat:
            # Check if list is in progress
            if self._progress_data.progress_status == ProgressStatus.InProgress:
                # Update statistical data if optimization is running
                self._progress_data.run_time = self._progress_run_time.get_runtime()

        return copy.deepcopy(self._progress_data)

    @staticmethod
    def generate_component_id_list_from_pkl_files(act_dir_name: str) -> tuple[bool, list[str]]:
        """Generate a list of the IDs from pickle filenames (inductor, transformer, capacitor).

        :param act_dir_name : Name of the directory containing the files
        :type  act_dir_name : str

        :return: tuple of bool and result list. True, if the directory exists and contains minimum one file
        :rtype: tuple
        """
        # Variable declaration
        component_id_list: list[str] = []
        is_component_id_list_generated = False

        # Check if target folder exists
        if os.path.exists(act_dir_name):
            # Create list of filepaths
            file_list = os.listdir(act_dir_name)
            # Filter basename without extension
            for file_name in file_list:
                # Create file path
                file_path = os.path.join(act_dir_name, file_name)
                # Check if it is a file
                if os.path.isfile(file_path):
                    component_id = os.path.splitext(os.path.basename(file_name))[0]
                    # Check file type
                    file_extension = os.path.splitext(os.path.basename(file_name))[1]
                    if file_extension == '.pkl':
                        component_id_list.append(component_id)
                    else:
                        logger.debug(f"File {component_id}{file_extension} has no extension '.pkl'!")
                else:
                    logger.info(f"File'{file_path}' does not exists!")
        else:
            logger.info(f"Path {act_dir_name} does not exists!")

        if component_id_list:
            is_component_id_list_generated = True

        return is_component_id_list_generated, component_id_list

    @staticmethod
    def _remove_invalid_design(act_summary_configuration_list: list[SummaryConfiguration], missing_component_circuit_index: str) -> list[SummaryConfiguration]:
        """Generate a list of the IDs from pickle filenames (inductor, transformer, capacitor).

        :param act_summary_configuration_list : summary configuration list
        :type  act_summary_configuration_list : str
        :param missing_component_circuit_index : Name of the circuit_id, which contains one or more missing components
        :type  missing_component_circuit_index : str
        :return: Summary configuration list without the entry defined by the missing component circuit index
        :rtype:  list[SummaryConfiguration]
        """
        # Variable declaration
        result_summary_configuration_list: list[SummaryConfiguration] = []

        for summary_configuration in act_summary_configuration_list:
            if summary_configuration.circuit_data_list[0]["circuit_id"] is not missing_component_circuit_index:
                result_summary_configuration_list.append(summary_configuration)

        return result_summary_configuration_list

    def initialize_processing(self, act_filter_data: FilterData, act_capacitor_data_list: list[CapacitorConfiguration],
                              act_inductor_data_list: list[InductorConfiguration],
                              act_transformer_data_list: list[TransformerConfiguration],
                              summary_study_data: StudyData, is_pre_summary: bool) -> None:
        """Initialize lists for from analytic or simulation run to perform the summary.

        :param act_filter_data: circuit design list
        :type act_filter_data: FilterData
        :param act_capacitor_data_list: List of capacitor configurations
        :type  act_capacitor_data_list: list[CapacitorConfiguration]
        :param act_inductor_data_list: List of inductor configurations
        :type act_inductor_data_list: list[InductorConfiguration]
        :param act_transformer_data_list: transformer study data
        :type act_transformer_data_list: list[TransformerConfiguration]
        :param summary_study_data: Study data for pre-summary or summary
        :type  summary_study_data: StudyData
        :param is_pre_summary: True for pre-summary, False for summary
        :type is_pre_summary: bool
        """
        # Variable declaration

        if is_pre_summary:
            # pre summary using reluctance model results (inductive components)
            inductor_result_directory = CIRCUIT_INDUCTOR_RELUCTANCE_LOSSES_FOLDER
            transformer_result_directory = CIRCUIT_TRANSFORMER_RELUCTANCE_LOSSES_FOLDER
        else:
            # final summary using FEM results (inductive components)
            inductor_result_directory = CIRCUIT_INDUCTOR_FEM_LOSSES_FOLDER
            transformer_result_directory = CIRCUIT_TRANSFORMER_FEM_LOSSES_FOLDER

        # Set study data
        self._summary_study_data = summary_study_data

        # Index of designs, which miss a component solution (No pkl-file available for this design)
        missing_component_index_list: list[str] = []

        circuits: list[dict] = []
        # iterate circuit numbers
        for circuit_id in act_filter_data.filtered_list_files:

            # Define tuple data
            circuit_data_list: list[dict] = []
            capacitor_data_list: list[list[dict]] = []
            inductor_data_list: list[list[dict]] = []
            transformer_data_list: list[list[dict]] = []

            # Assemble pkl-filename
            circuit_id_filepath = os.path.join(act_filter_data.filtered_list_pathname, f"{circuit_id}.pkl")

            # Get circuit results
            with open(circuit_id_filepath, 'rb') as pickle_file_data:
                circuit_dto = pickle.load(pickle_file_data)

            for number_in_circuit, act_circuit_data in enumerate(circuit_dto.circuit_thermal):
                # Calculate the thermal values
                if not circuit_dto.calc_losses:  # mypy avoid follow-up issues
                    raise ValueError("Incomplete loss calculation.")

                logger.debug(f"{act_filter_data.circuit_study_name=}")

                circuit_data = {
                    "circuit_id": circuit_id,
                    f"circuit_t_j_max_{number_in_circuit}": act_circuit_data.t_j_max,
                    f"circuit_r_th_ib_jhs_{number_in_circuit}": act_circuit_data.r_th_jhs,
                    f"circuit_area_{number_in_circuit}": act_circuit_data.area,
                    f"circuit_loss_array_{number_in_circuit}": act_circuit_data.loss_array,
                    f"circuit_temperature_heat_sink_max_array_{number_in_circuit}": act_circuit_data.temperature_heat_sink_max_array
                }
                # Store capacitor data set
                circuit_data_list.append(circuit_data)

            # iterate capacitor selection
            for act_capacitor_data in act_capacitor_data_list:
                # Assemble directory name for capacitor results
                capacitor_filepath_results = os.path.join(act_capacitor_data.study_data.optimization_directory,
                                                          circuit_id,
                                                          act_capacitor_data.study_data.study_name,
                                                          CIRCUIT_CAPACITOR_LOSS_FOLDER)

                # Check, if capacitor  number list cannot be generated
                is_capacitor_list_generated, capacitor_id_list = (SummaryProcessing.generate_component_id_list_from_pkl_files(capacitor_filepath_results))
                if not is_capacitor_list_generated:
                    logger.info(f"Path {capacitor_filepath_results} does not exists or does not contains any pkl-files!\n"
                                f"circuit design {circuit_id} cannot be realized!")
                    # Add circuit index to missing component index list
                    missing_component_index_list.append(circuit_id)
                    # Next circuit
                    continue
                logger.debug(f"{capacitor_id_list=}")

                # iterate capacitor numbers
                capacitors: list[dict] = []
                for capacitor_id in capacitor_id_list:
                    capacitor_filepath_number = os.path.join(capacitor_filepath_results, f"{capacitor_id}.pkl")

                    # get capacitor results
                    with open(capacitor_filepath_number, 'rb') as pickle_file_data:
                        capacitor_dto: CapacitorResults = pickle.load(pickle_file_data)

                    if capacitor_dto.circuit_id != circuit_id:
                        raise ValueError(f"{capacitor_dto.circuit_id=} != {circuit_id}")
                    if capacitor_dto.capacitor_id != capacitor_id:
                        raise ValueError(f"{capacitor_dto.capacitor_id=} != {capacitor_id}")

                    # Get number in circuit for the data frames
                    number_in_circuit = capacitor_dto.capacitor_number_in_circuit
                    # Assemble the data frame entry
                    capacitor_data = {
                        # capacitor
                        f"capacitor_study_name_{number_in_circuit}": act_capacitor_data.study_data.study_name,
                        f"capacitor_id_{number_in_circuit}": capacitor_id,
                        f"capacitor_volume_{number_in_circuit}": capacitor_dto.volume_total,
                        f"capacitor_loss_array_{number_in_circuit}": capacitor_dto.loss_total_array,
                        f"capacitor_area_{number_in_circuit}": capacitor_dto.area_total,
                        f"capacitor_n_parallel_{number_in_circuit}": capacitor_dto.n_parallel,
                        f"capacitor_n_series_{number_in_circuit}": capacitor_dto.n_series,
                    }
                    # Store capacitor data set
                    capacitors.append(capacitor_data)

                # Add to capacitor component list
                capacitor_data_list.append(capacitors)

            # iterate inductor study
            for act_inductor_data in act_inductor_data_list:
                inductor_study_results_filepath = os.path.join(act_inductor_data.study_data.optimization_directory,
                                                               circuit_id,
                                                               act_inductor_data.study_data.study_name,
                                                               inductor_result_directory)

                # Generate magnetic list
                is_inductor_list_generated, inductor_id_list = (
                    SummaryProcessing.generate_component_id_list_from_pkl_files(inductor_study_results_filepath))

                if not is_inductor_list_generated:
                    logger.info(f"Path {inductor_study_results_filepath} does not exists or does not contains any pkl-files!\n"
                                f"circuit design {circuit_id} cannot be realized!")
                    # Add circuit index to missing component index list
                    missing_component_index_list.append(circuit_id)
                    # Next circuit
                    continue

                # iterate inductor numbers
                logger.debug(f"{inductor_id_list=}")
                inductors: list[dict] = []
                for inductor_id in inductor_id_list:
                    inductor_id_filepath = os.path.join(inductor_study_results_filepath, f"{inductor_id}.pkl")

                    # Get inductor results
                    with open(inductor_id_filepath, 'rb') as pickle_file_data:
                        inductor_results_dto = pickle.load(pickle_file_data)

                    if inductor_results_dto.circuit_id != circuit_id:
                        raise ValueError(f"{inductor_results_dto.circuit_id=} != {circuit_id}")
                    if int(inductor_results_dto.inductor_id) != int(inductor_id):
                        raise ValueError(f"{inductor_results_dto.inductor_id=} != {inductor_id}")

                    inductance_loss_matrix = inductor_results_dto.loss_array

                    logger.debug(f"{act_inductor_data.study_data.study_name=}")

                    # Get number in circuit for the data frames
                    number_in_circuit = inductor_results_dto.inductor_number_in_circuit

                    # Assemble the data frame entry
                    inductor_data = {
                        # inductor
                        f"inductor_study_name_{number_in_circuit}": act_inductor_data.study_data.study_name,
                        f"inductor_id_{number_in_circuit}": inductor_id,
                        f"inductor_volume_{number_in_circuit}": inductor_results_dto.volume,
                        f"inductor_loss_array_{number_in_circuit}": inductance_loss_matrix,
                        f"inductor_t_max_{number_in_circuit}": 0,
                        f"inductor_area_{number_in_circuit}": inductor_results_dto.area_to_heat_sink,
                        f"r_th_ind_heat_sink_{number_in_circuit}": inductor_results_dto.r_th_ind_heat_sink
                    }
                    # Store inductor data set
                    inductors.append(inductor_data)

                # Add to inductor component list
                inductor_data_list.append(inductors)

            # iterate transformer study
            for act_transformer_data in act_transformer_data_list:
                stacked_transformer_study_results_filepath = os.path.join(act_transformer_data.study_data.optimization_directory,
                                                                          circuit_id,
                                                                          act_transformer_data.study_data.study_name,
                                                                          transformer_result_directory)

                # Check, if stacked transformer number list cannot be generated
                is_transformer_list_generated, transformer_id_list = (
                    SummaryProcessing.generate_component_id_list_from_pkl_files(stacked_transformer_study_results_filepath))

                if not is_transformer_list_generated:
                    logger.info(f"Path {stacked_transformer_study_results_filepath} does not exists or does not contains any pkl-files!\n"
                                f"circuit design {circuit_id} cannot be realized!")
                    # Add circuit index to missing component index list
                    missing_component_index_list.append(circuit_id)
                    # Next circuit
                    continue

                logger.debug(f"{transformer_id_list=}")

                # iterate transformer numbers
                transformers: list[dict] = []
                for transformer_id in transformer_id_list:
                    transformer_id_filepath = os.path.join(stacked_transformer_study_results_filepath, f"{transformer_id}.pkl")

                    # get transformer results
                    with open(transformer_id_filepath, 'rb') as pickle_file_data:
                        transformer_results_dto = pickle.load(pickle_file_data)

                    if transformer_results_dto.circuit_id != circuit_id:
                        raise ValueError(f"{transformer_results_dto.circuit_id=} != {circuit_id}")
                    if int(transformer_results_dto.transformer_id) != int(transformer_id):
                        raise ValueError(f"{transformer_results_dto.transformer_id=} != {transformer_id}")

                    logger.debug(f"{act_transformer_data.study_data.study_name=}")

                    # Get number in circuit for the data frames
                    number_in_circuit = transformer_results_dto.transformer_number_in_circuit
                    # Assemble the data frame entry
                    transformer_data = {
                        # transformer
                        f"transformer_study_name_{number_in_circuit}": act_transformer_data.study_data.study_name,
                        f"transformer_id_{number_in_circuit}": transformer_id,
                        f"transformer_volume_{number_in_circuit}": transformer_results_dto.volume,
                        f"transformer_loss_array_{number_in_circuit}": transformer_results_dto.loss_array,
                        f"transformer_t_max_{number_in_circuit}": 0,
                        f"transformer_area_{number_in_circuit}": transformer_results_dto.area_to_heat_sink,
                        f"r_th_xfmr_heat_sink_{number_in_circuit}": transformer_results_dto.r_th_xfmr_heat_sink
                    }

                    # Store transformer data set
                    transformers.append(transformer_data)

                # Add to inductor component list
                transformer_data_list.append(transformers)

            # Assemble design component tuple for actual circuit design requiring a heat sink
            self._summary_configuration_list.append(SummaryConfiguration(circuit_data_list=circuit_data_list,
                                                                         inductor_data_list=inductor_data_list,
                                                                         capacitor_data_list=capacitor_data_list,
                                                                         transformer_data_list=transformer_data_list))

        # Check, if any circuit design cannot be realized
        if missing_component_index_list:
            for missing_component in missing_component_index_list:
                # Remove circuit from summary configuration lists
                self._summary_configuration_list = SummaryProcessing._remove_invalid_design(
                    self._summary_configuration_list, missing_component)

    @staticmethod
    def component_to_dataframe(component_data: list[dict]) -> pd.DataFrame:
        """Generate a pd-dataframe form a list of dictionaries.

        :param component_data: List with component data within a dict
        :type  component_data: list[dict]
        :return: Data frame with result data of the dictionary
        :rtype:  pd.DataFrame
        """
        df_components = pd.DataFrame()
        for single_component_data in component_data:
            df_single_component = pd.DataFrame([single_component_data])
            df_components = pd.concat([df_components, df_single_component], axis=0)

        return df_components

    @staticmethod
    def _calculate_component_sum(act_df: pd.DataFrame, key_prefix: str) -> pd.Series | int:
        """
        Calculate the sum of pandas dataframe columns starting with key_prefix.

        :param act_df: Actual data frame, which shall be filtered for prefix columns
        :type  act_df: pd.DataFrame
        :param key_prefix: Name of the component prefix
        :type  key_prefix: str
        :return: pandas series with result data of the dictionary
        :rtype:  pd.Series
        """
        # Get index
        cols = act_df.columns[act_df.columns.str.startswith(key_prefix)]
        # Check, if no column with the prefix is found.
        if len(cols) == 0:
            return 0
        return act_df[cols].sum(axis=1)

    @staticmethod
    def _calculate_component_mean(act_df: pd.DataFrame, key_prefix: str) -> pd.Series | int:
        """Generate a pd-dataframe form a list of dictionaries.

        :param act_df: Actual data frame, which shall be filtered for prefix columns
        :type  act_df: pd.DataFrame
        :param key_prefix: Name of the component prefix
        :type  key_prefix: str
        :return: pandas series with result data of the dictionary
        :rtype:  pd.Series
        """
        # Get index
        cols = act_df.columns[act_df.columns.str.startswith(key_prefix)]
        # Check, if no column with the prefix is found.
        if len(cols) == 0:
            return 0

        # Initialize result
        result_series = 0

        # Calculate the mean value
        for col in cols:
            result_series = result_series + act_df[col].apply(lambda arr: arr.mean())

        return result_series

    @staticmethod
    def _calculate_minimum_component_temperature(act_df: pd.DataFrame, component_name: str) -> pd.Series | int:
        """Calculate the minimum temperature based on thermal data and power loss.

        :param act_df: Actual data frame, which shall be filtered for prefix columns
        :type  act_df: pd.DataFrame
        :param component_name: Name of the component prefix
        :type  component_name: str
        :return: Data frame with result data of the dictionary or 0
        :rtype:  pd.DataFrame | int
        """
        # Variable declaration
        components: list[str] = ["inductor", "transformer"]
        component_key_prefix: list[list[str]] = [["r_th_ind_heat_sink", "inductor_loss_array", "temperature_inductor_heat_sink_max_array"],
                                                 ["r_th_xfmr_heat_sink", "transformer_loss_array", "temperature_xfmr_heat_sink_max_array"]]
        component_result_key_prefix: str = "t_min_array"
        component_index: int = -1
        result_df = pd.DataFrame()

        # Identify the component
        for index, component in enumerate(components):
            if component == component_name:
                component_index = index
                break

        # Check if component is found
        if component_index == -1:
            return 0

        # Get index
        r_th_cols = act_df.columns[act_df.columns.str.startswith(component_key_prefix[component_index][0])]

        # Check, if no column with the prefix is found.
        if len(r_th_cols) == 0:
            return 0

        # Perform calculation with each identified column
        for r_th_col in r_th_cols:
            suffix = r_th_col[len(component_key_prefix[component_index][0]):]
            loss_array_name = component_key_prefix[component_index][1] + suffix
            temperature_name = component_key_prefix[component_index][2] + suffix
            result_name = component_result_key_prefix + suffix
            act_df[temperature_name] = 125 - act_df[r_th_col] * act_df[loss_array_name]
            # Calculate result by selection of minimal temperature of all components of the selected type
            result_df[result_name] = act_df[temperature_name].apply(lambda arr: arr.min())

        # Get result index
        result_cols = result_df.columns[result_df.columns.str.startswith(component_result_key_prefix)]
        # calculate total minimum
        total_result_df = result_df[result_cols].min(axis=1)
        return total_result_df

    @staticmethod
    def _get_minimum_temperature_value(total_min_temperature: pd.Series, component_temperature: pd.Series) -> pd.Series:
        """Generate a pd-dataframe form a list of dictionaries..

        :param act_df: Actual data series, which shall be filtered for prefix columns
        :type  act_df: pd.Series
        :param component_temperature: Data series with temperature to compare
        :type  component_temperature: pd.Series
        :return: Data series containing the minimal temperature values of both input data frames
        :rtype:  pd.Series
        """
        # Concatenate both data series to a data frame
        result_df = pd.concat([total_min_temperature, component_temperature], axis=1)
        # Calculate the minimum per line
        result_df['min_new_value'] = result_df.min(axis=1)

        return result_df['min_new_value']

    @staticmethod
    def _backup_columns(act_df: pd.DataFrame, column_name_prefix_list: list[str]) -> list[tuple[str, pd.Series]]:
        """Generate a list of tuples consists of column name and the column of the data frame.

        This method backs up all columns, which starts with the names in the column_name_prefix_list.
        Each tuple contains the columns name and the correspondent column.

        :param act_df: Actual data frame
        :type  act_df: pd.DataFrame
        :param column_name_prefix_list: List of column name prefix, which are to copy
        :type  column_name_prefix_list: list[str]
        :return: Data series containing the minimal temperature values of both input data frames
        :rtype:  pd.Series
        """
        # Variable declaration
        df_backup_list: list[tuple[str, pd.Series]] = []

        for column_name_prefix in column_name_prefix_list:
            # Get columns names
            column_name_list = act_df.columns[act_df.columns.str.startswith(column_name_prefix)]
            for column_name in column_name_list:
                df_backup_list.append((column_name, act_df[column_name]))

        return df_backup_list

    @staticmethod
    def _convert_for_csv(act_df: pd.DataFrame, column_name_prefix_list: list[str]) -> pd.DataFrame:
        """
        Generate a list of tuples consists of column name and the column of the data frame.

        This method backs up all columns, which starts with the names in the column_name_prefix_list.
        Each tuple contains the columns name and the correspondent column.

        :param act_df: Actual data frame
        :type  act_df: pd.DataFrame
        :param column_name_prefix_list: List of column name prefix, which are to copy
        :type  column_name_prefix_list: list[str]
        :return: Data series containing the minimal temperature values of both input data frames
        :rtype:  pd.Series
        """
        for column_name_prefix in column_name_prefix_list:
            # Get columns names
            column_name_list = act_df.columns[act_df.columns.str.startswith(column_name_prefix)]
            for column_name in column_name_list:
                # Convert the column and overwrite the dataframe
                act_df[column_name] = act_df[column_name].apply(lambda x: str(x.tolist()))
        return act_df

    def generate_cooling_requirement_database(self, heat_sink_boundary_conditions: HeatSinkBoundaryConditions) -> pd.DataFrame:
        """Generate a database df by summaries the calculation results for all components required a heat sink.

        :param heat_sink_boundary_conditions: Boundary conditions for the heat sink
        :type heat_sink_boundary_conditions: HeatSinkBoundaryConditions
        :return: DataFrame with result information of the pareto front
        :rtype:  pd.DataFrame
        """
        # Variable declaration

        # Check if initialization list is empty
        if not self._summary_configuration_list:
            raise ValueError("First you need to initialize summary by calling 'initialize_processing'!")

        # Start the progress time measurement
        with self._s_lock_stat:
            self._progress_run_time.reset_start_trigger()
            self._progress_data.run_time = self._progress_run_time.get_runtime()
            self._progress_data.progress_status = ProgressStatus.InProgress

        # Result DataFrame
        df = pd.DataFrame()

        # iterate circuit numbers
        for design in self._summary_configuration_list:

            # Put all inductors to a pd-design
            # append two dictionaries
            circuit_dict: dict = {}
            for dictionary in design.circuit_data_list:
                if "circuit_id" in circuit_dict:
                    if circuit_dict["circuit_id"] != dictionary["circuit_id"]:
                        raise ValueError("Programming Error: circuit_id's should be equal, but they aren't.")

                circuit_dict.update(dictionary)

            df_circuit_local = pd.DataFrame([circuit_dict])

            # Merge different df by creating a common key
            df_circuit_local['key'] = 0

            # Put all inductors to a pd-design
            inductor_pd_list: list[pd.DataFrame] = []
            for inductor_data in design.inductor_data_list:
                df_inductors = SummaryProcessing.component_to_dataframe(inductor_data)
                # Merge different df by creating a common key
                df_inductors['key'] = 0
                inductor_pd_list.append(df_inductors)

            # Put all transformers to a pd-design
            transformer_pd_list: list[pd.DataFrame] = []
            for transformer_data in design.transformer_data_list:
                df_transformers = SummaryProcessing.component_to_dataframe(transformer_data)
                # Merge different df by creating a common key
                df_transformers['key'] = 0
                transformer_pd_list.append(df_transformers)

            # Perform cross join
            df_merge_data = df_circuit_local

            for df_component in inductor_pd_list:
                df_merge_data = df_merge_data.merge(df_component, on='key', how='outer')

            for df_component in transformer_pd_list:
                df_merge_data = df_merge_data.merge(df_component, on='key', how='outer')

            # Add design to main data frame
            df = pd.concat([df, df_merge_data], axis=0)

        # Calculate the total area as sum of circuit,  inductor and transformer area based on df-command is like vector sum v1[:]= v2[:] + v3[:]
        # and v2 = sum_row [matrix m2[:]]; v3 = sum_row [matrix m3[:]] The number of columns of the matrix corresponds to
        # the number of components of the same type. heat sink area, capacitors do not need heat sink area
        # df["total_area"] = df["circuit_area"].apply(lambda x: np.sum(x)) + df["inductor_area"] + df["transformer_area"]
        df["total_cooling_array"] = (SummaryProcessing._calculate_component_sum(df, "circuit_area") + \
                                     SummaryProcessing._calculate_component_sum(df, "inductor_area") + \
                                     SummaryProcessing._calculate_component_sum(df, "transformer_area"))

        # Calculate the power loss of the design without capacitors loss
        df["total_loss_wo_capacitors_array"] = (SummaryProcessing._calculate_component_sum(df, "circuit_loss_array") + \
                                                SummaryProcessing._calculate_component_sum(df, "inductor_loss_array") + \
                                                SummaryProcessing._calculate_component_sum(df, "transformer_loss_array"))

        # maximum heat sink temperatures (minimum of all the maximum temperatures of single components)
        cols = df.columns[df.columns.str.startswith("circuit_temperature_heat_sink_max_array")]
        numbers = [col.replace("circuit_temperature_heat_sink_max_array_", "") for col in cols]
        for count, col in enumerate(cols):
            df[f"t_min_array_{numbers[count]}"] = df[col].apply(lambda arr: arr.min())

        df["t_min_array"] = df[df.columns[df.columns.str.startswith("t_min_array")]].min(axis=1)
        # Calculate the maximal temperature for all inductors
        component_temperature_series = SummaryProcessing._calculate_minimum_component_temperature(df, "inductor")
        if isinstance(component_temperature_series, pd.Series):
            df["t_min_array"] = SummaryProcessing._get_minimum_temperature_value(df["t_min_array"], component_temperature_series)
        # Calculate the maximal temperature for all transformers
        component_temperature_series = SummaryProcessing._calculate_minimum_component_temperature(df, "transformer")
        if isinstance(component_temperature_series, pd.Series):
            df["t_min_array"] = SummaryProcessing._get_minimum_temperature_value(df["t_min_array"], component_temperature_series)

        # Calculate boundary condition
        df["t_min_array"] = df.apply(lambda x: np.minimum(x["t_min_array"], heat_sink_boundary_conditions.t_hs_max), axis=1)
        # maximum delta temperature over the heat sink
        df["delta_t_max_heat_sink_array"] = df["t_min_array"] - heat_sink_boundary_conditions.t_ambient
        df["r_th_heat_sink_target_array"] = df["delta_t_max_heat_sink_array"] / df["total_loss_wo_capacitors_array"]
        df["r_th_heat_sink"] = df["r_th_heat_sink_target_array"].apply(lambda x: x.min())

        # Save results to file (ASA : later to store only on demand)
        df.to_csv(f"{self._summary_study_data.optimization_directory}/{DF_SUMMARY_WITHOUT_HEAT_SINK_WITHOUT_OFFSET}")

        # Start the progress time measurement
        with self._s_lock_stat:
            self._progress_run_time.stop_trigger()
            self._progress_data.run_time = self._progress_run_time.get_runtime()

        # return the database
        return df

    def select_heat_sink_configuration(self, heat_sink_study_data: StudyData, act_df_for_hs: pd.DataFrame) -> pd.DataFrame:
        """Select the heat sink configuration from calculated heat sink pareto front.

        :param heat_sink_study_data: Information about the heat sink study name and study path
        :type  heat_sink_study_data: StudyData
        :param act_df_for_hs: DataFrame with result information of the pareto front for heat sink selection
        :type  act_df_for_hs: pd.DataFrame
        :return: pandas data frame including heat sink volume
        :rtype: pd.DataFrame
        """
        # Variable declaration

        # Continue the progress time measurement
        with self._s_lock_stat:
            self._progress_run_time.continue_trigger()
            self._progress_data.run_time = self._progress_run_time.get_runtime()
            self._progress_data.progress_status = ProgressStatus.InProgress

        # load heat sink
        hs_config_filepath = os.path.join(heat_sink_study_data.optimization_directory,
                                          f"{heat_sink_study_data.study_name}.pkl")
        hs_config = hct.Optimization.load_config(hs_config_filepath)
        # Debug ASA Missing true simulations for remaining function

        df_hs = hct.Optimization.study_to_df(hs_config)

        # generate full summary as panda database operation
        logger.info("Select smallest heat sink volume")
        act_df_for_hs["heat_sink_volume"] = act_df_for_hs["r_th_heat_sink"].apply(
            lambda r_th_max: df_hs.loc[df_hs["values_1"] < r_th_max]["values_0"].nsmallest(n=1).values[0] \
            if np.any(df_hs.loc[df_hs["values_1"] < r_th_max]["values_0"].nsmallest(n=1).values) else None)

        # add heat sink ID
        logger.info("Add heat sink ID for smallest heat sink volume")
        act_df_for_hs["heat_sink_id"] = act_df_for_hs["r_th_heat_sink"].apply(
            lambda r_th_max: df_hs.loc[df_hs["values_1"] < r_th_max]["values_0"].nsmallest(n=1).index[0] \
            if np.any(df_hs.loc[df_hs["values_1"] < r_th_max]["values_0"].nsmallest(n=1).values) else None)

        # save full summary
        act_df_for_hs.to_csv(f"{self._summary_study_data.optimization_directory}/{DF_SUMMARY_WITH_HEAT_SINK_WITHOUT_OFFSET}")

        # Update statistical data for summary processing finalized
        with self._s_lock_stat:
            self._progress_run_time.stop_trigger()
            self._progress_data.run_time = self._progress_run_time.get_runtime()
            self._progress_data.progress_status = ProgressStatus.Done

        return act_df_for_hs

    def generate_result_database(self, act_df: pd.DataFrame, control_board_volume: float, control_board_loss: float,
                                 output_power: np.ndarray, weights: np.ndarray) -> pd.DataFrame:
        """
        Finalize database by add components without required heat sink and add remaining volumes and losses.

        :param act_df: dataframe including the selected heat sink
        :type act_df: pd.DataFrame
        :param control_board_volume: control board volume in m³
        :type control_board_volume: float
        :param control_board_loss: control board loss in W
        :type control_board_loss: float
        :param output_power: output power per operating point
        :type output_power: list[float]
        :param weights: weight power operating point
        :type weights: list[float]
        :return: pandas dataframe including the offset volume and offset losses
        :rtype: pd.DataFrame
        """
        # Variable declaration
        # Result DataFrame
        local_df = pd.DataFrame()

        # Check if initialization list is empty
        if not self._summary_configuration_list:
            raise ValueError("First you need to initialize summary by calling 'initialize_processing'!")

        # iterate circuit numbers
        for design in self._summary_configuration_list:
            df_merge_data = pd.DataFrame()
            df_merge_data['key'] = 0
            # Put all capacitors to a pd-design
            capacitor_pd_list: list[pd.DataFrame] = []
            for capacitor_data in design.capacitor_data_list:
                df_capacitors = SummaryProcessing.component_to_dataframe(capacitor_data)
                df_capacitors['key'] = 0
                capacitor_pd_list.append(df_capacitors)

            # Perform cross join
            for df_component in capacitor_pd_list:
                df_merge_data = df_merge_data.merge(df_component, on='key', how='outer')

            # Add circuit id for merge to main dataframe
            df_merge_data["circuit_id"] = design.circuit_data_list[0]["circuit_id"]

            # Add design to main data frame
            local_df = pd.concat([local_df, df_merge_data], axis=0)

        # Merge, if this component is part of the design
        if not local_df.empty:
            act_df = pd.merge(act_df, local_df, on='circuit_id')

        # Consume  design component data
        self._summary_configuration_list = []

        # Calculate the volume without heat sink volume
        act_df["volume_wo_heat_sink"] = (
            SummaryProcessing._calculate_component_sum(act_df, "inductor_volume") +\
            SummaryProcessing._calculate_component_sum(act_df, "transformer_volume") +\
            SummaryProcessing._calculate_component_sum(act_df, "capacitor_volume"))

        # Calculate the total volume
        act_df["total_volume"] = act_df["volume_wo_heat_sink"] + act_df["heat_sink_volume"] +\
            + control_board_volume
        act_df["total_loss_array"] = (
            act_df["total_loss_wo_capacitors_array"] + control_board_loss + \
            SummaryProcessing._calculate_component_sum(act_df, "capacitor_loss_array")
        )

        # Calculate the total mean loss
        act_df["total_mean_loss"] = (
            SummaryProcessing._calculate_component_mean(act_df, "circuit_loss_array") + \
            SummaryProcessing._calculate_component_mean(act_df, "capacitor_loss_array") + \
            SummaryProcessing._calculate_component_mean(act_df, "inductor_loss_array") + \
            SummaryProcessing._calculate_component_mean(act_df, "transformer_loss_array") + \
            control_board_loss)

        act_df["efficiency"] = act_df["total_loss_array"].apply(lambda x: (np.abs(output_power) - x) / np.abs(output_power))
        act_df["weighted_efficiency"] = act_df["efficiency"].apply(lambda x: np.sum(x * weights))

        # generate a new unique index for the combined dataframe
        # this helps to easily address unique combinations by the index
        act_df = act_df.reset_index(drop=True)
        act_df.index.name = "combination_id"

        df_backup_list = SummaryProcessing._backup_columns(
            act_df, ["total_loss_array", "circuit_loss_array", "capacitor_loss_array",
                     "inductor_loss_array", "transformer_loss_array"])

        # Convert for saving data in CSV-format
        act_df = SummaryProcessing._convert_for_csv(
            act_df, ["total_loss_array", "circuit_loss_array", "capacitor_loss_array",
                     "inductor_loss_array", "transformer_loss_array""circuit_loss_array"])

        act_df.to_csv(f"{self._summary_study_data.optimization_directory}/{DF_SUMMARY_FINAL}")

        # Restore list
        for colname, series_backup in df_backup_list:
            act_df[colname] = series_backup

        return act_df

    def filter_losses(self, df: pd.DataFrame, abs_max_losses: float, factor_min_max_losses_list: list[float]) -> pd.DataFrame:
        """
        Pareto front filter for losses vs. volume.

        :param df: dataframe
        :type df: pd.DataFrame
        :param abs_max_losses: absolute maximum losses of the converter to clip the Pareto front
        :type abs_max_losses: float
        :param factor_min_max_losses_list: list with factors for minimum and maximum losses (relative to minimum losses)
        :type factor_min_max_losses_list: list[float, float]
        :return:
        """
        df_filtered = CircuitOptimizationBase.filter_df_min_min(df, x="total_volume", y="total_mean_loss",
                                                                factor_relative_y_min_offset=factor_min_max_losses_list[0],
                                                                factor_relative_y_max_offset=factor_min_max_losses_list[1],
                                                                absolute_max_y=abs_max_losses)
        filename = f"{self._summary_study_data.optimization_directory}/{DF_SUMMARY_FINAL_FILTERED_MEAN_LOSS}"
        df_filtered.to_csv(filename)
        logger.info(f"Save filtered Pareto front (regarding losses) to {filename}")
        return df_filtered

    def filter_efficiency(self, df: pd.DataFrame, filter_distance: list[float]) -> pd.DataFrame:
        """
        Pareto front filter for efficiency vs. volume.

        :param df: dataframe with Pareto plane
        :type df: pd.DataFrame
        :param filter_distance: [relative y offset (efficiency offset), absolute minimum efficiency]
        :type filter_distance: list[float]
        :return: Filtered Pareto front in a data frame
        :rtype: pd.DataFrame
        """
        df_filtered = CircuitOptimizationBase.filter_df_min_max(df, x="total_volume", y="weighted_efficiency",
                                                                relative_y_offset=filter_distance[0], absolute_min_y=filter_distance[1])
        filename = f"{self._summary_study_data.optimization_directory}/{DF_SUMMARY_FINAL_FILTERED_WEIGHTED_EFFICIENCY}"
        df_filtered.to_csv(filename)
        logger.info(f"Save filtered Pareto front (regarding efficiency) to {filename}")
        return df_filtered
