"""Main control program to optimize the DAB converter."""

# python libraries
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Any
import os

# 3rd party libraries
from pydantic import BaseModel
import optuna
import pandas as pd
import numpy as np

# Own libraries
from pcdt.datasets_dtos import (FilterData, StudyData, PlotData, CapacitorConfiguration,
                                InductorConfiguration, TransformerConfiguration)
from pcdt.server_ctl_dtos import ProgressData
from pcdt.components.component_dtos import CapacitorRequirements, InductorRequirements, TransformerRequirements
from pcdt.circuit_enums import CalcModeEnum, TemplateType, TopologyEnum
from pcdt.constant_path import FILTERED_RESULTS_PATH

# Type of general optimization parameter
T_G_D = TypeVar("T_G_D", bound="TomlGData")
# Type of circuit optimization parameter
T_C_D = TypeVar("T_C_D", bound="TomlCData")


class TomlGData(BaseModel, ABC):
    """Contains the general toml data."""

    pass

class TomlCData(BaseModel, ABC):
    """Contains the circuit toml data."""

    pass


class CircuitOptimizationBase(Generic[T_G_D, T_C_D], ABC):
    """Represent the base class for electrical converter optimization depending on the topology."""

    # Thermal resistance
    r_th_per_unit_area_ind_heat_sink = 0
    r_th_per_unit_area_xfmr_heat_sink = 0

    # control board
    misc: float

    def __init__(self) -> None:
        """Initialize the member variables."""
        self.project_directory: str = ""
        self.circuit_study_data: StudyData = StudyData("", "", CalcModeEnum.new_mode)

    def init_study_information(self, study_name: str, project_directory: str,
                               sub_directory: str, calculation_mode: CalcModeEnum) -> None:
        """Initialize the study information.

        :param study_name: Name of the study
        :type  study_name: str
        :param project_directory: Project directory name
        :type  project_directory: Project folder path
        :param sub_directory: Sub directory of the circuit optimization
        :type  sub_directory: Sub directory path
        :param calculation_mode: Calculation mode of the circuit optimization
        :type  calculation_mode: CalcModeEnum
        """
        # Project directory
        self.project_directory = project_directory
        # Circuit path
        self.circuit_study_data = StudyData(
            study_name=study_name, optimization_directory=os.path.join(project_directory, sub_directory, study_name),
            calculation_mode=calculation_mode)

        # Filtered data path and list
        self.filter_data = FilterData(
            circuit_study_name=self.circuit_study_data.study_name,
            filtered_list_pathname=os.path.join(self.circuit_study_data.optimization_directory, FILTERED_RESULTS_PATH)
        )

    def is_circuit_optimization_skippable(self) -> tuple[bool, str]:
        """Control procedure of skippable optimization check.

        :return: True, if the optimization is skippable and empty string or False and report of the issues
        :rtype: tuple[bool, str]
        """
        if self.circuit_study_data is None and self.filter_data is None:
            # The method to initialize study information needs to be called first.
            # This has to be guaranteed by the workflow
            return False, ("Serious programming error 'Allocation issue circuit_study_data or filter_data'.\n"
                           "Please write an issue!")

        # Check, if all data are available
        is_skipable, issue_report = self.__class__._is_optimization_skippable(self.circuit_study_data, self.filter_data)

        # return result
        return is_skipable, issue_report

    @staticmethod
    def filter_df_min_min(df: pd.DataFrame, x: str = "values_0", y: str = "values_1", factor_relative_y_min_offset: float = 1.2,
                          factor_relative_y_max_offset: float = 10, absolute_max_y: float = 100_000) -> pd.DataFrame:
        """
        Remove designs with too high y-values (e.g. losses) compared to the minimum y-value (e.g. minimum loss).

        :param df: pandas DataFrame with study results
        :type df: pd.DataFrame
        :param x: x-value name for Pareto plot filtering
        :type x: str
        :param y: y-value name for Pareto plot filtering
        :type y: str
        :param factor_relative_y_min_offset: filter factor for the relative y minimum offset
        :type factor_relative_y_min_offset: float
        :param factor_relative_y_max_offset:
        :type factor_relative_y_max_offset: float
        :param absolute_max_y: Absolute maximum y-vales (e.g. maximum losses) (clip above this value)
        :type absolute_max_y: float
        :returns: pandas DataFrame with Pareto front near points
        :rtype: pd.DataFrame
        """
        # figure out pareto front
        # pareto_volume_list, pareto_core_hyst_list, pareto_dto_list = self.pareto_front(volume_list, core_hyst_loss_list, valid_design_list)

        pareto_df: pd.DataFrame = CircuitOptimizationBase.pareto_front_from_df_min_min(df, x, y)

        vector_to_sort = np.array([pareto_df[x], pareto_df[y]])

        # sorting 2d array by 1st row
        # https://stackoverflow.com/questions/49374253/sort-a-numpy-2d-array-by-1st-row-maintaining-columns
        sorted_vector = vector_to_sort[:, vector_to_sort[0].argsort()]
        x_pareto_vec = sorted_vector[0]
        y_pareto_vec = sorted_vector[1]

        total_losses_list = df[y][~np.isnan(df[y])].to_numpy()

        min_total_dc_losses = total_losses_list[np.argmin(total_losses_list)]
        loss_offset = factor_relative_y_min_offset * min_total_dc_losses

        ref_loss_max = np.interp(df[x], x_pareto_vec, y_pareto_vec) + loss_offset
        # clip losses to a maximum of the minimum losses
        ref_loss_max = np.clip(ref_loss_max, a_min=-1, a_max=factor_relative_y_max_offset * min_total_dc_losses)

        # clip point of the relative maximum losses given by the factor
        pareto_df_offset: pd.DataFrame = df[df[y] < ref_loss_max]

        # clip point to the absolute maximum losses
        pareto_df_offset = pareto_df_offset[pareto_df_offset[y] < absolute_max_y]

        return pareto_df_offset

    @staticmethod
    def filter_df_min_max(
            df: pd.DataFrame,
            x: str = "values_0",
            y: str = "values_1",
            relative_y_offset: float = 0.02,
            absolute_min_y: float | None = None,
    ) -> pd.DataFrame:
        """
        Remove designs with too low y-values (e.g. efficiency) compared to the maximum y-value (e.g. efficiency).

        :param df: pandas DataFrame with study results
        :type df: pd.DataFrame
        :param x: x-value name for Pareto plot filtering
        :type x: str
        :param y: y-value name for Pareto plot filtering
        :type y: str
        :param relative_y_offset: relative y offset (efficiency offset)
        :type relative_y_offset: float
        :param absolute_min_y: absolute minimum y value (e.g. minimum required efficiency)
        :type absolute_min_y: float
        :returns: pandas DataFrame with Pareto front near points
        :rtype: pd.DataFrame
        """
        pareto_df = CircuitOptimizationBase.pareto_front_from_df_min_max(df, x, y)
        pareto_df = pareto_df.sort_values(x)

        y_pareto_at_x = np.interp(
            df[x],
            pareto_df[x],
            pareto_df[y],
        )

        max_y = df[y].max()
        y_offset = relative_y_offset * max_y

        min_y_at_x = y_pareto_at_x - y_offset

        filtered_df: pd.DataFrame = df[df[y] >= min_y_at_x]

        if absolute_min_y is not None:
            filtered_df = filtered_df[filtered_df[y] >= absolute_min_y]

        return filtered_df

    @staticmethod
    def pareto_front_from_df_min_min(df: pd.DataFrame, x: str = "values_0", y: str = "values_1") -> pd.DataFrame:
        """
        Calculate the Pareto front from a Pandas DataFrame. Return a Pandas DataFrame.

        :param df: Pandas DataFrame
        :type df: pd.DataFrame
        :param x: Name of x-parameter from df to show in Pareto plane
        :type x: str
        :param y: Name of y-parameter from df to show in Pareto plane
        :type y: str
        :return: Pandas DataFrame with pareto efficient points
        :rtype: pd.DataFrame
        """
        x_vec = df[x][~np.isnan(df[x])]
        y_vec = df[y][~np.isnan(df[x])]
        numpy_zip = np.column_stack((x_vec, y_vec))
        pareto_tuple_mask_vec = CircuitOptimizationBase.is_pareto_efficient_min_min(numpy_zip)
        pareto_df: pd.DataFrame = df[~np.isnan(df[x])][pareto_tuple_mask_vec]
        return pareto_df

    @staticmethod
    def pareto_front_from_df_min_max(df: pd.DataFrame, x: str = "values_0", y: str = "values_1") -> pd.DataFrame:
        """
        Calculate the Pareto front from a Pandas DataFrame. Return a Pandas DataFrame.

        :param df: Pandas DataFrame
        :type df: pd.DataFrame
        :param x: Name of x-parameter from df to show in Pareto plane
        :type x: str
        :param y: Name of y-parameter from df to show in Pareto plane
        :type y: str
        :return: Pandas DataFrame with pareto efficient points
        :rtype: pd.DataFrame
        """
        x_vec = df[x][~np.isnan(df[x])]
        y_vec = df[y][~np.isnan(df[x])]
        numpy_zip = np.column_stack((x_vec, y_vec))

        # prepare y-values to use is_pareto_efficient_min_min as _min_max
        numpy_zip = numpy_zip.copy()
        numpy_zip[:, 1] *= -1

        pareto_tuple_mask_vec = CircuitOptimizationBase.is_pareto_efficient_min_min(numpy_zip)
        pareto_df: pd.DataFrame = df[~np.isnan(df[x])][pareto_tuple_mask_vec]
        return pareto_df

    @staticmethod
    def is_pareto_efficient_min_min(costs: np.ndarray, return_mask: bool = True) -> np.ndarray:
        """
        Find the pareto-efficient points.

        :param costs: An (n_points, n_costs) array
        :type costs: np.array
        :param return_mask: True to return a mask
        :type return_mask: bool
        :return: An array of indices of pareto-efficient points.
            If return_mask is True, this will be an (n_points, ) boolean array
            Otherwise it will be a (n_efficient_points, ) integer array of indices.
        :rtype: np.array
        """
        is_efficient = np.arange(costs.shape[0])
        n_points = costs.shape[0]
        next_point_index = 0  # Next index in the is_efficient array to search for
        while next_point_index < len(costs):
            non_dominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
            non_dominated_point_mask[next_point_index] = True  # type: ignore
            is_efficient = is_efficient[non_dominated_point_mask]  # Remove dominated points
            costs = costs[non_dominated_point_mask]
            next_point_index = int(np.sum(non_dominated_point_mask[:next_point_index])) + 1  # type: ignore
        if return_mask:
            is_efficient_mask = np.zeros(n_points, dtype=bool)
            is_efficient_mask[is_efficient] = True
            return is_efficient_mask
        else:
            return is_efficient

    # Abstract methods
    @staticmethod
    @abstractmethod
    def get_topology_type() -> TopologyEnum:
        """
        Provide the topology type.

        :return: Type of the topology
        :rtype:  TopologyEnum
        """
        pass

    @abstractmethod
    def load_and_verify_general_parameters(self, toml_dict: dict[str, Any]) -> tuple[bool, str]:
        """Verify the input parameter ranges.

        :param toml_dict: toml general configuration
        :type toml_dict: dict[str, Any]
        :return: True, if the configuration was consistent and empty string or False and report of the issues
        :rtype: tuple[bool, str]
        """
        pass

    @abstractmethod
    def load_and_verify_circuit_parameters(self, toml_dict: dict[str, Any], is_tdb_to_update: bool = False) -> tuple[bool, str]:
        """Load and verify the circuit input parameter.

        :param toml_dict: toml general configuration
        :type  toml_dict: dict[str, Any]
        :param is_tdb_to_update: indicated, if the transistor database is up to date
        :type  is_tdb_to_update: bool
        :return: True, if the configuration was consistent and empty string or False and report of the issues
        :rtype: tuple[bool, str]
        """
        pass

    @staticmethod
    @abstractmethod
    def _is_optimization_skippable(study_data: StudyData, filter_data: FilterData) -> tuple[bool, str]:
        """Verify the circuit optimization is skippable.

        :param study_data: Study data information
        :type  study_data: StudyData
        :param filter_data: Filtered data
        :type  filter_data: FilterData
        :return: True, if the configuration was consistent and empty string or False and report of the issues
        :rtype: tuple[bool, str]
        """
        pass

    @abstractmethod
    def initialize_circuit_optimization(self) -> bool:
        """Initialize the circuit optimization configuration.

        :return: True, if the configuration is initialized
        :rtype: bool
        """
        pass

    @abstractmethod
    def start_proceed_study(self, number_trials: int, database_type: str = 'sqlite',
                            sampler: optuna.samplers.BaseSampler = optuna.samplers.NSGAIIISampler()) -> None:
        """Proceed a study which is stored as sqlite database.

        :param number_trials: Number of trials adding to the existing study
        :type number_trials: int
        :param database_type: storage database, e.g. 'sqlite' or 'mysql'
        :type  database_type: str
        :param sampler: optuna.samplers.NSGAIISampler() or optuna.samplers.NSGAIIISampler(). Note about the brackets () !! Default: NSGAIII
        :type sampler: optuna.sampler-object
        """
        pass

    @abstractmethod
    def filter_study_results(self) -> tuple[bool, str]:
        """Filter the study result and (later to implement->) use ngspice for detailed calculation.

        :return: True, if the study results are filtered and empty string or False and report of the issues
        :rtype: tuple[bool, str]
        """
        pass

    # Server/ Performance methods

    @abstractmethod
    def get_progress_data(self) -> ProgressData:
        """Provide the progress data of the optimization.

        :return: Progress data: Processing start time, actual processing time, number of filtered operation points and status.
        :rtype: ProgressData
        """
        pass

    @abstractmethod
    def get_actual_pareto_html(self) -> str:
        """
        Read the current Pareto front from running optimization process.

        :return: Pareto front html page
        :rtype: str
        """
        pass

    @staticmethod
    @abstractmethod
    def get_pareto_html(study_name: str, path_name: str) -> str:
        """
        Read the current Pareto front from running optimization process.

        :param study_name: Name of the optuna study
        :type  study_name: str
        :param path_name: Path where optuna study is located
        :type  path_name: str

        :return: Pareto front html page
        :rtype: str
        """
        pass

    @staticmethod
    @abstractmethod
    def get_circuit_plot_data(act_study_data: StudyData) -> PlotData:
        """Provide the circuit data to plot.

        :param act_study_data: Information about the circuit study name and study path
        :type  act_study_data: StudyData
        :return: Plot data and legend
        :rtype: PlotData
        """
        pass

    @abstractmethod
    def get_capacitor_requirements(self) -> list[CapacitorRequirements]:
        """Get the capacitor requirements.

        :return: Capacitor requirements
        :rtype: CapacitorRequirements
        """
        pass

    @abstractmethod
    def get_inductor_requirements(self) -> list[InductorRequirements]:
        """Get the inductor requirements.

        :return: Inductor requirements
        :rtype: InductorRequirements
        """
        pass

    @abstractmethod
    def get_transformer_requirements(self) -> list[TransformerRequirements]:
        """Get the transformer requirements.

        :return: Transformer requirements
        :rtype: TransformerRequirements
        """
        pass

    @abstractmethod
    def get_output_power(self):
        """Get the sampling data output power.

        :return: Transformer requirements
        :rtype: TransformerRequirements
        """
        pass

    @abstractmethod
    def get_weights(self):
        """Get the sampling data weights.

        :return: Transformer requirements
        :rtype: TransformerRequirements
        """
        pass

    @staticmethod
    @abstractmethod
    def get_number_of_required_capacitors() -> int:
        """Get the number of  required capacitors.

        :return: Number of capacitors required by the actual topology
        :rtype: int
        """
        pass

    @staticmethod
    @abstractmethod
    def get_number_of_required_inductors() -> int:
        """Get the number of  required inductors.

        :return: Number of inductors required by the actual topology
        :rtype: int
        """
        pass

    @staticmethod
    @abstractmethod
    def get_number_of_required_transformers() -> int:
        """Get the number of  required transformers.

        :return: Number of transformers required by the actual topology
        :rtype: int
        """
        pass

    @abstractmethod
    def generate_result_dtos(self, summary_data: StudyData, capacitor_selection_data_list: list[CapacitorConfiguration],
                             inductor_configuration_list: list[InductorConfiguration],
                             transformer_configuration_list: list[TransformerConfiguration],
                             df: pd.DataFrame, is_pre_summary: bool = True) -> None:
        """
        Generate the result dtos from a given (filtered) result dataframe.

        :param summary_data: Summary Data
        :type  summary_data: StudyData
        :param capacitor_selection_data_list: List of capacitor selection data
        :type  capacitor_selection_data_list: list[CapacitorConfiguration]
        :param inductor_configuration_list: List of inductor study data
        :type  inductor_configuration_list: list[InductorConfiguration]
        :param transformer_configuration_list: List of transformer study data
        :type  transformer_configuration_list: list[TransformerConfiguration]
        :param df: dataframe to take the results from
        :type  df:  pd.DataFrame
        :param is_pre_summary: True for pre-summary, False for summary
        :type  is_pre_summary: bool
        :return: None
        :rtype:  None
        """
        pass

    @abstractmethod
    def add_time_domain_simulations(self) -> None:
        """Add time domain simulations to the existing circuit DTOs."""
        pass

    @staticmethod
    @abstractmethod
    def plot_compare_waveforms(dto_directory: str, output_filepath: str) -> None:
        """
        Compare calculated waveforms with simulated waveforms (GeckoCIRCUITS).

        :param dto_directory: Folder of circuit DTOs to read the values from
        :type dto_directory: str
        :param output_filepath: filepath of output plot
        :type output_filepath: str
        """
        pass

    @staticmethod
    @abstractmethod
    def generate_operating_point_table(circuit_id_filepath: str, results_path: str) -> None:
        """
        Generate operating point table for lab work.

        :param circuit_id_filepath: circuit id filepath
        :type circuit_id_filepath: str
        :param results_path: Path to store the result table
        :type results_path: str
        """
        pass

    @staticmethod
    @abstractmethod
    def plot_single_design_operating_points(combination_id_filepath: str, results_path: str, combination_id: str) -> None:
        """
        Generate plot outputs to show the operating points and compare the converters.

        :param combination_id_filepath: combination id filepath
        :type combination_id_filepath: str
        :param results_path: Path to store the result table
        :type results_path: str
        :param combination_id: combination ID
        :type combination_id: int
        """
        pass

    @staticmethod
    @abstractmethod
    def get_inductor_conf_dict(component_id: int = 0) -> dict:
        """
        Provide dictionary for inductor default configuration.

        :param component_id: component id
        :type component_id: int
        :return: Dictionary of inductor default configuration
        :rtype:  dict
        """
        pass

    @staticmethod
    @abstractmethod
    def get_template(template_type: TemplateType) -> tuple[bool, str]:
        """
        Provide the template source for default configuration of the topology.

        :param template_type: Type of the template (circuit or general)
        :type  template_type: TemplateType
        :return: validity indicator and template source data
        :rtype:  tuple[bool, str]
        """
        pass
