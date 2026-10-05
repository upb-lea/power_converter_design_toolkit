"""Pareto optimization classes and functions."""
# Python libraries
import os
import logging
import pickle
import datetime
import threading
import copy
from typing import Any

# 3rd party libraries
import optuna
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import deepdiff

# own libraries
import pcdt.sampling as sampling
from pcdt.components.component_dtos import InductorRequirements
from pcdt.generalplotsettings import colors
from pcdt.constant_path import GECKO_COMPONENT_MODELS_DIRECTORY
from pcdt.topology.dab import dab_datasets_dtos as d_dtos
from pcdt.topology.dab import dab_circuit_topology_dtos as circuit_dtos
from pcdt.components.summary_processing import SummaryProcessing
import transistordatabase as tdb
from pcdt.boundary_check import CheckCondition as c_flag
from pcdt.boundary_check import BoundaryCheck
from pcdt.topology.dab import dab_toml_checker as dab_tc
from pcdt.topology.dab.dab_datasets import HandleDabDto
from pcdt.datasets_dtos import (StudyData, FilterData, PlotData, CapacitorConfiguration,
                                InductorConfiguration, TransformerConfiguration)
from pcdt.server_ctl_dtos import ProgressData, ProgressStatus
from pcdt.server_ctl_dtos import RunTimeMeasurement as RunTime
from pcdt.circuit_enums import SamplingEnum
from pcdt.topology.circuit_optimization_base import CircuitOptimizationBase
import pcdt.generalplotsettings as gps
from pcdt.components.component_dtos import (CapacitorRequirements, ComponentRequirements,
                                            TransformerRequirements, ComponentCooling)
from pcdt.constant_path import (CIRCUIT_INDUCTOR_RELUCTANCE_LOSSES_FOLDER, CIRCUIT_TRANSFORMER_RELUCTANCE_LOSSES_FOLDER,
                                CIRCUIT_INDUCTOR_FEM_LOSSES_FOLDER, CIRCUIT_TRANSFORMER_FEM_LOSSES_FOLDER,
                                CIRCUIT_CAPACITOR_LOSS_FOLDER, SUMMARY_COMBINATION_FOLDER, FILTERED_RESULTS_PATH)
from pcdt.constants import FACTOR_SECONDS_TO_NANOSECONDS
from pcdt.topology.dab.dab_plot_waveforms import plot_calc_vs_requirements, plot_calc_waveforms, plot_calc_i_hf_waveforms

logger = logging.getLogger(__name__)

class DabCircuitOptimization(CircuitOptimizationBase[dab_tc.TomlDabGeneral, dab_tc.TomlDabCircuitParetoDesign]):
    """Optimize the DAB converter regarding maximum ZVS coverage and minimum conduction losses."""

    # Definition of constant values
    # Topological resource constants
    _number_of_required_capacitors: int = 2
    _number_of_required_inductors: int = 1
    _number_of_required_transformers: int = 1

    # Declaration of member types
    _c_lock_stat: threading.Lock
    _progress_data: ProgressData
    _progress_run_time: RunTime
    _dab_config: circuit_dtos.CircuitParetoDabDesign | None
    _study_in_memory: optuna.Study | None
    _study_in_storage: optuna.Study | None
    _fixed_parameters: d_dtos.FixedParameters | None

    # Areas and transistor cooling parameter
    copper_coin_area_1: float
    transistor_b1_cooling: ComponentCooling
    copper_coin_area_2: float
    transistor_b2_cooling: ComponentCooling

    misc: float

    def __init__(self) -> None:
        """Initialize the configuration list for the circuit optimizations."""
        # Call the constructor of the base class
        super().__init__()
        # Variable allocation
        self._c_lock_stat = threading.Lock()
        # Initialize the statistical data (For more configuration it needs to become instance instead of static
        self._progress_data = ProgressData(run_time=0, number_of_filtered_points=0, progress_status=ProgressStatus.Idle)
        self._progress_run_time = RunTime()
        self._dab_config = None
        self._is_study_available = False

        self._study_in_memory = None
        self._study_in_storage = None
        self._fixed_parameters = None

        # General optimization parameter
        self._toml_general: dab_tc.TomlDabGeneral | None = None
        # Circuit optimization parameter
        self._toml_circuit: dab_tc.TomlDabCircuitParetoDesign | None = None

        # Areas and transistor cooling parameter
        self.copper_coin_area_1 = 0
        self.copper_coin_area_2 = 0
        self.misc = 0

    def save_config(self) -> None:
        """Save the actual configuration file as pickle file on the disk."""
        # Check if a configuration is loaded
        if self._dab_config is None:
            logger.warning("Circuit configuration is empty!\n    Configuration is not saved!")
            return

        os.makedirs(self.circuit_study_data.optimization_directory, exist_ok=True)
        with open(f"{self.circuit_study_data.optimization_directory}/{self._dab_config.circuit_study_name}.pkl", 'wb') as output:
            pickle.dump(self._dab_config, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_stored_config(act_circuit_study_data: StudyData) -> circuit_dtos.CircuitParetoDabDesign:
        """
        Load pickle configuration file from disk.

        :param act_circuit_study_data: Information about the circuit study name and study path
        :type  act_circuit_study_data: StudyData
        :return: Configuration file as circuit_dtos.DabDesign
        :rtype: circuit_dtos.CircuitParetoDabDesign
        """
        config_pickle_filepath = os.path.join(act_circuit_study_data.optimization_directory,
                                              f"{act_circuit_study_data.study_name}.pkl")

        with open(config_pickle_filepath, 'rb') as pickle_file_data:
            loaded_pareto_dto = pickle.load(pickle_file_data)
            if not isinstance(loaded_pareto_dto, circuit_dtos.CircuitParetoDabDesign):
                raise TypeError(f"Loaded pickle file {loaded_pareto_dto} not of type CircuitParetoDabDesign.")

        return loaded_pareto_dto

    def load_and_verify_general_parameters(self, toml_dict: dict[str, Any]) -> tuple[bool, str]:
        """Verify the input parameter ranges.

        :param toml_dict: toml general configuration
        :type toml_dict: dict[str, Any]
        :return: True, if the configuration was consistent and empty string or False and report of the issues
        :rtype: tuple[bool, str]
        """
        # Variable declaration
        inconsistency_report: str = ""
        is_consistent: bool = True
        toml_check_min_max_value_multi_list: list[tuple[list[float], str, list[float], str]]

        # Convert and check toml file content
        toml_general: dab_tc.TomlDabGeneral = dab_tc.TomlDabGeneral(**toml_dict)

        # Check v1_min_max_list and v2_min_max_list
        toml_check_min_max_value_multi_list = (
            [(toml_general.output_range.v1_min_max_list, "v1_min_max_list",
              toml_general.sampling.v1_additional_user_point_list, "v1_additional_user_point_list"),
             (toml_general.output_range.v2_min_max_list, "v2_min_max_list",
              toml_general.sampling.v2_additional_user_point_list, "v2_additional_user_point_list")])

        # Output range parameter and sampling parameter check
        group_name = "output_range or sampling"
        # Init is_user_point_list_consistent-flag
        is_user_point_list_consistent = False
        # Evaluate list length
        len_additional_user_v1 = len(toml_general.sampling.v1_additional_user_point_list)
        len_additional_user_v2 = len(toml_general.sampling.v2_additional_user_point_list)
        len_additional_user_p = len(toml_general.sampling.p_additional_user_point_list)
        len_additional_user_w = len(toml_general.sampling.additional_user_weighting_point_list)
        len_check1 = len_additional_user_v1 == len_additional_user_v2 and len_additional_user_p == len_additional_user_w
        len_check2 = len_additional_user_p == len_additional_user_w
        # Check if the additional user point lists are consistent
        if len_check1 and len_check2:
            is_user_point_list_consistent = True

        # Perform the boundary check
        for check_parameter in toml_check_min_max_value_multi_list:
            is_check_passed, issue_report = BoundaryCheck.check_float_min_max_values(
                0, 1500, check_parameter[0], f"output_range: {check_parameter[1]}", c_flag.check_exclusive, c_flag.check_exclusive)
            if not is_check_passed:
                inconsistency_report = inconsistency_report + issue_report
                is_consistent = False
            elif is_user_point_list_consistent:
                for voltage_value in check_parameter[2]:
                    is_check_passed, issue_report = BoundaryCheck.check_float_value(
                        check_parameter[0][0], check_parameter[0][1], voltage_value,
                        f"sampling: {check_parameter[3]}", c_flag.check_inclusive, c_flag.check_inclusive)
                    if not is_check_passed:
                        inconsistency_report = inconsistency_report + issue_report
                        is_consistent = False
            else:
                act_report = f"    The number of list entries in v1_additional_user_point_list ({len_additional_user_v1}), "
                act_report = act_report + f"v2_additional_user_point_list ({len_additional_user_v2}),\n"
                act_report = act_report + f"    p_additional_user_point_list ({len_additional_user_p}) and "
                act_report = act_report + f"additional_user_weighting_point_list ({len_additional_user_w} "
                act_report = act_report + "needs to be the same!\n)"
                inconsistency_report = (inconsistency_report + act_report)
                is_consistent = False

        # Perform the boundary check  of p_min_max_list
        is_check_passed, issue_report = BoundaryCheck.check_float_min_max_values(
            -100000, 100000, toml_general.output_range.p_min_max_list, "output_range: p_min_max_list", c_flag.check_exclusive, c_flag.check_exclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False
        elif is_user_point_list_consistent:
            for power_value in toml_general.sampling.p_additional_user_point_list:
                is_check_passed, issue_report = BoundaryCheck.check_float_value(
                    toml_general.output_range.p_min_max_list[0], toml_general.output_range.p_min_max_list[1], power_value,
                    "sampling: p_additional_user_point_list", c_flag.check_inclusive, c_flag.check_inclusive)
                if not is_check_passed:
                    inconsistency_report = inconsistency_report + issue_report
                    is_consistent = False

        # Remaining Sampling parameter check
        group_name = "sampling"
        # Check additional_user_weighting_point_list
        # Initialize variable
        weighting_sum: float = 0.0
        # Perform the boundary check  of p_min_max_list
        for weight_value in toml_general.sampling.additional_user_weighting_point_list:
            is_check_passed, issue_report = BoundaryCheck.check_float_value(
                0, 1, weight_value, "additional_user_weighting_point_list", c_flag.check_inclusive, c_flag.check_inclusive)
            if not is_check_passed:
                inconsistency_report = inconsistency_report + issue_report
                is_consistent = False

            weighting_sum = weighting_sum + weight_value

        # Check the sum
        if weighting_sum > 1:
            is_consistent = False
            act_report = "    The sum of parameter entries of parameter additional_user_weighting_point_list "
            act_report = act_report + f"{weighting_sum} has to be less equal 1!\n"
            inconsistency_report = inconsistency_report + act_report

        # Perform the boundary check for sampling points
        is_check_passed, issue_report = BoundaryCheck.check_float_value(
            0, 1, float(toml_general.sampling.sampling_points),
            f"{group_name}: sampling_points", c_flag.check_exclusive, c_flag.check_ignore)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Check sampling random seed
        # Perform the boundary check for number_filtered_designs
        is_check_passed, issue_report = BoundaryCheck.check_float_value(
            0, 1, float(toml_general.sampling.sampling_random_seed),
            f"{group_name}: sampling_random_seed", c_flag.check_inclusive, c_flag.check_ignore)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False
            # delete old parameter
            self._toml_general = None
        else:
            # Overtake the parameter
            self._toml_general = toml_general

        return is_consistent, inconsistency_report

    def load_and_verify_circuit_parameters(self, toml_dict: dict[str, Any], is_tdb_to_update: bool = False) -> tuple[bool, str]:
        """Load and verify the circuit input parameter.

        :param toml_dict: dictionary with circuit configuration
        :type  toml_dict: dict[str, Any]
        :param is_tdb_to_update: True to update the transistor database
        :type is_tdb_to_update: bool
        :return: True, if the configuration was consistent and empty string or False and report of the issues
        :rtype: tuple[bool, str]
        """
        # Variable declaration
        inconsistency_report: str = ""
        is_consistent: bool = True
        toml_check_keyword_list: list[tuple[list[str], str]]
        toml_check_min_max_values_list: list[tuple[list[float], str]]
        toml_check_value_list: list[tuple[float, str]]

        # Convert and check toml file content
        toml_circuit: dab_tc.TomlDabCircuitParetoDesign = dab_tc.TomlDabCircuitParetoDesign(**toml_dict)

        # Design space parameter check
        # Create dictionary from transistor database list
        db = tdb.DatabaseManager()
        db.set_operation_mode_json()
        if is_tdb_to_update:
            logger.info("Update transistor data from TDB file exchange.")
            db.update_from_fileexchange(overwrite=True)
        else:
            logger.info(f"No transistor data update from TDB file exchange. ({is_tdb_to_update=})")

        # Get available keywords
        keyword_list: list[str] = db.get_transistor_names_list()
        keyword_dictionary: dict = {}
        # Create dictionary
        for keyword_entry in keyword_list:
            keyword_dictionary[keyword_entry] = 0

        # Check transistors
        toml_check_keyword_list = (
            [(toml_circuit.design_space.transistor_1_name_list, "transistor_1_name_list"),
             (toml_circuit.design_space.transistor_2_name_list, "transistor_2_name_list")])

        # Perform the boundary check
        for check_keyword in toml_check_keyword_list:
            if len(check_keyword[0]) == 0:
                inconsistency_report = f"   Circuit parameter: List {check_keyword[1]} is empty!\n"
                is_consistent = False
            else:
                # Perform dictionary check
                for keyword_entry in check_keyword[0]:
                    is_check_passed, issue_report = BoundaryCheck.check_dictionary(keyword_dictionary, keyword_entry, check_keyword[1])
                    # Check if boundary check fails
                    if not is_check_passed:
                        inconsistency_report = inconsistency_report + issue_report
                        is_consistent = False

        # Check switching frequency range
        float_f_s_min_max_list = BoundaryCheck.convert_int_list_to_float_list(toml_circuit.design_space.f_s_min_max_list)
        is_check_passed, issue_report = BoundaryCheck.check_float_min_max_values(
            1000, 1e7, float_f_s_min_max_list, "f_s_min_max_list", c_flag.check_exclusive, c_flag.check_exclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Check the maximum dead time for dead_time_1_max
        is_check_passed, issue_report = BoundaryCheck.check_float_min_max_values(
            1e-9, 10e-6, [toml_circuit.design_space.t_dead_1_max, toml_circuit.design_space.t_dead_1_max],
            "t_dead_1_max", c_flag.check_exclusive, c_flag.check_exclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Check the maximum dead time for dead_time_2_max
        is_check_passed, issue_report = BoundaryCheck.check_float_min_max_values(
            1e-9, 10e-6, [toml_circuit.design_space.t_dead_2_max, toml_circuit.design_space.t_dead_2_max],
            "t_dead_2_max", c_flag.check_exclusive, c_flag.check_exclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Check l_s_min_max_list, l_1_min_max_list and l_2__min_max_list
        toml_check_min_max_values_list = (
            [(toml_circuit.design_space.l_s_min_max_list, "l_s_min_max_list"),
             (toml_circuit.design_space.l_1_min_max_list, "l_1_min_max_list"),
             (toml_circuit.design_space.l_2__min_max_list, "l_2__min_max_list")])

        # Perform the boundary check
        is_check_passed, issue_report = BoundaryCheck.check_float_min_max_values_list(
            0, 1, toml_check_min_max_values_list, c_flag.check_exclusive, c_flag.check_exclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        is_check_passed, issue_report = BoundaryCheck.check_float_min_max_values(
            0, 100, toml_circuit.design_space.n_min_max_list, "n_min_max_list", c_flag.check_exclusive, c_flag.check_exclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Check l_s_min_max_list, l_1_min_max_list and l_2__min_max_list
        toml_check_min_max_values_list = (
            [(toml_circuit.design_space.l_s_min_max_list, "l_s_min_max_list"),
             (toml_circuit.design_space.l_1_min_max_list, "l_1_min_max_list"),
             (toml_circuit.design_space.l_2__min_max_list, "l_2__min_max_list")])

        # Perform the boundary check
        is_check_passed, issue_report = BoundaryCheck.check_float_min_max_values_list(
            0, 1, toml_check_min_max_values_list, c_flag.check_inclusive, c_flag.check_exclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Check c_par_1 and c_par_2
        toml_check_value_list = (
            [(toml_circuit.design_space.c_par_1, "c_par_1"),
             (toml_circuit.design_space.c_par_2, "c_par_2")])

        # Perform the boundary check
        # Check c_par_1 and c_par_2
        is_check_passed, issue_report = BoundaryCheck.check_float_value_list(
            0, 1e-3, toml_check_value_list, c_flag.check_exclusive, c_flag.check_exclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Perform filter_distance value check
        group_name = "filter_distance"
        # Perform the boundary check for number_filtered_designs
        is_check_passed, issue_report = BoundaryCheck.check_float_value(
            0, 100, float(toml_circuit.filter_distance.number_filtered_designs),
            f"{group_name}: number_filtered_designs", c_flag.check_exclusive, c_flag.check_ignore)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Perform the boundary check for number_filtered_designs
        is_check_passed, issue_report = BoundaryCheck.check_float_value(
            0.01, 100, toml_circuit.filter_distance.difference_percentage,
            f"{group_name}: difference_percentage", c_flag.check_exclusive, c_flag.check_inclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Perform thermal resistance data check
        group_name = "thermal_data"
        # Create the list
        toml_check_value_list1: list[tuple[float, str]] = []
        toml_check_value_list2: list[tuple[float, str]] = []

        # Perform list length check for transistor_b1_cooling
        if len(toml_circuit.thermal_data.transistor_b1_cooling) != 2:
            inconsistency_report = inconsistency_report + "    Number of values in parameter 'transistor_b1_cooling' is not equal 2!\n"
            is_consistent = False
        else:
            toml_check_value_list1.append(
                (toml_circuit.thermal_data.transistor_b1_cooling[0], f"{group_name}: transistor_b1_cooling[0]-tim_thickness"))
            toml_check_value_list2.append(
                (toml_circuit.thermal_data.transistor_b1_cooling[1], f"{group_name}: transistor_b1_cooling[1]-tim_conductivity"))

        # Perform list length check for transistor_b1_cooling
        if len(toml_circuit.thermal_data.transistor_b1_cooling) != 2:
            inconsistency_report = inconsistency_report + "    Number of values in parameter 'transistor_b2_cooling' is not equal 2!\n"
            is_consistent = False
        else:
            toml_check_value_list1.append(
                (toml_circuit.thermal_data.transistor_b1_cooling[0], f"{group_name}: transistor_b2_cooling[0]-tim_thickness"))
            toml_check_value_list2.append(
                (toml_circuit.thermal_data.transistor_b1_cooling[1], f"{group_name}: transistor_b2_cooling[1]-tim_conductivity"))

        # Perform the boundary check for tim-thickness
        is_check_passed, issue_report = BoundaryCheck.check_float_value_list(
            0, 0.01, toml_check_value_list1, c_flag.check_exclusive, c_flag.check_inclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Perform the boundary check for tim-conductivity
        is_check_passed, issue_report = BoundaryCheck.check_float_value_list(
            1, 100, toml_check_value_list2, c_flag.check_exclusive, c_flag.check_inclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False
            # delete old parameter
            self._toml_circuit = None
        else:
            # Overtake the parameter
            self._toml_circuit = toml_circuit

        return is_consistent, inconsistency_report

    @staticmethod
    def _is_optimization_skippable(study_data: StudyData, filter_data: FilterData) -> tuple[bool, str]:
        # Variable declaration
        # skippable flag
        is_skippable: bool = False
        # Report string
        issue_report: str = ""

        # Check, if data are available (skip case)
        # Check if filtered results folder exists
        if os.path.exists(filter_data.filtered_list_pathname):
            # Add filtered result list
            for filtered_circuit_result in os.listdir(filter_data.filtered_list_pathname):
                if os.path.isfile(os.path.join(filter_data.filtered_list_pathname, filtered_circuit_result)):
                    filter_data.filtered_list_files.append(os.path.splitext(filtered_circuit_result)[0])

            if not filter_data.filtered_list_files:
                issue_report = f"Filtered results folder {filter_data.filtered_list_pathname} is empty."
            else:
                is_skippable = True
        else:
            issue_report = f"Filtered circuit results folder {filter_data.filtered_list_pathname} does not exist."

        # Return evaluation result
        return is_skippable, issue_report

    def initialize_circuit_optimization(self) -> bool:
        """
        Initialize the circuit_dto for circuit optimization.

        :return: True, if the configuration was successful initialized
        :rtype: bool
        """
        # Check if toml_circuit, toml_general and study data are still initialized
        if self._toml_general is None or self._toml_circuit is None or self.circuit_study_data is None:
            # Serious programming error: In verification check these variables should be initialized
            # This has to be guaranteed by the workflow
            raise ValueError("Serious programming error 'multiple allocation failure'. Please write an issue!")

        # Initialize the circuit_dtos
        design_space = circuit_dtos.CircuitParetoDesignSpace(
            f_s_min_max_list=self._toml_circuit.design_space.f_s_min_max_list,
            l_s_min_max_list=self._toml_circuit.design_space.l_s_min_max_list,
            l_1_min_max_list=self._toml_circuit.design_space.l_1_min_max_list,
            l_2__min_max_list=self._toml_circuit.design_space.l_2__min_max_list,
            n_min_max_list=self._toml_circuit.design_space.n_min_max_list,
            transistor_1_name_list=self._toml_circuit.design_space.transistor_1_name_list,
            transistor_2_name_list=self._toml_circuit.design_space.transistor_2_name_list,
            c_par_1=self._toml_circuit.design_space.c_par_1,
            c_par_2=self._toml_circuit.design_space.c_par_2,
            t_dead_1_max=self._toml_circuit.design_space.t_dead_1_max,
            t_dead_2_max=self._toml_circuit.design_space.t_dead_2_max
        )

        output_range = circuit_dtos.CircuitOutputRange(
            v1_min_max_list=self._toml_general.output_range.v1_min_max_list,
            v2_min_max_list=self._toml_general.output_range.v2_min_max_list,
            p_min_max_list=self._toml_general.output_range.p_min_max_list)

        filter = circuit_dtos.CircuitFilter(
            number_filtered_designs=self._toml_circuit.filter_distance.number_filtered_designs,
            difference_percentage=self._toml_circuit.filter_distance.difference_percentage
        )

        # None can not be handled by toml correct, so this is a workaround. By default, "random" in toml equals "None"
        local_sampling_random_seed: int | None = None
        # In case of a concrete seed was given, overwrite None with the given one
        if isinstance(self._toml_general.sampling.sampling_random_seed, int):
            local_sampling_random_seed = int(self._toml_general.sampling.sampling_random_seed)

        sampling = circuit_dtos.CircuitSampling(
            sampling_method=self._toml_general.sampling.sampling_method,
            sampling_points=self._toml_general.sampling.sampling_points,
            sampling_random_seed=local_sampling_random_seed,
            v1_additional_user_point_list=self._toml_general.sampling.v1_additional_user_point_list,
            v2_additional_user_point_list=self._toml_general.sampling.v2_additional_user_point_list,
            p_additional_user_point_list=self._toml_general.sampling.p_additional_user_point_list,
            additional_user_weighting_point_list=self._toml_general.sampling.additional_user_weighting_point_list
        )

        self._dab_config = circuit_dtos.CircuitParetoDabDesign(
            circuit_study_name=self.circuit_study_data.study_name,
            project_directory=self.project_directory,
            design_space=design_space,
            output_range=output_range,
            sampling=sampling,
            filter=filter)

        return True

    def get_config(self) -> circuit_dtos.CircuitParetoDabDesign | None:
        """
        Return the actual loaded configuration file.

        :return: Configuration file as circuit_dtos.DabDesign
        :rtype: circuit_dtos.CircuitParetoDabDesign
        """
        if self._dab_config is None:
            logger.warning("Configuration is not loaded!")

        return copy.deepcopy(self._dab_config)

    def get_progress_data(self) -> ProgressData:
        """Provide the progress data of the optimization.

        :return: Progress data: Processing start time, actual processing time, number of filtered operation points and status.
        :rtype: ProgressData
        """
        # Lock statistical performance data access
        with self._c_lock_stat:
            # Check if list is in progress
            if self._progress_data.progress_status == ProgressStatus.InProgress:
                # Update statistical data if optimization is running
                self._progress_data.run_time = self._progress_run_time.get_runtime()

        return copy.deepcopy(self._progress_data)

    def get_actual_pareto_html(self) -> str:
        """
        Read the current Pareto front from running optimization process.

        :return: Pareto front html page
        :rtype: str
        """
        # Variable declaration
        pareto_html: str = ""

        if self._study_in_memory is not None:
            fig = optuna.visualization.plot_pareto_front(self._study_in_memory)
            pareto_html = fig.to_html(full_html=False)

        return pareto_html

    @staticmethod
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
        # Variable declaration
        pareto_html = ""

        # Load study from file
        try:
            study = optuna.load_study(study_name=study_name, storage="sqlite:///" + path_name)
            fig = optuna.visualization.plot_pareto_front(study)
            pareto_html = fig.to_html(full_html=False)
        except KeyError:
            logger.warning(f"Study with name '{study_name}' are not found.")

        return pareto_html

    @staticmethod
    def _objective(trial: optuna.Trial, dab_config: circuit_dtos.CircuitParetoDabDesign, fixed_parameters: d_dtos.FixedParameters) -> tuple:
        """
        Objective function to optimize.

        :param dab_config: DAB optimization configuration file
        :type dab_config: circuit_dtos.CircuitParetoDabDesign
        :param trial: optuna trial
        :type trial: optuna.Trial
        :param fixed_parameters: fixed parameters (loaded transistor DTOs)
        :type fixed_parameters: d_dtos.FixedParameters

        :return:
        """
        f_s_suggest = trial.suggest_int('f_s_suggest', dab_config.design_space.f_s_min_max_list[0], dab_config.design_space.f_s_min_max_list[1])
        l_s_suggest = trial.suggest_float('l_s_suggest', dab_config.design_space.l_s_min_max_list[0], dab_config.design_space.l_s_min_max_list[1])
        l_1_suggest = trial.suggest_float('l_1_suggest', dab_config.design_space.l_1_min_max_list[0], dab_config.design_space.l_1_min_max_list[1])
        l_2__suggest = trial.suggest_float('l_2__suggest', dab_config.design_space.l_2__min_max_list[0], dab_config.design_space.l_2__min_max_list[1])
        n_suggest = trial.suggest_float('n_suggest', dab_config.design_space.n_min_max_list[0], dab_config.design_space.n_min_max_list[1])
        transistor_1_name_suggest = trial.suggest_categorical('transistor_1_name_suggest', dab_config.design_space.transistor_1_name_list)
        transistor_2_name_suggest = trial.suggest_categorical('transistor_2_name_suggest', dab_config.design_space.transistor_2_name_list)

        for _, transistor_dto in enumerate(fixed_parameters.transistor_1_dto_list):
            if transistor_dto.name == transistor_1_name_suggest:
                transistor_1_dto: d_dtos.TransistorDTO = transistor_dto

        for _, transistor_dto in enumerate(fixed_parameters.transistor_2_dto_list):
            if transistor_dto.name == transistor_2_name_suggest:
                transistor_2_dto: d_dtos.TransistorDTO = transistor_dto

        dab_calc = HandleDabDto.init_config(
            circuit_id=dab_config.circuit_study_name,
            mesh_v1=fixed_parameters.mesh_v1,
            mesh_v2=fixed_parameters.mesh_v2,
            mesh_p=fixed_parameters.mesh_p,
            mesh_weights=fixed_parameters.mesh_weights,
            sampling=dab_config.sampling,
            n=n_suggest,
            ls=l_s_suggest,
            fs=f_s_suggest,
            lc1=l_1_suggest,
            lc2=l_2__suggest / n_suggest ** 2,
            lossfilepath=fixed_parameters.transistorlosses_filepath,
            c_par_1=dab_config.design_space.c_par_1,
            c_par_2=dab_config.design_space.c_par_2,
            transistor_dto_1=transistor_1_dto,
            transistor_dto_2=transistor_2_dto,
            t_dead_1_max=dab_config.design_space.t_dead_1_max,
            t_dead_2_max=dab_config.design_space.t_dead_2_max
        )

        if (np.any(np.isnan(dab_calc.calc_modulation.phi)) or np.any(np.isnan(dab_calc.calc_modulation.tau1)) \
                or np.any(np.isnan(dab_calc.calc_modulation.tau2))):
            return float('nan'), float('nan')

        if dab_calc.calc_dead_time is None:
            raise ValueError("Incomplete calculation, as dead time is missing.")

        # check if the minimum dead time in all operating points is less equal than the maximum allowed dead time
        dead_time_1_less_maximum = np.less_equal(dab_calc.calc_dead_time.t_dead_1, dab_calc.input_config.t_dead_1_max)
        dead_time_2_less_maximum = np.less_equal(dab_calc.calc_dead_time.t_dead_2, dab_calc.input_config.t_dead_2_max)

        if not np.all(dead_time_1_less_maximum):
            logger.info(f"Needed dead time of bridge 1 exceeds maximum dead time of {dab_calc.input_config.t_dead_1_max}.")
            return float('nan'), float('nan')
        if not np.all(dead_time_2_less_maximum):
            logger.info(f"Needed dead time of bridge 2 exceeds maximum dead time of {dab_calc.input_config.t_dead_2_max}.")
            return float('nan'), float('nan')

        # Calculate the cost function.
        i_cost_matrix = dab_calc.calc_currents.i_hf_1_rms ** 2 + dab_calc.calc_currents.i_hf_2_rms ** 2
        # consider weighting
        i_cost_matrix_weighted = i_cost_matrix * fixed_parameters.mesh_weights

        # Sum for not-NaN values, as there will be too many NaN results.
        i_cost = np.sum(i_cost_matrix_weighted[~np.isnan(i_cost_matrix_weighted)])

        trial.set_user_attr('dead_time_zvs_coverage', dab_calc.calc_dead_time.zvs_coverage * 100)
        trial.set_user_attr('zvs_coverage', dab_calc.calc_modulation.zvs_coverage * 100)

        return dab_calc.calc_dead_time.zvs_coverage * 100, i_cost

    @staticmethod
    def calculate_fixed_parameters(act_dab_config: circuit_dtos.CircuitParetoDabDesign) -> d_dtos.FixedParameters:
        """
        Calculate time-consuming parameters which are same for every single simulation.

        :param act_dab_config: DAB circuit configuration
        :type act_dab_config: circuit_dtos.CircuitParetoDabDesign
        :return: Fix parameters (transistor DTOs)
        :rtype: d_dtos.FixedParameters
        """
        transistor_1_dto_list = []
        transistor_2_dto_list = []

        for transistor in act_dab_config.design_space.transistor_1_name_list:
            transistor_1_dto_list.append(HandleDabDto.tdb_to_transistor_dto(transistor))

        for transistor in act_dab_config.design_space.transistor_2_name_list:
            transistor_2_dto_list.append(HandleDabDto.tdb_to_transistor_dto(transistor))

        # choose sampling method
        if act_dab_config.sampling.sampling_method == SamplingEnum.meshgrid:
            steps_per_dimension = int(np.ceil(np.power(act_dab_config.sampling.sampling_points, 1 / 3)))
            logger.info(f"Number of sampling points has been updated from {act_dab_config.sampling.sampling_points} to {steps_per_dimension ** 3}.")
            logger.info("Note: meshgrid sampling does not take user-given operating points into account")
            if act_dab_config.output_range.v1_min_max_list[0] == act_dab_config.output_range.v1_min_max_list[1]:
                v1_dimension = np.array([act_dab_config.output_range.v1_min_max_list[0]])
                logger.info(f"Reducing v1 dimension to {act_dab_config.output_range.v1_min_max_list[0]} V.")
            else:
                v1_dimension = np.linspace(act_dab_config.output_range.v1_min_max_list[0], act_dab_config.output_range.v1_min_max_list[1], steps_per_dimension)
            if act_dab_config.output_range.v2_min_max_list[0] == act_dab_config.output_range.v2_min_max_list[1]:
                v2_dimension = np.array(act_dab_config.output_range.v2_min_max_list[0])
                logger.info(f"Reducing v2 dimension to {act_dab_config.output_range.v2_min_max_list[0]} V.")
            else:
                v2_dimension = np.linspace(act_dab_config.output_range.v2_min_max_list[0], act_dab_config.output_range.v2_min_max_list[1], steps_per_dimension)
            if act_dab_config.output_range.p_min_max_list[0] == act_dab_config.output_range.p_min_max_list[1]:
                p_dimension = np.array(act_dab_config.output_range.p_min_max_list[0])
                logger.info(f"Reducing p dimension to {act_dab_config.output_range.p_min_max_list[0]} W.")
            else:
                p_dimension = np.linspace(act_dab_config.output_range.p_min_max_list[0], act_dab_config.output_range.p_min_max_list[1], steps_per_dimension)
            v1_operating_points, v2_operating_points, p_operating_points = np.meshgrid(v1_dimension, v2_dimension, p_dimension, sparse=False)
        elif act_dab_config.sampling.sampling_method == SamplingEnum.latin_hypercube:
            v1_operating_points, v2_operating_points, p_operating_points = sampling.latin_hypercube(
                act_dab_config.output_range.v1_min_max_list[0], act_dab_config.output_range.v1_min_max_list[1],
                act_dab_config.output_range.v2_min_max_list[0], act_dab_config.output_range.v2_min_max_list[1],
                act_dab_config.output_range.p_min_max_list[0], act_dab_config.output_range.p_min_max_list[1],
                total_number_points=act_dab_config.sampling.sampling_points,
                dim_1_user_given_points_list=act_dab_config.sampling.v1_additional_user_point_list,
                dim_2_user_given_points_list=act_dab_config.sampling.v2_additional_user_point_list,
                dim_3_user_given_points_list=act_dab_config.sampling.p_additional_user_point_list,
                sampling_random_seed=act_dab_config.sampling.sampling_random_seed)
        elif act_dab_config.sampling.sampling_method == SamplingEnum.dessca:
            v1_operating_points, v2_operating_points, p_operating_points = sampling.dessca(
                act_dab_config.output_range.v1_min_max_list[0], act_dab_config.output_range.v1_min_max_list[1],
                act_dab_config.output_range.v2_min_max_list[0], act_dab_config.output_range.v2_min_max_list[1],
                act_dab_config.output_range.p_min_max_list[0], act_dab_config.output_range.p_min_max_list[1],
                total_number_points=act_dab_config.sampling.sampling_points,
                dim_1_user_given_points_list=act_dab_config.sampling.v1_additional_user_point_list,
                dim_2_user_given_points_list=act_dab_config.sampling.v2_additional_user_point_list,
                dim_3_user_given_points_list=act_dab_config.sampling.p_additional_user_point_list,
                sampling_random_seed=act_dab_config.sampling.sampling_random_seed)

        else:
            raise ValueError(f"sampling_method '{act_dab_config.sampling.sampling_method}' not available.")

        logger.debug(f"{v1_operating_points=}")

        # calculate weighting
        if act_dab_config.sampling.sampling_method == SamplingEnum.meshgrid:
            weight_sum = 0
            given_user_points = 0
        else:
            weight_sum = np.sum(act_dab_config.sampling.additional_user_weighting_point_list)
            logger.debug(f"{weight_sum=}")
            given_user_points = len(act_dab_config.sampling.v1_additional_user_point_list)
        logger.debug(f"{given_user_points=}")
        logger.debug(f"{v1_operating_points.size=}")

        if weight_sum > 1 or weight_sum < 0:
            raise ValueError("Sum of weighting point list must be within 0 and 1.")
        else:
            leftover_auto_weight = (1 - weight_sum) / (v1_operating_points.size - given_user_points)
            logger.info(f"Auto-weight given for all other {v1_operating_points.size - given_user_points} operating points: {leftover_auto_weight}")
            # default case, same weights for all points
            weights = np.full_like(v1_operating_points, leftover_auto_weight)
            # for user point weightings, both lists must be filled.
            if act_dab_config.sampling.additional_user_weighting_point_list and act_dab_config.sampling.sampling_method != SamplingEnum.meshgrid:
                logger.debug("Given user weighting point list detected, fill up with user-given weights.")
                weights[-len(act_dab_config.sampling.additional_user_weighting_point_list):] = act_dab_config.sampling.additional_user_weighting_point_list
            logger.debug(f"{weights=}")
            logger.debug(f"Double check: Sum of weights = {np.sum(weights)}")

        return d_dtos.FixedParameters(
            transistor_1_dto_list=transistor_1_dto_list,
            transistor_2_dto_list=transistor_2_dto_list,
            transistorlosses_filepath="Invalid_Path",
            mesh_v1=np.atleast_3d(v1_operating_points),
            mesh_v2=np.atleast_3d(v2_operating_points),
            mesh_p=np.atleast_3d(p_operating_points),
            mesh_weights=np.atleast_3d(weights)
        )

    def run_optimization_sqlite(self, act_number_trials: int) -> None:
        """Proceed a study which is stored as sqlite database.

        :type act_number_trials: int
        :param act_number_trials: Number of optimization trials
        """
        if self._dab_config is None:
            logger.warning("Circuit configuration is not initialized!")
            return
        elif self._fixed_parameters is None:
            logger.warning("Parameter calculation is missing!")
            return
        elif self._study_in_memory is None:
            logger.warning("Study is not initialized!")
            return

        # Function to execute
        func = lambda trial: DabCircuitOptimization._objective(trial, self._dab_config, self._fixed_parameters)

        try:
            self._study_in_memory.optimize(func, n_trials=act_number_trials, n_jobs=1, show_progress_bar=True)
        except KeyboardInterrupt:
            pass

    def run_optimization_mysql(self, act_storage_url: str, act_number_trials: int) -> None:
        """Proceed a study which is stored as sqlite database.

        :param act_storage_url: url-Name of the database path
        :type act_storage_url: str
        :param act_number_trials: Number of trials adding to the existing study
        :type  act_number_trials: int
        """
        if self._dab_config is None:
            logger.warning("Circuit configuration is not initialized!")
            return
        elif self._fixed_parameters is None:
            logger.warning("Parameter calculation is missing!")
            return

        # Function to execute
        func = lambda trial: DabCircuitOptimization._objective(trial, self._dab_config, self._fixed_parameters)

        # Each process create his own study instance with the same database and study name
        act_study = optuna.load_study(storage=act_storage_url, study_name=self._dab_config.circuit_study_name)
        # Run optimization
        try:
            act_study.optimize(func, n_trials=act_number_trials, n_jobs=1, show_progress_bar=True)
        except KeyboardInterrupt:
            pass
        finally:
            # study_in_storage.add_trials(act_study_name.trials[-number_trials:])
            logger.info(f"Finished {act_number_trials} trials.")
            logger.info(f"current time: {datetime.datetime.now()}")
            # Save method from RAM-Disk to where ever (Currently opened by missing RAM-DISK)

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
        if self._dab_config is None:
            logger.warning("Method 'initialize_circuit_optimization' is not called!\n"
                           "    No list is generated so that no optimization can be performed!")
            return

        circuit_study_sqlite_database = os.path.join(self.circuit_study_data.optimization_directory,
                                                     f"{self._dab_config.circuit_study_name}.sqlite3")

        # Assemble the name for c_oss_storage_directory
        new_c_oss_directory: str = os.path.join(self.circuit_study_data.optimization_directory, GECKO_COMPONENT_MODELS_DIRECTORY)
        # Create c_oss_storage_directory, it not exists
        if not os.path.exists(new_c_oss_directory):
            os.makedirs(new_c_oss_directory)

        # Set the directory path
        HandleDabDto.set_c_oss_storage_directory(new_c_oss_directory)

        if os.path.exists(circuit_study_sqlite_database):
            logger.info("Existing circuit study found. Proceeding.")
        else:
            os.makedirs(f"{self.circuit_study_data.optimization_directory}", exist_ok=True)

        # set logging verbosity: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.logger.set_verbosity.html#optuna.logging.set_verbosity
        # .INFO: all messages (default)
        # .WARNING: fails and warnings
        # .ERROR: only errors
        optuna.logging.set_verbosity(optuna.logging.ERROR)

        # check for differences with the old configuration file
        config_on_disk_filepath = f"{self.circuit_study_data.optimization_directory}/{self._dab_config.circuit_study_name}.pkl"
        if os.path.exists(config_on_disk_filepath):
            config_on_disk = DabCircuitOptimization.load_stored_config(self.circuit_study_data)
            difference = deepdiff.DeepDiff(config_on_disk, self._dab_config, ignore_order=True, significant_digits=10)
            if difference:
                raise ValueError("Configuration file has changed from previous simulation.\n"
                                 f"Program is terminated.\n"
                                 f"Difference: {difference}")

        directions = ['maximize', 'minimize']

        # Calculate the fixed parameters
        self._fixed_parameters = DabCircuitOptimization.calculate_fixed_parameters(self._dab_config)
        # Add path to losses
        self._fixed_parameters.transistorlosses_filepath = os.path.join(self.circuit_study_data.optimization_directory, GECKO_COMPONENT_MODELS_DIRECTORY)

        # Update statistical data
        with self._c_lock_stat:
            self._progress_data.run_time = self._progress_run_time.get_runtime()
            self._progress_run_time.reset_start_trigger()
            self._progress_data.progress_status = ProgressStatus.InProgress

        # introduce study in storage, e.g. sqlite or mysql
        if database_type == 'sqlite':
            # Note: for sqlite operation, there needs to be three slashes '///' even before the path '/home/...'
            # Means, in total there are four slashes including the path itself '////home/.../database.sqlite3'
            storage = f"sqlite:///{circuit_study_sqlite_database}"

            # Create study object in drive
            self._study_in_storage = optuna.create_study(study_name=self._dab_config.circuit_study_name,
                                                         storage=storage,
                                                         directions=directions,
                                                         load_if_exists=True, sampler=sampler)

            # Create study object in memory
            self._study_in_memory = optuna.create_study(study_name=self._dab_config.circuit_study_name, directions=directions, sampler=sampler)
            # If trials exists, add them to study_in_memory
            self._study_in_memory.add_trials(self._study_in_storage.trials)
            # Inform about sampler type
            logger.info(f"Sampler is {self._study_in_storage.sampler.__class__.__name__}")
            # actual number of trials
            overtaken_no_trials = len(self._study_in_memory.trials)
            # Start optimization
            self.run_optimization_sqlite(number_trials)
            # Store memory to storage
            self._study_in_storage.add_trials(self._study_in_memory.trials[-number_trials:])
            logger.info(f"Add {number_trials} new calculated trials to existing {overtaken_no_trials} trials ="
                        f"{len(self._study_in_memory.trials)} trials.")
            logger.info(f"current time: {datetime.datetime.now()}")
            self.save_config()

            self.save_study_results_pareto(show_results=False)

        elif database_type == 'mysql':

            # connection to MySQL-database
            storage_url = "mysql+pymysql://oaml_optuna:optuna@localhost/optuna_db"

            # Create storage-object for Optuna on drive (Later RAMDISK)
            storage_mysql = optuna.storages.RDBStorage(storage_url)
            # storage = "mysql://oaml_optuna:optuna@localhost/optuna_db"

            # Create study object in drive
            self._study_in_storage = optuna.create_study(study_name=self._dab_config.circuit_study_name,
                                                         storage=storage_mysql,
                                                         directions=directions,
                                                         load_if_exists=True, sampler=sampler)

            # Inform about sampler type
            logger.info(f"Sampler is {self._study_in_storage.sampler.__class__.__name__}")
            # Start optimization
            self.run_optimization_mysql(storage_url, number_trials)

            # Stop time measurement
            self._progress_run_time.stop_trigger()

        # Set flag _is_study_available to indicate, that the study is available for filtering
        self._is_study_available = True

        # Parallelization Test with mysql
        # Number of processes
        #   num_processes = 1
        # Process list
        #    processes = []
        # Loop to start the processes
        #   for proc in range(num_processes):
        #       logger.info(f"Process {proc} started")
        #       p = multiprocessing.Process(target=Optimization.run_optimization,
        #                                   args=(storage_url, dab_config.circuit_study_name,
        #                                         number_trials, dab_config,fixed_parameters
        #                                        )
        #                                  )
        #       p.start()
        #       processes.append(p)

        # Wait for joining
        #   for proc in processes:
        # wait until each process is joined
        #       p.join()
        #       logger.info(f"Process {proc} joins")

    def save_study_results_pareto(self, show_results: bool = False) -> None:
        """Show the results of a study.

        A local .html file is generated under config.working_directory to store the interactive plotly plots on disk.

        :param show_results: True to directly open the browser to show the study results.
        :type show_results: bool
        """
        if self._study_in_storage is None:
            logger.warning("The study is not initialized!")
            return
        elif self._dab_config is None:
            logger.warning("Circuit configuration is not initialized!")
            return

        fig = optuna.visualization.plot_pareto_front(self._study_in_storage, target_names=["ZVS coverage / %", r"i_\mathrm{cost}"])
        fig.update_layout(title=f"{self._dab_config.circuit_study_name} <br><sup>{self._dab_config.project_directory}</sup>")
        fig.write_html(
            f"{self.circuit_study_data.optimization_directory}"
            f"_{datetime.datetime.now().isoformat(timespec='minutes')}.html")
        if show_results:
            fig.show()

    @staticmethod
    def load_dab_dto_from_study(act_study_data: StudyData, dab_config: circuit_dtos.CircuitParetoDabDesign,
                                trial_number: int | None = None) -> d_dtos.DabCircuitDTO:
        """
        Load a DAB-DTO from an optuna study.

        :param act_study_data: Information about the circuit study name and study path
        :type  act_study_data: StudyData
        :param dab_config: DAB optimization configuration file
        :type dab_config: circuit_dtos.CircuitParetoDabDesign
        :param trial_number: trial number to load to the DTO
        :type trial_number: int
        :return: DTO
        :rtype:  d_dtos.DabCircuitDTO
        """
        if trial_number is None:
            raise NotImplementedError("needs to be implemented")

        database_url = DabCircuitOptimization.create_sqlite_database_url(act_study_data)

        loaded_study = optuna.create_study(study_name=act_study_data.study_name,
                                           storage=database_url, load_if_exists=True)
        logger.info(f"The study '{act_study_data.study_name}' contains {len(loaded_study.trials)} trials.")
        trials_dict = loaded_study.trials[trial_number].params

        fix_parameters = DabCircuitOptimization.calculate_fixed_parameters(dab_config)

        dab_dto = HandleDabDto.init_config(
            circuit_id=str(trial_number),
            mesh_v1=fix_parameters.mesh_v1,
            mesh_v2=fix_parameters.mesh_v2,
            mesh_p=fix_parameters.mesh_p,
            mesh_weights=fix_parameters.mesh_weights,
            sampling=dab_config.sampling,
            n=trials_dict["n_suggest"],
            ls=trials_dict["l_s_suggest"],
            fs=trials_dict["f_s_suggest"],
            lc1=trials_dict["l_1_suggest"],
            lc2=trials_dict["l_2__suggest"] / trials_dict["n_suggest"] ** 2,
            c_par_1=dab_config.design_space.c_par_1,
            c_par_2=dab_config.design_space.c_par_2,
            transistor_dto_1=trials_dict["transistor_1_name_suggest"],
            transistor_dto_2=trials_dict["transistor_2_name_suggest"],
            lossfilepath=fix_parameters.transistorlosses_filepath,
            t_dead_1_max=dab_config.design_space.t_dead_1_max,
            t_dead_2_max=dab_config.design_space.t_dead_2_max,
        )
        return dab_dto

    def df_to_dab_dto_list(self, df: pd.DataFrame) -> list[d_dtos.DabCircuitDTO]:
        """
        Load a DAB-DTO from an optuna study.

        :param df: Pandas DataFrame to convert to the DAB-DTO list
        :type df: pd.DataFrame
        :return: List of DTO
        :rtype:  list[d_dtos.DabCircuitDTO]
        """
        dab_dto_list: list[d_dtos.DabCircuitDTO] = []

        # Check if configuration is not available or fixed parameters are not available
        if self._dab_config is None:
            logger.warning("Circuit configuration is not initialized!")
            return dab_dto_list
        elif self._fixed_parameters is None:
            logger.warning("Missing initialized fixed parameters!")
            return dab_dto_list

        logger.info(f"The study '{self._dab_config.circuit_study_name}' contains {len(df)} trials.")

        for idx, _ in df.iterrows():
            index = int(str(idx))
            transistor_dto_1 = HandleDabDto.tdb_to_transistor_dto(str(df.at[index, "params_transistor_1_name_suggest"]))
            transistor_dto_2 = HandleDabDto.tdb_to_transistor_dto(str(df.at[index, "params_transistor_2_name_suggest"]))

            dab_dto = HandleDabDto.init_config(
                circuit_id=str(df["number"][index].item()),
                mesh_v1=self._fixed_parameters.mesh_v1,
                mesh_v2=self._fixed_parameters.mesh_v2,
                mesh_p=self._fixed_parameters.mesh_p,
                mesh_weights=self._fixed_parameters.mesh_weights,
                sampling=self._dab_config.sampling,
                n=df["params_n_suggest"][index].item(),
                ls=df["params_l_s_suggest"][index].item(),
                fs=df["params_f_s_suggest"][index].item(),
                lc1=df["params_l_1_suggest"][index].item(),
                lc2=df["params_l_2__suggest"][index].item() / df["params_n_suggest"][index].item() ** 2,
                c_par_1=self._dab_config.design_space.c_par_1,
                c_par_2=self._dab_config.design_space.c_par_2,
                transistor_dto_1=transistor_dto_1,
                transistor_dto_2=transistor_dto_2,
                lossfilepath=self._fixed_parameters.transistorlosses_filepath,
                t_dead_1_max=self._dab_config.design_space.t_dead_1_max,
                t_dead_2_max=self._dab_config.design_space.t_dead_2_max,
            )

            dab_dto_list.append(dab_dto)

        return dab_dto_list

    @staticmethod
    def study_to_df(act_study_data: StudyData) -> pd.DataFrame:
        """Create a DataFrame from a study.

        :param act_study_data: Information about the circuit study name and study path
        :type  act_study_data: StudyData
        :return: study result data transferred to Pandas dataframe
        :rtype: pd.DataFrame
        """
        database_url = DabCircuitOptimization.create_sqlite_database_url(act_study_data)
        loaded_study = optuna.create_study(study_name=act_study_data.study_name, storage=database_url, load_if_exists=True)
        df = loaded_study.trials_dataframe()
        df.to_csv(f'{act_study_data.optimization_directory}/{act_study_data.study_name}.csv')
        return df

    @staticmethod
    def create_sqlite_database_url(act_study_data: StudyData) -> str:
        """
        Create the DAB circuit optimization sqlite URL.

        :param act_study_data: Information about the circuit study name and study path
        :type  act_study_data: StudyData
        :return: SQLite URL
        :rtype: str
        """
        sqlite_storage_url = f"sqlite:///{act_study_data.optimization_directory}/{act_study_data.study_name}.sqlite3"
        return sqlite_storage_url

    @staticmethod
    def df_plot_pareto_front(df: pd.DataFrame, figure_size: tuple) -> None:
        """Plot an interactive Pareto diagram (losses vs. volume) to select the transformers to re-simulate.

        :param df: DataFrame, generated from an optuna study (exported by optuna)
        :type df: pd.DataFrame
        :param figure_size: figure size as x,y-tuple in mm, e.g. (160, 80)
        :type figure_size: tuple
        """
        logger.info(df.head())

        names = df["number"].to_numpy()
        # plt.figure()
        fig, ax = plt.subplots(figsize=[x / 25.4 for x in figure_size] if figure_size is not None else None, dpi=80)
        sc = plt.scatter(df["values_0"], df["values_1"], s=10)

        annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)

        def update_annot(ind):
            pos = sc.get_offsets()[ind["ind"][0]]
            annot.xy = pos
            text = f"{[names[n] for n in ind['ind']]}"
            annot.set_text(text)
            annot.get_bbox_patch().set_alpha(0.4)

        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = sc.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", hover)

        plt.xlabel(r'ZVS coverage in \%')
        plt.ylabel(r'$i_\mathrm{HF,1}^2 + i_\mathrm{HF,2}^2$ in A')
        plt.grid()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def load_csv_to_df(csv_filepath: str) -> pd.DataFrame:
        """
        Load a csv file (previously stored from a Pandas DataFrame) back to a Pandas DataFrame.

        :param csv_filepath: File path of .csv file
        :type csv_filepath: str
        :return: loaded results from the given .csv file
        :rtype: pandas.DataFrame
        """
        df = pd.read_csv(csv_filepath, header=0, index_col=0)
        # reading a pandas DataFrame seems to change a global variable in the c subsystem
        # after reading csv values, there are issues running onelab/gmsh, as gmsh writes ',' instead '.' to its own files
        # reading the file again with setting back the delimiter to ';', is a workaround for the mentioned problem.
        pd.read_csv(csv_filepath, header=0, index_col=0, delimiter=';')
        return df

    def filter_study_results(self) -> tuple[bool, str]:
        """Filter the study result and use GeckoCIRCUITS for detailed calculation."""
        # Variable initialization
        is_filter_available: bool = False
        issue_report: str = ""

        # Check if configuration is not available and if study is not available
        if self._dab_config is None:
            issue_report = "Circuit configuration is not loaded!"
            return is_filter_available, issue_report
        if not self._is_study_available:
            issue_report = "Study is not calculated. First run 'start_proceed_study'!"
            return is_filter_available, issue_report
        if self._study_in_storage is None:
            issue_report = "Study is not calculated. First run 'start_proceed_study'!"
            return is_filter_available, issue_report
        if self._toml_circuit is None:
            raise ValueError("Serious programming error in 'filter study results'. Please write an issue!")

        is_filter_available = True

        # Evaluate previous result
        if not is_filter_available:
            logger.warning(issue_report)
            return is_filter_available, issue_report

        df = self._study_in_storage.trials_dataframe()
        df.to_csv(f'{self.circuit_study_data.optimization_directory}/{self._dab_config.circuit_study_name}.csv')

        # get 100 percent ZVS coverage designs
        df = df[df["values_0"] == 100]

        smallest_dto_list: list[d_dtos.DabCircuitDTO] = []
        df_smallest_all = df.nsmallest(n=1, columns=["values_1"])
        df_smallest = df.nsmallest(n=1, columns=["values_1"])

        smallest_dto_list.append(self.df_to_dab_dto_list(df_smallest)[0])

        # Continue time measurement
        with self._c_lock_stat:
            self._progress_run_time.continue_trigger()

        for count in np.arange(0, self._dab_config.filter.number_filtered_designs - 1):
            logger.info("------------------")
            logger.info(f"{count=}")
            n_suggest = df_smallest['params_n_suggest'].item()
            f_s_suggest = df_smallest['params_f_s_suggest'].item()
            l_s_suggest = df_smallest['params_l_s_suggest'].item()
            l_1_suggest = df_smallest['params_l_1_suggest'].item()
            l_2__suggest = df_smallest['params_l_2__suggest'].item()
            transistor_1_name_suggest = df_smallest['params_transistor_1_name_suggest'].item()
            transistor_2_name_suggest = df_smallest['params_transistor_2_name_suggest'].item()

            # make sure to use parameters with minimum x % difference.
            difference = self._dab_config.filter.difference_percentage / 100

            df = df.loc[
                ~((df["params_n_suggest"].ge(n_suggest * (1 - difference)) & df["params_n_suggest"].le(n_suggest * (1 + difference))) & \
                  (df["params_f_s_suggest"].ge(f_s_suggest * (1 - difference)) & df["params_f_s_suggest"].le(f_s_suggest * (1 + difference))) & \
                  (df["params_l_s_suggest"].ge(l_s_suggest * (1 - difference)) & df["params_l_s_suggest"].le(l_s_suggest * (1 + difference))) & \
                  (df["params_l_1_suggest"].ge(l_1_suggest * (1 - difference)) & df["params_l_1_suggest"].le(l_1_suggest * (1 + difference))) & \
                  (df["params_l_2__suggest"].ge(l_2__suggest * (1 - difference)) & df["params_l_2__suggest"].le(l_2__suggest * (1 + difference)))
                  )]

            df_smallest = df.nsmallest(n=1, columns=["values_1"])
            df_smallest_all = pd.concat([df_smallest_all, df_smallest], axis=0)

        smallest_dto_list = self.df_to_dab_dto_list(df_smallest_all)
        logger.info("Save filtered circuit results into csv file.")
        df_smallest_all.to_csv(f'{self.circuit_study_data.optimization_directory}/{self._dab_config.circuit_study_name}_filtered.csv')

        dto_directory = self.filter_data.filtered_list_pathname
        os.makedirs(dto_directory, exist_ok=True)
        for dto in smallest_dto_list:
            # generate the target requirements for inductor, transformer and capacitor
            dto = HandleDabDto.generate_components_target_requirements(dto)
            # Get thermal data
            transistor_b1_cooling: ComponentCooling = ComponentCooling(
                tim_thickness=self._toml_circuit.thermal_data.transistor_b1_cooling[0],
                tim_conductivity=self._toml_circuit.thermal_data.transistor_b1_cooling[1])
            transistor_b2_cooling: ComponentCooling = ComponentCooling(
                tim_thickness=self._toml_circuit.thermal_data.transistor_b2_cooling[0],
                tim_conductivity=self._toml_circuit.thermal_data.transistor_b2_cooling[1])
            # generate the thermal parameters for the given design
            dto = HandleDabDto.add_transistor_turn_off_currents_losses(dto)
            dto = HandleDabDto.generate_thermal_transistor_parameters(dto, transistor_b1_cooling, transistor_b2_cooling)

            HandleDabDto.save(dto, dto.circuit_id, directory=dto_directory, timestamp=False)

        # Update the filtered result list
        self.filter_data.filtered_list_files = []
        for filtered_circuit_result in os.listdir(dto_directory):
            # Check if it is a file
            if os.path.isfile(os.path.join(dto_directory, filtered_circuit_result)):
                self.filter_data.filtered_list_files.append(os.path.splitext(filtered_circuit_result)[0])

        # Evaluate the result: Has the list minimum one entry
        if len(self.filter_data.filtered_list_files) == 0:
            is_filter_available = False
            issue_report = "No design could be selected!"

        # Stop runtime measurement and update statistical data
        with self._c_lock_stat:
            self._progress_run_time.stop_trigger()
            self._progress_data.run_time = self._progress_run_time.get_runtime()
            self._progress_data.progress_status = ProgressStatus.Done

        return is_filter_available, issue_report

    def get_capacitor_requirements(self) -> list[CapacitorRequirements]:
        """Get the capacitor requirements.

        :return: Capacitor Requirements
        :rtype: CapacitorRequirements
        """
        capacitor_requirements_list: list[CapacitorRequirements] = []
        for circuit_id_file in self.filter_data.filtered_list_files:
            circuit_id_filepath = os.path.join(self.filter_data.filtered_list_pathname, f"{circuit_id_file}.pkl")
            circuit_dto = HandleDabDto.load_from_file(circuit_id_filepath)
            if not isinstance(circuit_dto.component_requirements, ComponentRequirements):
                # due to mypy checker
                raise TypeError("Loaded component requirements have wrong type.")

            if not isinstance(circuit_dto.component_requirements.capacitor_requirements[0], CapacitorRequirements):
                # due to mypy checker
                raise TypeError("Loaded capacitor requirements have wrong type.")

            capacitor_requirements_list = capacitor_requirements_list + circuit_dto.component_requirements.capacitor_requirements

        return capacitor_requirements_list

    def get_inductor_requirements(self) -> list[InductorRequirements]:
        """Get the inductor requirements.

        :return: Inductor Requirements
        :rtype: InductorRequirements
        """
        inductor_requirements_list: list[InductorRequirements] = []
        for circuit_id_file in self.filter_data.filtered_list_files:
            circuit_id_filepath = os.path.join(self.filter_data.filtered_list_pathname, f"{circuit_id_file}.pkl")
            circuit_dto = HandleDabDto.load_from_file(circuit_id_filepath)
            if not isinstance(circuit_dto.component_requirements, ComponentRequirements):
                # due to mypy checker
                raise TypeError("Loaded component requirements have wrong type.")

            if not isinstance(circuit_dto.component_requirements.inductor_requirements[0], InductorRequirements):
                # due to mypy checker
                raise TypeError("Loaded capacitor requirements have wrong type.")

            inductor_requirements_list = inductor_requirements_list + circuit_dto.component_requirements.inductor_requirements

        return inductor_requirements_list

    def get_transformer_requirements(self) -> list[TransformerRequirements]:
        """Get the transformer requirements.

        :return: Transformer Requirements
        :rtype: TransformerRequirements
        """
        transformer_requirements_list: list[TransformerRequirements] = []
        for circuit_id_file in self.filter_data.filtered_list_files:
            circuit_id_filepath = os.path.join(self.filter_data.filtered_list_pathname, f"{circuit_id_file}.pkl")
            circuit_dto = HandleDabDto.load_from_file(circuit_id_filepath)
            if not isinstance(circuit_dto.component_requirements, ComponentRequirements):
                # due to mypy checker
                raise TypeError("Loaded component requirements have wrong type.")

            if not isinstance(circuit_dto.component_requirements.transformer_requirements[0], TransformerRequirements):
                # due to mypy checker
                raise TypeError("Loaded capacitor requirements have wrong type.")
            transformer_requirements_list = transformer_requirements_list + circuit_dto.component_requirements.transformer_requirements

        return transformer_requirements_list

    def get_output_power(self) -> np.ndarray:
        """Get operating point output power."""
        dto_directory = os.path.join(self.circuit_study_data.optimization_directory, FILTERED_RESULTS_PATH)
        pkl_file_list = [f for f in os.listdir(dto_directory) if os.path.isfile(os.path.join(dto_directory, f))]

        circuit_id_filepath = os.path.join(dto_directory, pkl_file_list[0])

        with open(circuit_id_filepath, 'rb') as pickle_file_data:
            combination_dto: d_dtos.DabCircuitDTO = pickle.load(pickle_file_data)

        if combination_dto.calc_dead_time is None:
            raise TypeError("Incomplete DTO, as dead time is missing.")

        return np.array(combination_dto.input_config.mesh_p).flatten()

    def get_weights(self) -> np.ndarray:
        """Get operating point weights."""
        dto_directory = os.path.join(self.circuit_study_data.optimization_directory, FILTERED_RESULTS_PATH)
        pkl_file_list = [f for f in os.listdir(dto_directory) if os.path.isfile(os.path.join(dto_directory, f))]

        circuit_id_filepath = os.path.join(dto_directory, pkl_file_list[0])

        with open(circuit_id_filepath, 'rb') as pickle_file_data:
            combination_dto: d_dtos.DabCircuitDTO = pickle.load(pickle_file_data)

        if combination_dto.calc_dead_time is None:
            raise TypeError("Incomplete DTO, as dead time is missing.")

        return np.array(combination_dto.input_config.mesh_weights).flatten()

    @staticmethod
    def get_circuit_plot_data(act_study_data: StudyData) -> PlotData:
        """Provide the circuit data to plot.

        :param act_study_data: Information about the circuit study name and study path
        :type  act_study_data: StudyData
        :return: Plot data and legend
        :rtype: PlotData
        """
        # Load circuit study data
        df_circuit = DabCircuitOptimization.study_to_df(act_study_data)

        # Convert from list[Series[Any]] to list[list[float]]
        circuit_x_values_list: list[list[float]] = [df_circuit["values_0"].to_list()]
        circuit_y_values_list: list[list[float]] = [df_circuit["values_1"].to_list()]

        # Extract circuit plot data from data frame
        circuit_plot_data: PlotData = (
            PlotData(x_values_list=circuit_x_values_list, y_values_list=circuit_y_values_list,
                     color_list=[gps.colors()["black"]], alpha_list=[0.5],
                     x_label=r"$\mathcal{L}_\mathrm{v}$ / \%", y_label=r"$\mathcal{L}_\mathrm{i}$ / A²",
                     label_list=[None], fig_name_path=act_study_data.study_name))

        return circuit_plot_data

    @staticmethod
    def get_number_of_required_capacitors() -> int:
        """Get the number of  required capacitors.

        :return: Number of capacitors required by the actual topology
        :rtype: int
        """
        return DabCircuitOptimization._number_of_required_capacitors

    @staticmethod
    def get_number_of_required_inductors() -> int:
        """Get the number of  required inductors.

        :return: Number of inductors required by the actual topology
        :rtype: int
        """
        return DabCircuitOptimization._number_of_required_inductors

    @staticmethod
    def get_number_of_required_transformers() -> int:
        """Get the number of  required transformers.

        :return: Number of transformers required by the actual topology
        :rtype: int
        """
        return DabCircuitOptimization._number_of_required_transformers

    @staticmethod
    def generate_general_toml(file_path: str) -> None:
        """Generate the default DabCircuitConf.toml file.

        :param file_path: filename including absolute path
        :type file_path: str
        """
        toml_data = '''
        [default_data] # After update this configuration file according your project delete this line to validate it
        [output_range]
            v1_min_max_list=[690, 710]
            v2_min_max_list=[175, 295]
            p_min_max_list=[-2000, 2200]


        [sampling]
            sampling_method="dessca"
            sampling_points=2
            sampling_random_seed=10
            v1_additional_user_point_list=[]
            v2_additional_user_point_list=[]
            p_additional_user_point_list=[]
            additional_user_weighting_point_list=[]
        '''
        with open(file_path, 'w') as output:
            output.write(toml_data)

    @staticmethod
    def generate_circuit_toml(file_path: str) -> None:
        """
        Generate the default DabCircuitConf.toml file.

        :param file_path: filename including absolute path
        :type file_path: str
        """
        toml_data = '''
        [default_data] # After update this configuration file according your project delete this line to validate it
        [design_space]
            f_s_min_max_list=[50e3, 300e3]
            l_s_min_max_list=[20e-6, 900e-6]
            l_1_min_max_list=[10e-6, 10e-3]
            l_2__min_max_list=[10e-6, 1e-3]
            n_min_max_list=[3, 7]
            transistor_1_name_list=['CREE_C3M0065100J', 'CREE_C3M0120100J']
            transistor_2_name_list=['CREE_C3M0060065J', 'CREE_C3M0120065J']
            c_par_1=16e-12
            c_par_2=16e-12
            t_dead_1_max = 500e-9
            t_dead_2_max = 500e-9

        [output_range]
            v1_min_max_list=[690, 710]
            v2_min_max_list=[175, 295]
            p_min_max_list=[-2000, 2200]

        [sampling]
            sampling_method="latin_hypercube"
            sampling_points=4
            sampling_random_seed=10
            v1_additional_user_point_list=[700]
            v2_additional_user_point_list=[180]
            p_additional_user_point_list=[2000]

        [filter_distance]
            number_filtered_designs = 1
            difference_percentage = 5

        [thermal_data]
            # [tim_thickness, tim_conductivity]
            transistor_b1_cooling = [1e-3,12.0]
            transistor_b2_cooling = [1e-3,12.0]            

        '''
        with open(file_path, 'w') as output:
            output.write(toml_data)

    _transformer_study_configuration_list: list[TransformerConfiguration]
    _inductor_study_configuration_list: list[InductorConfiguration]
    _capacitor_selection_configuration_list: list[CapacitorConfiguration]

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
        :type  df: pd.DataFrame
        :param is_pre_summary: True for pre-summary, False for summary
        :type  is_pre_summary: bool
        """
        if is_pre_summary:
            # pre summary using reluctance model results (inductive components)
            inductor_result_directory = CIRCUIT_INDUCTOR_RELUCTANCE_LOSSES_FOLDER
            transformer_result_directory = CIRCUIT_TRANSFORMER_RELUCTANCE_LOSSES_FOLDER
        else:
            # final summary using FEM results (inductive components)
            inductor_result_directory = CIRCUIT_INDUCTOR_FEM_LOSSES_FOLDER
            transformer_result_directory = CIRCUIT_TRANSFORMER_FEM_LOSSES_FOLDER

        # note: the index represents the combination_id
        for combination_id, row in df.iterrows():
            circuit_id = row['circuit_id']
            inductor_id = row['inductor_id_0']
            inductor_study_name = row['inductor_study_name_0']
            transformer_id = row['transformer_id_0']
            transformer_study_name = row['transformer_study_name_0']
            capacitor_1_id = row['capacitor_id_0']
            capacitor_1_study_name = row['capacitor_study_name_0']
            capacitor_2_id = row['capacitor_id_1']
            capacitor_2_study_name = row['capacitor_study_name_1']

            # load circuit DTO
            circuit_id_filepath = os.path.join(self.filter_data.filtered_list_pathname, f"{circuit_id}.pkl")
            with open(circuit_id_filepath, 'rb') as pickle_file_data:
                circuit_dto: d_dtos.DabCircuitDTO = pickle.load(pickle_file_data)

            # load inductor DTO
            study_data = inductor_configuration_list[0].study_data
            inductor_study_results_filepath = os.path.join(study_data.optimization_directory, circuit_id,
                                                           inductor_study_name,
                                                           inductor_result_directory)

            inductor_id_filepath = os.path.join(inductor_study_results_filepath, f"{inductor_id}.pkl")
            with open(inductor_id_filepath, 'rb') as pickle_file_data:
                inductor_results = pickle.load(pickle_file_data)

            # load transformer DTO
            study_data = transformer_configuration_list[0].study_data
            transformer_study_results_filepath = os.path.join(study_data.optimization_directory, circuit_id,
                                                              transformer_study_name, transformer_result_directory)

            transformer_id_filepath = os.path.join(transformer_study_results_filepath, f"{transformer_id}.pkl")
            with open(transformer_id_filepath, 'rb') as pickle_file_data:
                transformer_results = pickle.load(pickle_file_data)

            # load capacitor 1 DTO
            study_data = capacitor_selection_data_list[0].study_data
            capacitor_1_study_results_filepath = os.path.join(study_data.optimization_directory, circuit_id,
                                                              capacitor_1_study_name, CIRCUIT_CAPACITOR_LOSS_FOLDER)

            capacitor_1_id_filepath = os.path.join(capacitor_1_study_results_filepath, f"{capacitor_1_id}.pkl")
            with open(capacitor_1_id_filepath, 'rb') as pickle_file_data:
                capacitor_1_results = pickle.load(pickle_file_data)

            # load capacitor 1 DTO
            study_data = capacitor_selection_data_list[1].study_data
            capacitor_2_study_results_filepath = os.path.join(study_data.optimization_directory, circuit_id,
                                                              capacitor_2_study_name, CIRCUIT_CAPACITOR_LOSS_FOLDER)

            capacitor_2_id_filepath = os.path.join(capacitor_2_study_results_filepath, f"{capacitor_2_id}.pkl")
            with open(capacitor_2_id_filepath, 'rb') as pickle_file_data:
                capacitor_2_results = pickle.load(pickle_file_data)

            circuit_dto.inductor_results = inductor_results
            circuit_dto.stacked_transformer_results = transformer_results
            circuit_dto.capacitor_1_results = capacitor_1_results
            circuit_dto.capacitor_2_results = capacitor_2_results

            results_folder = os.path.join(summary_data.optimization_directory, SUMMARY_COMBINATION_FOLDER)
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)

            HandleDabDto.save(circuit_dto, str(combination_id), directory=results_folder, timestamp=False)

    @staticmethod
    def generate_operating_point_table(circuit_id_filepath: str, results_path: str) -> None:
        """
        Generate operating point table for lab work.

        :param circuit_id_filepath: circuit id filepath
        :type circuit_id_filepath: str
        :param results_path: Path to store the result table
        :type results_path: str
        """
        # Get circuit results
        with open(circuit_id_filepath, 'rb') as pickle_file_data:
            combination_dto: d_dtos.DabCircuitDTO = pickle.load(pickle_file_data)

        if combination_dto.calc_dead_time is None:
            raise TypeError("Incomplete DTO, as dead time is missing.")

        data = {
            "v1": np.array(combination_dto.input_config.mesh_v1).flatten(),
            "v2": np.array(combination_dto.input_config.mesh_v2).flatten(),
            "p": np.array(combination_dto.input_config.mesh_p).flatten(),
            "phi / deg": np.array(np.rad2deg(combination_dto.calc_modulation.phi)).flatten(),
            "tau_1 / deg": np.array(np.rad2deg(combination_dto.calc_modulation.tau1)).flatten(),
            "tau_2 / deg": np.array(np.rad2deg(combination_dto.calc_modulation.tau2)).flatten(),
            "t_dead_1 / ns": np.array(combination_dto.calc_dead_time.t_dead_1).flatten() * FACTOR_SECONDS_TO_NANOSECONDS,
            "t_dead_2 / ns": np.array(combination_dto.calc_dead_time.t_dead_2).flatten() * FACTOR_SECONDS_TO_NANOSECONDS}

        df = pd.DataFrame(data)

        df.to_csv(f"{results_path}/control_parameters.csv")

        parameters_microcontroller = (
            "/*\n"
            "Code generated file by the optimization code.\n"
            "For a new converter design, replace this file by the new code generated file.\n"
            f"Timestamp calculation: {combination_dto.timestamp}\n"
            f"Timestamp of this c-file generation: {np.asarray(datetime.datetime.now().isoformat())}\n"
            f"Circuit ID: {combination_dto.circuit_id}\n"
            f"Switching frequency: {combination_dto.input_config.fs}\n"
            f"Ls: {combination_dto.input_config.Ls}\n"
            f"Lc1: {combination_dto.input_config.Lc1}\n"
            f"Lc2: {combination_dto.input_config.Lc2}\n"
            f"n: {combination_dto.input_config.n}\n"
            f"transistor_1: {combination_dto.input_config.transistor_dto_1.name}\n"
            f"transistor_2: {combination_dto.input_config.transistor_dto_2.name}\n"
            "*/\n\n"
            '#include "driverlib.h"\n'
            '#include "stddef.h"\n\n'
            "// Define a struct to store the values for each entry\n"
            "typedef struct {\n"
            "    uint8_t number;\n"
            "    uint8_t target_voltage_1;\n"
            "    uint8_t target_voltage_2;\n"
            "    uint8_t target_power;\n"
            "    double phi_deg;\n"
            "    double tau1_deg;\n"
            "    double tau2_deg;\n"
            "    double t_dead_1_ns;\n"
            "    double t_dead_2_ns;\n"
            "} DataEntry;\n\n"
            "DataEntry data[] =\n"
            "{\n")

        for count, row in df.iterrows():
            line = (
                "    {" + (f'{count}, {row["v1"]}, {row["v2"]}, {row["p"]}, {row["phi / deg"]}, {row["tau_1 / deg"]}, {row["tau_2 / deg"]}, '
                           f'{int(row["t_dead_1 / ns"])}, {int(row["t_dead_2 / ns"])}') + "},\n")
            parameters_microcontroller = parameters_microcontroller + line
        parameters_microcontroller += (
            "};\n\n"
            "size_t data_size(void){\n"
            "    return sizeof(data) / sizeof(data[0]);\n"
            "};\n\n"
            f"uint32_t frequency = {int(combination_dto.input_config.fs)};"
        )

        with open(f"{results_path}/control_parameters.c", "w") as f:
            f.write(parameters_microcontroller)

    @staticmethod
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
        with open(combination_id_filepath, 'rb') as pickle_file_data:
            combination_dto: d_dtos.DabCircuitDTO = pickle.load(pickle_file_data)

        if not os.path.exists(results_path):
            os.makedirs(results_path)

        if combination_dto.calc_losses is None:
            raise ValueError("Incomplete dataset.")
        if combination_dto.capacitor_1_results is None:
            raise ValueError("Incomplete dataset.")
        if combination_dto.capacitor_2_results is None:
            raise ValueError("Incomplete dataset.")
        if combination_dto.inductor_results is None:
            raise ValueError("Incomplete dataset.")
        if combination_dto.stacked_transformer_results is None:
            raise ValueError("Incomplete dataset.")

        data = {
            "circuit b1": combination_dto.calc_losses.p_m1_conduction.flatten(),
            "circuit b2": combination_dto.calc_losses.p_m2_conduction.flatten(),
            "capacitor b1": combination_dto.capacitor_1_results.loss_total_array.flatten(),
            "capacitor b2": combination_dto.capacitor_2_results.loss_total_array.flatten(),
            "inductor winding": combination_dto.inductor_results.winding_loss_array.flatten(),
            "inductor core": combination_dto.inductor_results.core_loss_array.flatten(),
            "transformer winding 1": combination_dto.stacked_transformer_results.winding_1_loss_array.flatten(),
            "transformer winding 2": combination_dto.stacked_transformer_results.winding_2_loss_array.flatten(),
            "transformer core": combination_dto.stacked_transformer_results.core_loss_array.flatten()
        }

        bar_colors = {
            "circuit b1": colors()["blue"],
            "circuit b2": colors()["blue"],
            "capacitor b1": colors()["orange"],
            "capacitor b2": colors()["orange"],
            "inductor winding": colors()["red"],
            "inductor core": colors()["red"],
            "transformer winding 1": colors()["gray"],
            "transformer winding 2": colors()["gray"],
            "transformer core": colors()["gray"]
        }

        textures = {
            "circuit b1": "",
            "circuit b2": ".",
            "capacitor b1": "",
            "capacitor b2": ".",
            "inductor winding": "",
            "inductor core": ".",
            "transformer winding 1": "",
            "transformer winding 2": "o",
            "transformer core": "."
        }

        # set up operating point x-labels
        x_labels = []
        for count, _ in enumerate(combination_dto.input_config.mesh_v1.flatten()):
            v1 = int(combination_dto.input_config.mesh_v1.flatten()[count])
            v2 = int(combination_dto.input_config.mesh_v2.flatten()[count])
            power = int(combination_dto.input_config.mesh_p.flatten()[count])
            operating_point_str = f"{v1} V,\n{v2} V,\n{power} W"
            x_labels.append(operating_point_str)

        number_operating_points = len(np.array(combination_dto.calc_modulation.phi).flatten())
        operating_point_list = np.linspace(1, number_operating_points, number_operating_points).tolist()

        # generate bar graph
        fig, ax = plt.subplots()
        bottom = np.zeros(number_operating_points)
        for label, data_count in data.items():
            ax.bar(operating_point_list, data_count, bottom=bottom, label=label, color=bar_colors[label], hatch=textures[label], edgecolor="black")
            bottom += data_count
        ax.set_xticks(operating_point_list, labels=x_labels)
        ax.set_xlabel("Operating points")
        ax.set_ylabel("Loss / W")
        ax.grid(axis="y")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{results_path}/operating_points_losses.pdf")
        plt.close()

    def add_time_domain_simulations(self) -> None:
        """Add time domain simulations to the existing circuit DTOs."""
        dto_source_directory = os.path.join(self.circuit_study_data.optimization_directory, FILTERED_RESULTS_PATH)
        dto_target_directory = os.path.join(self.circuit_study_data.optimization_directory, FILTERED_RESULTS_PATH)

        _, circuit_id_list = SummaryProcessing.generate_component_id_list_from_pkl_files(dto_source_directory)

        if not os.path.exists(dto_target_directory):
            os.makedirs(dto_target_directory)

        for circuit_id in circuit_id_list:
            # Assemble pkl-filename
            circuit_source_id_filepath = os.path.join(dto_source_directory, f"{circuit_id}.pkl")
            circuit_target_id_filepath = os.path.join(dto_target_directory, f"{circuit_id}.pkl")

            # Get circuit results
            with open(circuit_source_id_filepath, 'rb') as pickle_file_data:
                combination_dto: d_dtos.DabCircuitDTO = pickle.load(pickle_file_data)

            combination_dto = HandleDabDto.add_gecko_simulation_results(combination_dto, get_waveforms=True)

            with open(circuit_target_id_filepath, 'wb') as handle:
                pickle.dump(combination_dto, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def plot_compare_waveforms(circuit_id_filepath: str, output_filepath: str) -> None:
        """
        Compare calculated waveforms with simulated waveforms (GeckoCIRCUITS).

        :param circuit_id_filepath: Folder of circuit DTOs to read the values from
        :type circuit_id_filepath: str
        :param output_filepath: filepath of output plot
        :type output_filepath: str
        """
        # Get circuit results
        with open(circuit_id_filepath, 'rb') as pickle_file_data:
            combination_dto: d_dtos.DabCircuitDTO = pickle.load(pickle_file_data)

        plot_calc_waveforms(combination_dto, output_filepath, compare_gecko_waveforms=False)
        plot_calc_i_hf_waveforms(combination_dto, output_filepath, compare_gecko_waveforms=False)
        plot_calc_vs_requirements(combination_dto, output_filepath)
