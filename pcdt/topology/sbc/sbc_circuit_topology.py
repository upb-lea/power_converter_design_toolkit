"""Pareto optimization classes and functions."""
# Python libraries
import os
import logging
import pickle
import datetime
import threading
import copy
import sqlite3
from typing import Any

# 3rd party libraries
import optuna
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import pcdt.sampling as sampling
from sklearn.cluster import KMeans
from scipy.signal import savgol_filter
from typing import cast, SupportsFloat

# own libraries
from pcdt.topology.sbc import sbc_datasets_dtos as s_dtos
from pcdt.topology.sbc import sbc_circuit_topology_dtos as circuit_dtos
from pcdt.topology.sbc import sbc_datasets as d_sets
import transistordatabase as tdb
from pcdt.boundary_check import CheckCondition as c_flag
from pcdt.boundary_check import BoundaryCheck
from pcdt.topology.sbc import sbc_toml_checker as sbc_tc
from pcdt.datasets_dtos import (FilterData, StudyData, PlotData, CapacitorConfiguration,
                                InductorConfiguration, TransformerConfiguration)
from pcdt.server_ctl_dtos import ProgressData
from pcdt.server_ctl_dtos import ProgressStatus
from pcdt.server_ctl_dtos import RunTimeMeasurement as RunTime
from pcdt.circuit_enums import SamplingEnum, TemplateType, TopologyEnum
from pcdt.topology.circuit_optimization_base import CircuitOptimizationBase
import pcdt.generalplotsettings as gps
from pcdt.components.component_dtos import (CapacitorRequirements, InductorRequirements, TransformerRequirements,
                                            ComponentRequirements, ComponentCooling)
from pcdt.constant_path import (CIRCUIT_INDUCTOR_RELUCTANCE_LOSSES_FOLDER, CIRCUIT_TRANSFORMER_RELUCTANCE_LOSSES_FOLDER,
                                CIRCUIT_INDUCTOR_FEM_LOSSES_FOLDER, CIRCUIT_TRANSFORMER_FEM_LOSSES_FOLDER,
                                CIRCUIT_CAPACITOR_LOSS_FOLDER, SUMMARY_COMBINATION_FOLDER, PCDT_ROOT,
                                TOMPL_TEMPLATE_FOLDER_NAME, SBC_TOPOLOGY_FOLDER, GENERAL_CONF_TPL, CIRCUIT_CONF_TPL)

from pcdt.topology.sbc.sbc_constants import MAX_DUTY_CYCLE

logger = logging.getLogger(__name__)

class SbcCircuitOptimization(CircuitOptimizationBase[sbc_tc.TomlSbcGeneral, sbc_tc.TomlSbcCircuitParetoDesign]):
    """Optimize the SBC converter regarding maximum ZVS coverage and minimum conduction losses."""

    # Definition of constant values
    # Topological resource constants
    _number_of_required_capacitors: int = 2
    _number_of_required_inductors: int = 1
    _number_of_required_transformers: int = 0

    # Declaration of member types
    _c_lock_stat: threading.Lock
    _progress_data: ProgressData
    _progress_run_time: RunTime
    _sbc_config: circuit_dtos.CircuitParetoSbcDesign | None
    _study_in_memory: optuna.Study | None
    _study_in_storage: pd.DataFrame | None
    _fixed_parameters: s_dtos.FixedParameters | None

    def __init__(self) -> None:
        """Initialize the configuration list for the circuit optimizations."""
        # Call the constructor of the base class
        super().__init__()
        # Variable allocation
        self._c_lock_stat = threading.Lock()
        # Initialize the statistical data (For more configuration it needs to become instance instead of static
        self._progress_data = ProgressData(run_time=0, number_of_filtered_points=0,
                                           progress_status=ProgressStatus.Idle)
        self._progress_run_time = RunTime()
        self._sbc_config = None
        self._is_study_available = False

        self._study_in_memory = None
        self._study_in_storage = None
        self._fixed_parameters = None

        # General optimization parameter
        self._toml_general: sbc_tc.TomlSbcGeneral | None = None
        # Circuit optimization parameter
        self._toml_circuit: sbc_tc.TomlSbcCircuitParetoDesign | None = None

    @staticmethod
    def get_topology_type() -> TopologyEnum:
        """
        Provide the topology type.

        :return: Type of the topology
        :rtype:  TopologyEnum
        """
        return TopologyEnum.sbc

    def save_config(self) -> None:
        """Save the actual configuration file as pickle file on the disk."""
        # Check if a configuration is loaded
        if self._sbc_config is None:
            logger.warning("Circuit configuration is empty!\n    Configuration is not saved!")
            return

        os.makedirs(self.circuit_study_data.optimization_directory, exist_ok=True)
        with open(f"{self.circuit_study_data.optimization_directory}/{self._sbc_config.circuit_study_name}.pkl", 'wb') as output:
            pickle.dump(self._sbc_config, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_stored_config(act_circuit_study_data: StudyData) -> circuit_dtos.CircuitParetoSbcDesign:
        """
        Load pickle configuration file from disk.

        :param act_circuit_study_data: Information about the circuit study name and study path
        :type  act_circuit_study_data: StudyData
        :return: Configuration file as circuit_dtos.CircuitParetoSbcDesign
        :rtype: circuit_dtos.CircuitParetoSbcDesign
        """
        config_pickle_filepath = os.path.join(act_circuit_study_data.optimization_directory,
                                              f"{act_circuit_study_data.study_name}.pkl")

        with open(config_pickle_filepath, 'rb') as pickle_file_data:
            loaded_pareto_dto = pickle.load(pickle_file_data)
            if not isinstance(loaded_pareto_dto, circuit_dtos.CircuitParetoSbcDesign):
                raise TypeError(f"Loaded pickle file {loaded_pareto_dto} not of type CircuitParetoSbcDesign.")

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

        # Convert and check toml file content
        toml_general: sbc_tc.TomlSbcGeneral = sbc_tc.TomlSbcGeneral(**toml_dict)

        # Output range parameter and sampling parameter check
        group_name = "parameter_range or sampling"
        # Init is_user_point_list_consistent-flag
        is_user_point_list_consistent = False
        # Evaluate list length
        len_additional_user_v1 = len(toml_general.sampling.v1_additional_user_point_list)
        len_additional_user_v2 = len(toml_general.sampling.v2_additional_user_point_list)
        len_additional_user_i2 = len(toml_general.sampling.i2_additional_user_point_list)
        len_additional_user_w = len(toml_general.sampling.additional_user_weighting_point_list)
        len_check1 = len_additional_user_v1 == len_additional_user_v2 and len_additional_user_i2 == len_additional_user_w
        len_check2 = len_additional_user_i2 == len_additional_user_w
        # Check if the additional user point lists are consistent
        if len_check1 and len_check2:
            is_user_point_list_consistent = True
        else:
            act_report = f"    The number of list entries in v1_additional_user_point_list ({len_additional_user_v1}), "
            act_report = act_report + f"v2_additional_user_point_list ({len_additional_user_v2}),\n"
            act_report = act_report + f"    i2_additional_user_point_list ({len_additional_user_i2}) and "
            act_report = act_report + f"additional_user_weighting_point_list ({len_additional_user_w} "
            act_report = act_report + "needs to be the same!\n)"
            inconsistency_report = (inconsistency_report + act_report)
            is_consistent = False

        # Perform the boundary check of v1_min_max_list
        is_check_passed, issue_report = BoundaryCheck.check_float_min_max_values(
            0, 1000, toml_general.parameter_range.v1_min_max_list, "parameter_range: v1_min_max_list", c_flag.check_inclusive,
            c_flag.check_inclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False
        elif is_user_point_list_consistent:
            for power_value in toml_general.sampling.v1_additional_user_point_list:
                is_check_passed, issue_report = BoundaryCheck.check_float_value(
                    toml_general.parameter_range.v1_min_max_list[0], toml_general.parameter_range.v1_min_max_list[1],
                    power_value,
                    "sampling: v1_additional_user_point_list", c_flag.check_inclusive, c_flag.check_inclusive)
                if not is_check_passed:
                    inconsistency_report = inconsistency_report + issue_report
                    is_consistent = False

        # Perform the boundary check  of v2_min_max_list
        is_check_passed, issue_report = BoundaryCheck.check_float_min_max_values(
            0, 1000, toml_general.parameter_range.v2_min_max_list, "parameter_range: v2_min_max_list", c_flag.check_inclusive,
            c_flag.check_inclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False
        else:
            # Check lower boundary with respect to v1_min_max_list if v2_min_max_list is consistent
            if toml_general.parameter_range.v1_min_max_list[0] * MAX_DUTY_CYCLE < toml_general.parameter_range.v2_min_max_list[1]:
                issue_report = (f"Inconsistency in v2_min_max_list: Minimum of v1="
                                f"{toml_general.parameter_range.v1_min_max_list[0]} * f{MAX_DUTY_CYCLE} is less equal maximum "
                                f"of v2={toml_general.parameter_range.v1_min_max_list[0]}.")
                inconsistency_report = inconsistency_report + issue_report
                is_check_passed = False
            else:
                is_check_passed = True
            # Check upper boundary with respect to v1_min_max_list
            if toml_general.parameter_range.v1_min_max_list[0] <= toml_general.parameter_range.v2_min_max_list[1]:
                issue_report = (f"Inconsistency in v2_min_max_list: Maximum of v1="
                                f"{toml_general.parameter_range.v1_min_max_list[0]} is less equal maximum "
                                f"of v2={toml_general.parameter_range.v1_min_max_list[1]}.")
                inconsistency_report = inconsistency_report + issue_report
                is_check_passed = False
            if not is_check_passed:
                is_consistent = False
            elif is_user_point_list_consistent:
                for index, voltage_value in enumerate(toml_general.sampling.v2_additional_user_point_list):
                    is_check_passed, issue_report = BoundaryCheck.check_float_value(
                        toml_general.parameter_range.v2_min_max_list[0],
                        toml_general.parameter_range.v2_min_max_list[1], voltage_value,
                        "sampling: v2_additional_user_point_list", c_flag.check_inclusive, c_flag.check_inclusive)
                    if not is_check_passed:
                        inconsistency_report = inconsistency_report + issue_report
                        is_consistent = False
                    if voltage_value >= toml_general.sampling.v1_additional_user_point_list[index]:
                        issue_report = (f"Inconsistency in additional point list at index {index}."
                                        f"v1={toml_general.sampling.v2_additional_user_point_list[index]} "
                                        f"is less equal v2={voltage_value}.")
                        inconsistency_report = inconsistency_report + issue_report
                        is_consistent = False

        # Perform the boundary check  of i2_min_max_list
        is_check_passed, issue_report = BoundaryCheck.check_float_min_max_values(
            -300, 300, toml_general.parameter_range.i2_min_max_list, "parameter_range: i2_min_max_list", c_flag.check_inclusive,
            c_flag.check_inclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False
        elif is_user_point_list_consistent:
            for power_value in toml_general.sampling.i2_additional_user_point_list:
                is_check_passed, issue_report = BoundaryCheck.check_float_value(
                    toml_general.parameter_range.i2_min_max_list[0], toml_general.parameter_range.i2_min_max_list[1],
                    power_value,
                    "sampling: i2_additional_user_point_list", c_flag.check_inclusive, c_flag.check_inclusive)
                if not is_check_passed:
                    inconsistency_report = inconsistency_report + issue_report
                    is_consistent = False

        # Remaining Sampling parameter check
        group_name = "sampling"
        # Check additional_user_weighting_point_list
        # Initialize variable
        weighting_sum: float = 0.0
        # Perform the boundary check  of additional_user_weighting_point_list
        for weight_value in toml_general.sampling.additional_user_weighting_point_list:
            is_check_passed, issue_report = BoundaryCheck.check_float_value(
                0, 1, weight_value, "additional_user_weighting_point_list", c_flag.check_inclusive,
                c_flag.check_inclusive)
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
        toml_check_value_list: list[tuple[float, str]]

        toml_circuit: sbc_tc.TomlSbcCircuitParetoDesign = sbc_tc.TomlSbcCircuitParetoDesign(**toml_dict)

        # Design space parameter check
        # Create dictionary from transistor database list
        db = tdb.DatabaseManager()
        db.set_operation_mode_json()
        if not is_tdb_to_update:
            # Debug ASA comment out
            # db.update_from_file_exchange(True)
            pass

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

        # Check l_s_min_max_list
        is_check_passed, issue_report = BoundaryCheck.check_float_min_max_values(
            0, 1, toml_circuit.design_space.l_s_min_max_list,
            "parameter_range: l_s_min_max_list", c_flag.check_exclusive, c_flag.check_exclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Check l_s_min_max_list
        is_check_passed, issue_report = BoundaryCheck.check_float_min_max_values(
            0, 1, toml_circuit.design_space.l_s_min_max_list,
            "l_s_min_max_list", c_flag.check_inclusive, c_flag.check_exclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Check ripple_quality
        # Perform the boundary check
        is_check_passed, issue_report = BoundaryCheck.check_float_value(
            0.5, 1e4, toml_circuit.design_space.ripple_quality, "ripple_quality", c_flag.check_inclusive, c_flag.check_inclusive)
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

        # Perform list length check for transistor_hs_cooling
        if len(toml_circuit.thermal_data.transistor_hs_cooling) != 2:
            inconsistency_report = inconsistency_report + "    Number of values in parameter 'transistor_hs_cooling' is not equal 2!\n"
            is_consistent = False
        else:
            toml_check_value_list1.append(
                (toml_circuit.thermal_data.transistor_hs_cooling[0], f"{group_name}: transistor_hs_cooling[0]-tim_thickness"))
            toml_check_value_list2.append(
                (toml_circuit.thermal_data.transistor_hs_cooling[1], f"{group_name}: transistor_hs_cooling[1]-tim_conductivity"))

        # Perform list length check for transistor_ls_cooling
        if len(toml_circuit.thermal_data.transistor_hs_cooling) != 2:
            inconsistency_report = inconsistency_report + "    Number of values in parameter 'transistor_ls_cooling' is not equal 2!\n"
            is_consistent = False
        else:
            toml_check_value_list1.append(
                (toml_circuit.thermal_data.transistor_hs_cooling[0], f"{group_name}: transistor_ls_cooling[0]-tim_thickness"))
            toml_check_value_list2.append(
                (toml_circuit.thermal_data.transistor_hs_cooling[1], f"{group_name}: transistor_ls_cooling[1]-tim_conductivity"))

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
            raise ValueError("Serious programming error 'sbc multiple allocation failure'. Please write an issue!")

        # Initialize the circuit_dtos
        design_space = circuit_dtos.CircuitParetoDesignSpace(
            f_s_min_max_list=self._toml_circuit.design_space.f_s_min_max_list,
            l_s_min_max_list=self._toml_circuit.design_space.l_s_min_max_list,
            ripple_quality=self._toml_circuit.design_space.ripple_quality,
            transistor_1_name_list=self._toml_circuit.design_space.transistor_1_name_list,
            transistor_2_name_list=self._toml_circuit.design_space.transistor_2_name_list,
            c_par_1=self._toml_circuit.design_space.c_par_1,
            c_par_2=self._toml_circuit.design_space.c_par_2

        )

        parameter_range = circuit_dtos.CircuitParameterRange(
            v1_min_max_list=self._toml_general.parameter_range.v1_min_max_list,
            duty_cycle_min_max_list=SbcCircuitOptimization.calculate_voltage_to_duty_cycle_min_max_list(
                self._toml_general.parameter_range),
            i2_min_max_list=self._toml_general.parameter_range.i2_min_max_list)

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
            duty_cycle_additional_user_point_list=self._toml_general.sampling.v2_additional_user_point_list,
            i2_additional_user_point_list=self._toml_general.sampling.i2_additional_user_point_list,
            additional_user_weighting_point_list=self._toml_general.sampling.additional_user_weighting_point_list
        )

        self._sbc_config = circuit_dtos.CircuitParetoSbcDesign(
            circuit_study_name=self.circuit_study_data.study_name,
            project_directory=self.project_directory,
            design_space=design_space,
            parameter_range=parameter_range,
            sampling=sampling,
            filter=filter)

        return True

    @staticmethod
    def calculate_voltage_to_duty_cycle_min_max_list(act_parameter_range: sbc_tc.TomlSbcParameterRange) -> list[float]:
        """Calculate the duty cycle list based on v1 and v2 lists.

        :param act_parameter_range: Parameter range of voltages and current
        :type  act_parameter_range: sbc_tc.TomlSbcParameterRange
        :return: Duty cycle minimum-maximum list
        :rtype:  list[float]
        """
        # Variable declaration
        duty_cycle_min_max_list: list[float] = []
        # Minimum value calculated from maximum v1-voltage and minimum v2-voltage
        duty_cycle_min_max_list.append(act_parameter_range.v2_min_max_list[0] / act_parameter_range.v1_min_max_list[1])
        # Maximum value calculated from minimum v1-voltage and maximum v2-voltage
        duty_cycle_min_max_list.append(act_parameter_range.v2_min_max_list[1] / act_parameter_range.v1_min_max_list[0])

        return duty_cycle_min_max_list

    def get_config(self) -> circuit_dtos.CircuitParetoSbcDesign | None:
        """
        Return the actual loaded configuration file.

        :return: Configuration file as circuit_dtos.CircuitParetoSbcDesign
        :rtype: circuit_dtos.CircuitParetoSbcDesign
        """
        if self._sbc_config is None:
            logger.warning("Configuration is not loaded!")

        return copy.deepcopy(self._sbc_config)

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
    def calculate_fixed_parameters(act_sbc_config: circuit_dtos.CircuitParetoSbcDesign) -> s_dtos.FixedParameters:
        """
        Calculate time-consuming parameters which are same for every single simulation.

        :param act_sbc_config: Sbc circuit configuration
        :type act_sbc_config: circuit_dtos.CircuitParetoSbcDesign
        :return: Fix parameters (transistor DTOs)
        :rtype: s_dtos.FixedParameters
        """
        transistor_1_dto_list = []
        transistor_2_dto_list = []

        for transistor in act_sbc_config.design_space.transistor_1_name_list:
            transistor_1_dto_list.append(d_sets.HandleTransistorDto.tdb_to_transistor_dto(transistor))

        for transistor in act_sbc_config.design_space.transistor_2_name_list:
            transistor_2_dto_list.append(d_sets.HandleTransistorDto.tdb_to_transistor_dto(transistor))

        # choose sampling method
        if act_sbc_config.sampling.sampling_method == SamplingEnum.meshgrid:
            steps_per_dimension = int(np.ceil(np.power(act_sbc_config.sampling.sampling_points, 1 / 3)))
            logger.info(f"Number of sampling points has been updated from {act_sbc_config.sampling.sampling_points} to {steps_per_dimension ** 3}.")
            logger.info("Note: meshgrid sampling does not take user-given operating points into account")
            v_operating_points, duty_cycle_operating_points, i_operating_points = np.meshgrid(
                np.linspace(act_sbc_config.parameter_range.v1_min_max_list[0], act_sbc_config.parameter_range.v1_min_max_list[1],
                            steps_per_dimension),
                np.linspace(act_sbc_config.parameter_range.duty_cycle_min_max_list[0],
                            act_sbc_config.parameter_range.duty_cycle_min_max_list[1], steps_per_dimension),
                np.linspace(act_sbc_config.parameter_range.i2_min_max_list[0], act_sbc_config.parameter_range.i2_min_max_list[1],
                            steps_per_dimension),
                sparse=False)
        elif act_sbc_config.sampling.sampling_method == SamplingEnum.latin_hypercube:
            v_operating_points, duty_cycle_operating_points, i_operating_points = sampling.latin_hypercube(
                act_sbc_config.parameter_range.v1_min_max_list[0], act_sbc_config.parameter_range.v1_min_max_list[1],
                act_sbc_config.parameter_range.duty_cycle_min_max_list[0], act_sbc_config.parameter_range.duty_cycle_min_max_list[1],
                act_sbc_config.parameter_range.i2_min_max_list[0], act_sbc_config.parameter_range.i2_min_max_list[1],
                total_number_points=act_sbc_config.sampling.sampling_points,
                dim_1_user_given_points_list=act_sbc_config.sampling.v1_additional_user_point_list,
                dim_2_user_given_points_list=act_sbc_config.sampling.duty_cycle_additional_user_point_list,
                dim_3_user_given_points_list=act_sbc_config.sampling.i2_additional_user_point_list,
                sampling_random_seed=act_sbc_config.sampling.sampling_random_seed)
        elif act_sbc_config.sampling.sampling_method == SamplingEnum.dessca:
            v_operating_points, duty_cycle_operating_points, i_operating_points = sampling.dessca(
                act_sbc_config.parameter_range.v1_min_max_list[0], act_sbc_config.parameter_range.v1_min_max_list[1],
                act_sbc_config.parameter_range.duty_cycle_min_max_list[0], act_sbc_config.parameter_range.duty_cycle_min_max_list[1],
                act_sbc_config.parameter_range.i2_min_max_list[0], act_sbc_config.parameter_range.i2_min_max_list[1],
                total_number_points=act_sbc_config.sampling.sampling_points,
                dim_1_user_given_points_list=act_sbc_config.sampling.v1_additional_user_point_list,
                dim_2_user_given_points_list=act_sbc_config.sampling.duty_cycle_additional_user_point_list,
                dim_3_user_given_points_list=act_sbc_config.sampling.i2_additional_user_point_list,
                sampling_random_seed=act_sbc_config.sampling.sampling_random_seed)
        else:
            raise ValueError(f"sampling_method '{act_sbc_config.sampling.sampling_method}' not available.")

        logger.debug(f"{v_operating_points=}")

        # calculate weighting

        if act_sbc_config.sampling.sampling_method == SamplingEnum.meshgrid:
            weight_sum = 0
            given_user_points = 0
        else:
            weight_sum = np.sum(act_sbc_config.sampling.additional_user_weighting_point_list)
            logger.debug(f"{weight_sum=}")
            given_user_points = len(act_sbc_config.sampling.v1_additional_user_point_list)
        logger.debug(f"{given_user_points=}")
        logger.debug(f"{v_operating_points.size=}")

        if weight_sum > 1 or weight_sum < 0:
            raise ValueError("Sum of weighting point list must be within 0 and 1.")
        else:
            leftover_auto_weight = (1 - weight_sum) / (v_operating_points.size - given_user_points)
            logger.info(f"Auto-weight given for all other {v_operating_points.size - given_user_points} operating points: {leftover_auto_weight}")
            # default case, same weights for all points
            weights = np.full_like(v_operating_points, leftover_auto_weight)
            # for user point weightings, both lists must be filled.
            if act_sbc_config.sampling.additional_user_weighting_point_list and act_sbc_config.sampling.sampling_method != SamplingEnum.meshgrid:
                logger.debug("Given user weighting point list detected, fill up with user-given weights.")
                weights[-len(act_sbc_config.sampling.additional_user_weighting_point_list):] = act_sbc_config.sampling.additional_user_weighting_point_list
            logger.debug(f"{weights=}")
            logger.debug(f"Double check: Sum of weights = {np.sum(weights)}")

        # Return result as column vector
        return s_dtos.FixedParameters(
            transistor_1_dto_list=transistor_1_dto_list,
            transistor_2_dto_list=transistor_2_dto_list,
            mesh_v1=v_operating_points[:, np.newaxis],
            mesh_duty_cycle=duty_cycle_operating_points[:, np.newaxis],
            mesh_i2=i_operating_points[:, np.newaxis],
            mesh_weights=weights[:, np.newaxis]
        )

    def run_optimization(self, act_number_trials: int) -> pd.DataFrame:
        """Proceed a study which bases on functional data.

        :type act_number_trials: int
        :param act_number_trials: Number of optimization trials
        """
        # Variable declaration
        circuit_id: int = 0
        df_circuit_study = pd.DataFrame()

        if self._sbc_config is None:
            logger.warning("Circuit configuration is not initialized!")
            return df_circuit_study
        elif self._fixed_parameters is None:
            logger.warning("Parameter calculation is missing!")
            return df_circuit_study

        # Function to execute
        # Start with fs low
        ts_min = 1 / self._sbc_config.design_space.f_s_min_max_list[1]
        ts_max = 1 / self._sbc_config.design_space.f_s_min_max_list[0]
        ts_norm = ts_max - ts_min
        # Calculate oversampling with 10
        act_number_points = round(act_number_trials / len(self._fixed_parameters.transistor_1_dto_list))
        n = act_number_points * 10

        # Calculate the switch period factor to get the required inductor
        # i_ripple defined by ripple_quality : i_ripple = i_2/ ripple_quality
        i_ripple = self._fixed_parameters.mesh_i2 / self._sbc_config.design_space.ripple_quality
        # L  = max(V_out/i_ripple * t_off) =  max(V_out/i_ripple * (1 - duty_cycle) * ts ) = max(V_in * duty_cycle /i_ripple * (1 - duty_cycle) * ts )
        # i_ripple defined by quality factor ripple_quality: L = max(V_in * duty_cycle * ripple_quality /i_2 * (1 - duty_cycle) * ts )

        k_ts_inductor_array = self._fixed_parameters.mesh_v1 * self._fixed_parameters.mesh_duty_cycle * (1 - self._fixed_parameters.mesh_duty_cycle) / i_ripple
        k_ts_inductor = k_ts_inductor_array.max()

        # Calculate curve length and curve points of all transistors
        for transistor_dto in self._fixed_parameters.transistor_1_dto_list:
            curve_ts: list[float] = []
            curve_ploss: list[np.ndarray] = []
            actual_ts = ts_min
            required_L = k_ts_inductor * actual_ts
            # Calculate the loss value at maximum ts
            sbc_calc = d_sets.HandleSbcDto.init_config(
                name=self._sbc_config.circuit_study_name,
                mesh_v1=self._fixed_parameters.mesh_v1,
                mesh_duty_cycle=self._fixed_parameters.mesh_duty_cycle,
                mesh_i2=self._fixed_parameters.mesh_i2,
                sampling=self._sbc_config.sampling,
                ls=required_L,
                fs=1 / actual_ts,
                transistor_dto_1=transistor_dto,
                transistor_dto_2=transistor_dto
            )
            max_loss = sbc_calc.calc_losses.p_sbc_total.max()

            actual_ts = ts_max
            required_L = k_ts_inductor * actual_ts
            # Calculate the loss value at maximum ts
            sbc_calc = d_sets.HandleSbcDto.init_config(
                name=self._sbc_config.circuit_study_name,
                mesh_v1=self._fixed_parameters.mesh_v1,
                mesh_duty_cycle=self._fixed_parameters.mesh_duty_cycle,
                mesh_i2=self._fixed_parameters.mesh_i2,
                sampling=self._sbc_config.sampling,
                ls=required_L,
                fs=1 / actual_ts,
                transistor_dto_1=transistor_dto,
                transistor_dto_2=transistor_dto
            )
            loss_norm = max_loss - sbc_calc.calc_losses.p_sbc_total.max()

            # Store the first point
            curve_ts.append(actual_ts)
            curve_ploss.append(sbc_calc.calc_losses.p_sbc_total.max())
            # Define initial fine granular dynamic delta_x
            d_ts = (ts_max - ts_min) / n
            actual_ts = actual_ts - d_ts
            required_L = k_ts_inductor * actual_ts
            # Calculate the loss value at maximum ts
            sbc_calc = d_sets.HandleSbcDto.init_config(
                name=self._sbc_config.circuit_study_name,
                mesh_v1=self._fixed_parameters.mesh_v1,
                mesh_duty_cycle=self._fixed_parameters.mesh_duty_cycle,
                mesh_i2=self._fixed_parameters.mesh_i2,
                sampling=self._sbc_config.sampling,
                ls=required_L,
                fs=1 / actual_ts,
                transistor_dto_1=transistor_dto,
                transistor_dto_2=transistor_dto
            )

            # Store second point
            curve_ts.append(actual_ts)
            curve_ploss.append(sbc_calc.calc_losses.p_sbc_total.max())

            # Calculate the normalized fine constant arc distance
            m = (curve_ploss[-1] - curve_ploss[-2]) / loss_norm / d_ts * ts_norm
            darc_fine_norm = d_ts / ts_norm * ((1 + m ** 2) ** 0.5)

            # Calculate the curve length by fine granular approach
            while actual_ts > ts_min:
                actual_ts = actual_ts - d_ts
                curve_ts.append(actual_ts)
                required_L = k_ts_inductor * actual_ts
                # Calculate the loss value
                sbc_calc = d_sets.HandleSbcDto.init_config(
                    name=self._sbc_config.circuit_study_name,
                    mesh_v1=self._fixed_parameters.mesh_v1,
                    mesh_duty_cycle=self._fixed_parameters.mesh_duty_cycle,
                    mesh_i2=self._fixed_parameters.mesh_i2,
                    sampling=self._sbc_config.sampling,
                    ls=required_L,
                    fs=1/actual_ts,
                    transistor_dto_1=transistor_dto,
                    transistor_dto_2=transistor_dto
                )
                curve_ploss.append(sbc_calc.calc_losses.p_sbc_total.max())

                m = (curve_ploss[-1] - curve_ploss[-2]) / loss_norm / d_ts * ts_norm
                d_ts = darc_fine_norm * ts_norm / ((1 + m ** 2) ** 0.5)

            # Calculate x-values
            index_list = SbcCircuitOptimization._get_ts_index(curve_ts, act_number_points)
            # Collect data at x-values
            l_s_list: list[float] = []
            circuit_id_list: list[int] = []
            result_curve_ploss: list[np.ndarray] = []
            result_curve_ts: list[float] = []
            result_curve_fs: list[float] = []
            for idx in index_list:
                l_s_list.append(k_ts_inductor * curve_ts[idx])
                result_curve_ts.append(curve_ts[idx])
                result_curve_fs.append(1/curve_ts[idx])
                result_curve_ploss.append(curve_ploss[idx])
                circuit_id_list.append(circuit_id)
                circuit_id = circuit_id + 1

            # Transfer to DataFrame
            df = pd.DataFrame(
                {"number": circuit_id_list, "values_0": result_curve_ts, "values_1": result_curve_ploss,
                 "params_l_s_suggest": l_s_list, "params_f_s_suggest": result_curve_fs,
                 "params_transistor_1_name_suggest": transistor_dto.name,
                 "params_transistor_2_name_suggest": transistor_dto.name})

            df_circuit_study = pd.concat([df_circuit_study, df], ignore_index=True)

        return df_circuit_study

    @staticmethod
    def _get_ts_index(act_curve_x: list[float], act_number_points: int) -> list[int]:
        """Provide the list of x-Values from curve_x.

        :param act_curve_x: curve with x-values
        :type  act_curve_x: float
        :param act_number_points: Number of points to get from curve
        :type  act_number_points: int
        """
        # Variable declaration
        index_list: list[int] = []
        # Calculate the number of requested points
        n_fine = len(act_curve_x)
        dn = n_fine / act_number_points
        index = 0.0
        # Sample the curve
        while round(index) < len(act_curve_x):
            idx = round(index)
            index_list.append(idx)
            index = index + dn

        return index_list

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
        # Check if data are initialized
        if self._sbc_config is None:
            logger.warning("Method 'initialize_circuit_optimization' is not called!\n"
                           "    No list is generated so that no optimization can be performed!")
            return

        # Create dummy database file path
        circuit_study_sqlite_database = os.path.join(self.circuit_study_data.optimization_directory,
                                                     f"{self._sbc_config.circuit_study_name}.sqlite3")

        if os.path.exists(circuit_study_sqlite_database):
            logger.info("Existing circuit study will be overwritten.")
        else:
            os.makedirs(f"{self.circuit_study_data.optimization_directory}", exist_ok=True)

        # Calculate the fixed parameters
        self._fixed_parameters = SbcCircuitOptimization.calculate_fixed_parameters(self._sbc_config)

        # Update statistical data
        with self._c_lock_stat:
            self._progress_data.run_time = self._progress_run_time.get_runtime()
            self._progress_run_time.reset_start_trigger()
            self._progress_data.progress_status = ProgressStatus.InProgress

            # Inform about sampler type
            logger.info("No sampler is used")
            # Start optimization
            self._study_in_storage = self.run_optimization(number_trials)
            # Store as sqlite file
            sqlite_handler = sqlite3.connect(circuit_study_sqlite_database)
            self._study_in_storage.to_sql(self.circuit_study_data.study_name, sqlite_handler, if_exists='replace', index=False)
            sqlite_handler.close()

            logger.info(f"Calculate {number_trials} trials.")
            logger.info(f"current time: {datetime.datetime.now()}")
            self.save_config()

            self.save_study_results_pareto(show_results=False)

        # Set flag _is_study_available to indicate, that the study is available for filtering
        self._is_study_available = True

    def save_study_results_pareto(self, show_results: bool = False) -> None:
        """Show the results of a study.

        A local .html file is generated under config.working_directory to store the interactive plotly plots on disk.

        :param show_results: True to directly open the browser to show the study results.
        :type show_results: bool
        """
        if self._study_in_storage is None:
            logger.warning("The study is not initialized!")
            return
        elif self._sbc_config is None:
            logger.warning("Circuit configuration is not initialized!")
            return

        self._study_in_storage.plot.scatter(x='values_0', y='values_1')
        plt.title('Circuit optimization result')
        plt.xlabel('switching period/s')
        plt.ylabel('transistor loss/W')
        plt.savefig(f"{self.circuit_study_data.optimization_directory}"
                    f"_{datetime.datetime.now().isoformat(timespec='minutes')}.png")
        if show_results:
            plt.show()

    @staticmethod
    def load_sbc_dto_from_study(act_study_data: StudyData, sbc_config: circuit_dtos.CircuitParetoSbcDesign,
                                trial_number: int | None = None) -> s_dtos.SbcCircuitDTO:
        """
        Load a SBC-DTO from an optuna study.

        :param act_study_data: Information about the circuit study name and study path
        :type  act_study_data: StudyData
        :param sbc_config: SBC optimization configuration file
        :type  sbc_config: circuit_dtos.CircuitParetoSbcDesign
        :param trial_number: trial number to load to the DTO
        :type trial_number: int
        :return: DTO
        :rtype:  s_dtos.SbcCircuitDTO
        """
        if trial_number is None:
            raise NotImplementedError("needs to be implemented")

        database_url = SbcCircuitOptimization.create_sqlite_database_url(act_study_data)

        loaded_study = optuna.create_study(study_name=act_study_data.study_name,
                                           storage=database_url, load_if_exists=True)
        logger.info(f"The study '{act_study_data.study_name}' contains {len(loaded_study.trials)} trials.")
        trials_dict = loaded_study.trials[trial_number].params

        fix_parameters = SbcCircuitOptimization.calculate_fixed_parameters(sbc_config)

        sbc_dto = d_sets.HandleSbcDto.init_config(
            name=str(trial_number),
            mesh_v1=fix_parameters.mesh_v1,
            mesh_duty_cycle=fix_parameters.mesh_duty_cycle,
            mesh_i2=fix_parameters.mesh_i2,
            sampling=sbc_config.sampling,
            ls=trials_dict["l_s_suggest"],
            fs=trials_dict["f_s_suggest"],
            transistor_dto_1=trials_dict["transistor_1_name_suggest"],
            transistor_dto_2=trials_dict["transistor_2_name_suggest"]
        )

        return sbc_dto

    def df_to_sbc_dto_list(self, df: pd.DataFrame) -> list[s_dtos.SbcCircuitDTO]:
        """
        Load a SBC-DTO from an optuna study.

        :param df: Pandas DataFrame to convert to the SBC-DTO list
        :type df: pd.DataFrame
        :return: List of DTO
        :rtype:  list[s_dtos.SbcCircuitDTO]
        """
        sbc_dto_list: list[s_dtos.SbcCircuitDTO] = []

        # Check if configuration is not available or fixed parameters are not available
        if self._sbc_config is None:
            logger.warning("Circuit configuration is not initialized!")
            return sbc_dto_list
        elif self._fixed_parameters is None:
            logger.warning("Missing initialized fixed parameters!")
            return sbc_dto_list

        logger.info(f"The study '{self._sbc_config.circuit_study_name}' contains {len(df)} trials.")

        index: int
        for idx, _ in df.iterrows():

            index = cast(int, idx)

            transistor_dto_1 = d_sets.HandleTransistorDto.tdb_to_transistor_dto(str(df.at[index, "params_transistor_1_name_suggest"]))
            transistor_dto_2 = d_sets.HandleTransistorDto.tdb_to_transistor_dto(str(df.at[index, "params_transistor_1_name_suggest"]))

            sbc_dto = d_sets.HandleSbcDto.init_config(
                name=str(df.at[index, "number"]),
                mesh_v1=self._fixed_parameters.mesh_v1,
                mesh_duty_cycle=self._fixed_parameters.mesh_duty_cycle,
                mesh_i2=self._fixed_parameters.mesh_i2,
                sampling=self._sbc_config.sampling,
                ls=float(cast(SupportsFloat, df.at[index, "params_l_s_suggest"])),
                fs=float(cast(SupportsFloat, df.at[index, "params_f_s_suggest"])),
                transistor_dto_1=transistor_dto_1,
                transistor_dto_2=transistor_dto_2
            )
            sbc_dto_list.append(sbc_dto)

        return sbc_dto_list

    @staticmethod
    def study_to_df(act_study_data: StudyData) -> pd.DataFrame:
        """Create a DataFrame from a study.

        :param act_study_data: Information about the circuit study name and study path
        :type  act_study_data: StudyData
        :return: study result data transferred to Pandas dataframe
        :rtype: pd.DataFrame
        """
        df = pd.read_csv(f"{act_study_data.optimization_directory}/{act_study_data.study_name}.csv")
        return df

    @staticmethod
    def create_sqlite_database_url(act_study_data: StudyData) -> str:
        """
        Create the SBC circuit optimization sqlite URL.

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

    @staticmethod
    def hybrid_pareto_sampling(pareto_matrix: np.ndarray, n_points: int = 8) -> np.ndarray:
        """
        Filter points from pareto front by hybrid-strategy: Extremes + Knees + Density.

        This method filters points from a Pareto front using a hybrid strategy that combines three complementary
        selection criteria: extremes, knees, and density.
        Extremes: Points that represent the best values in at least one objective, ensuring that the boundary solutions
        with maximal trade-offs are preserved.
        Knees: Points located at significant curvature changes along the Pareto front, highlighting well-balanced compromises
        where small sacrifices in one objective yield large gains in others.
        Density: Points selected based on the distribution density of solutions to reduce redundancy by representing clusters
        with fewer, but diverse, representatives.
        By integrating these strategies, the function provides a compact, representative subset of the Pareto front
        that balances multiple selection approaches.

        :param pareto_matrix: pareto front points with shape (n,2)
        :type pareto_matrix: np.ndarray
        :param n_points: Number of filtered points
        :type n_points: int
        :returns: filtered points
        :rtype: np.ndarray
        """
        # Constant values
        # Filter window and filter order
        smooth_window: int = 11
        filter_order: int = 3
        # Prevent division by 0
        prev_diff_zero = 1e-12

        # Number of selected points with point density method(round up)
        n_points_cdf: int = (n_points+1) // 2

        # Copy, sort and normalize
        pareto_matrix_copy = pareto_matrix.copy().astype(float)
        # Sort to target 1
        pareto_matrix_copy = pareto_matrix_copy[np.argsort(pareto_matrix_copy[:, 0])]
        # Reduce similar x values (Strong monotone falling) (function returns [value, index]
        _, unique_x_indices = np.unique(pareto_matrix_copy[:, 0], return_index=True)
        pareto_matrix_copy = pareto_matrix_copy[unique_x_indices]
        # Normalize the curve to [0,1]
        pareto_norm = (pareto_matrix_copy - pareto_matrix_copy.min(axis=0)) / (pareto_matrix_copy.max(axis=0) - pareto_matrix_copy.min(axis=0))

        # #### Debug  ###############################################################
        df_filtered_par = pd.DataFrame({"x": pareto_norm[:, 0], "y": pareto_norm[:, 1]})
        # Store the data
        # df_filtered_par.to_csv("path to define + /Pareto_data.csv")
        # ###########################################################################

        # Curvature analysis due to the knee
        # Smooth the curve with Savitzky–Golay-Filter
        smooth_target_2 = savgol_filter(pareto_norm[:, 1], smooth_window, filter_order)
        # #### Debug  ###############################################################
        df_filtered_file = pd.DataFrame({"x": pareto_norm[:, 0], "y": smooth_target_2})
        # Store the data
        # df_filtered_file.to_csv("path to define + Pareto_filter_data.csv")
        # ###########################################################################

        # First and second derivative
        d_target_2 = np.gradient(smooth_target_2, pareto_norm[:, 0])
        d2_target_2 = np.gradient(d_target_2, pareto_norm[:, 0])
        # curvature calculation (Formula at 13.3.8 see url)
        # https://math.libretexts.org/Bookshelves/Calculus/Calculus_(OpenStax)/13%3A_Vector-Valued_Functions/13.03%3A_Arc_Length_and_Curvature
        curvature = np.abs(d2_target_2) / (1 + d_target_2 ** 2) ** 1.5

        # Length of the curve
        d_target_1 = np.diff(pareto_norm[:, 0])
        d_target_2 = np.diff(pareto_norm[:, 1])
        d_curve = np.sqrt(d_target_1 ** 2 + d_target_2 ** 2)
        curve = np.concatenate(([0], np.cumsum(d_curve)))
        curve = curve / curve[-1]  # Standardize to [0,1]

        # Weighting: Curvature * local density
        density = np.ones_like(curve)
        for i in range(1, len(curve) - 1):
            density[i] = 0.5 * (curve[i + 1] - curve[i - 1])
        inv_density = 1 / (density + prev_diff_zero)
        weight = curvature * inv_density

        # Selection of points according point density (inclusive extreme values)
        result_point_vector = np.linspace(0, 1, n_points_cdf)
        cdf = np.cumsum(weight)
        cdf = cdf / cdf[-1]
        # Create point density index vector
        cdf_filtered_id: list[int] = []
        # For loop over result_point_vector
        for t in result_point_vector:
            idx = int(np.argmin(np.abs(cdf - t)))
            cdf_filtered_id.append(idx)

        # Remove same points
        cdf_filtered_id = np.unique(cdf_filtered_id).tolist()
        n_points_cdf = len(cdf_filtered_id)

        # Set number of cluster points
        n_points_cluster = n_points - n_points_cdf

        # Find clusters in regions of high point density
        km = KMeans(n_clusters=n_points_cluster, n_init=15)
        km.fit(pareto_norm)
        cluster_centers = km.cluster_centers_

        cluster_filtered_id: list = []
        # For loop over result_point_vector
        for cluster_center in cluster_centers:
            # Square of  distance of assigned points to the cluster
            dists = np.sum((pareto_norm - cluster_center) ** 2, axis=1)
            # Point index with minimal distance
            idx = int(np.argmin(dists))
            cluster_filtered_id.append(idx)

        # Hybrid strategy: Combine points density selection and cluster points
        chosen_idx = np.concatenate([cdf_filtered_id, cluster_filtered_id])
        # Remove duplicates
        chosen_idx = np.unique(chosen_idx)
        # Return the result in original units
        pareto_matrix_result: np.ndarray = pareto_matrix_copy[chosen_idx]

        return pareto_matrix_result

    @staticmethod
    def filter_equidistant_sampling(pareto_matrix: np.ndarray, n_points: int = 8) -> np.ndarray:
        """
        Filter the study result.

        For SBC a Pseudo Pareto front is provided. Each point have similar distance, if function is normalized.
        Filtering are separated in 2 ranges:
        The middle is identified by the normalized smallest distance to 0
        Based on this middle both branches are taken for filter point
        1 Design: normalized middle (NM)
        2 Designs: Started from NM  1/3 between NM and max and similar to minimum
        3 Designs: Minimum, middle and maximum
        4 Designs: Minimum, 1/3 , 2/3 and maximum
        More than 4 Designs: Max, min and normalized equidistant arc length in between

        :param pareto_matrix: pareto front points with shape (n,2)
        :type pareto_matrix: np.ndarray
        :param n_points: Number of filtered points
        :type n_points: int
        :returns: filtered points
        :rtype: np.ndarray
        """
        # Variable declaration
        pareto_matrix_result: np.ndarray
        p_idx_list: list = []

        # Constant values
        # Copy, sort and normalize
        pareto_matrix_copy = pareto_matrix.copy().astype(float)
        # Normalize the curve to [0,1]
        pareto_norm = (
            (pareto_matrix_copy - pareto_matrix_copy.min(axis=0)) / (pareto_matrix_copy.max(axis=0) - pareto_matrix_copy.min(axis=0)))

        distances = np.linalg.norm(pareto_norm, axis=1)
        middle_idx = np.argmin(distances)
        max_idx = len(pareto_matrix_copy)-1

        # Check if sufficient points are available
        if middle_idx < 2 or (max_idx - middle_idx) < 2:
            raise ValueError(f"The number of Pareto points are {max_idx+1} and "
                             f"the middle index is {middle_idx}."
                             f"The allowed range of middle index is between {2} and {max_idx - 2}."
                             f"Increase the number of trials to solve this issue.")

        # Number of selected points with point density method(round up)
        if n_points == 1:
            pareto_matrix_result = pareto_matrix_copy[middle_idx]
        elif n_points == 2:
            p_idx_list.append(round(middle_idx * 2 / 3))
            p_idx_list.append(round((max_idx / 3) + (2 * middle_idx / 3)))
            pareto_matrix_result = pareto_matrix_copy[p_idx_list]
        elif n_points == 3:
            p_idx_list.append(0)
            p_idx_list.append(middle_idx)
            p_idx_list.append(max_idx)
            pareto_matrix_result = pareto_matrix_copy[p_idx_list]
        elif n_points == 4:
            p_idx_list.append(0)
            p_idx_list.append(round(middle_idx * 2 / 3))
            p_idx_list.append(round((max_idx / 3) + (2 * middle_idx / 3)))
            p_idx_list.append(max_idx)
            pareto_matrix_result = pareto_matrix_copy[p_idx_list]
        else:
            # Calculate the number of requested points
            n_fine = max_idx + 1
            dn = n_fine / n_points
            index = 0.0
            # Sample the curve
            while round(index) < len(pareto_matrix_copy):
                idx = round(index)
                p_idx_list.append(idx)
                index = index + dn
            pareto_matrix_result = pareto_matrix_copy[p_idx_list]

        return pareto_matrix_result

    def filter_study_results(self) -> tuple[bool, str]:
        """Filter the study result.

        For SBC a Pseudo Pareto front is provided, which is filtered for requested number of designs.
        """
        # Variable initialization
        is_filter_available: bool = False
        issue_report: str = ""

        # Check if configuration is not available and if study is not available
        if self._sbc_config is None:
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

        # Set filter data to True
        is_filter_available = True

        self._study_in_storage.to_csv(f'{self.circuit_study_data.optimization_directory}/{self._sbc_config.circuit_study_name}.csv')

        df_pareto_front = CircuitOptimizationBase.pareto_front_from_df_min_min(self._study_in_storage)

        # Filter Pseudo Pareto front
        filtered_points: np.ndarray = SbcCircuitOptimization.filter_equidistant_sampling(
            df_pareto_front[["values_0", "values_1"]].to_numpy(), self._sbc_config.filter.number_filtered_designs)

        # Assign the identified points to the dataframe index
        filtered_indices: list[int] = []
        for point in filtered_points:
            # boolean mask for rows of identical content
            mask = (self._study_in_storage["values_0"].values == point[0]) & (self._study_in_storage["values_1"].values == point[1])
            # Find index in data frame
            idx = int(np.where(mask)[0][0])
            filtered_indices.append(idx)

        # Create selection data frame
        df_selected = self._study_in_storage.loc[filtered_indices]

        # Continue time measurement
        with self._c_lock_stat:
            self._progress_run_time.continue_trigger()

        # Store the result to filtered point folder
        selected_dto_list = self.df_to_sbc_dto_list(df_selected)

        dto_directory = self.filter_data.filtered_list_pathname
        os.makedirs(dto_directory, exist_ok=True)
        for dto in selected_dto_list:
            # Calculate component requirement
            dto = d_sets.HandleSbcDto.generate_components_target_requirements(dto, self.circuit_study_data.study_name)
            # Get thermal data
            transistor_hs_cooling: ComponentCooling = ComponentCooling(
                tim_thickness=self._toml_circuit.thermal_data.transistor_hs_cooling[0],
                tim_conductivity=self._toml_circuit.thermal_data.transistor_hs_cooling[1])
            transistor_ls_cooling: ComponentCooling = ComponentCooling(
                tim_thickness=self._toml_circuit.thermal_data.transistor_ls_cooling[0],
                tim_conductivity=self._toml_circuit.thermal_data.transistor_ls_cooling[1])
            # generate the thermal parameters for the given design
            dto = d_sets.HandleTransistorDto.generate_thermal_transistor_parameters(dto, transistor_hs_cooling, transistor_ls_cooling)

            d_sets.HandleSbcDto.save(dto, dto.circuit_id, directory=dto_directory, timestamp=False)

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
        # Variable declaration
        capacitor_requirements_list: list[CapacitorRequirements] = []

        for circuit_id_file in self.filter_data.filtered_list_files:
            circuit_id_filepath = os.path.join(self.filter_data.filtered_list_pathname, f"{circuit_id_file}.pkl")
            circuit_dto = d_sets.HandleSbcDto.load_from_file(circuit_id_filepath)
            if not isinstance(circuit_dto.component_requirements, ComponentRequirements):
                # due to mypy checker
                raise TypeError("Loaded component requirements have wrong type.")

            if not isinstance(circuit_dto.component_requirements.capacitor_requirements[0], CapacitorRequirements):
                # due to mypy checker
                raise TypeError("Loaded capacitor requirements have wrong type.")

            # Add capacitor requirements to the list
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
            circuit_dto = d_sets.HandleSbcDto.load_from_file(circuit_id_filepath)
            if not isinstance(circuit_dto.component_requirements, ComponentRequirements):
                # due to mypy checker
                raise TypeError("Loaded component requirements have wrong type.")

            if not isinstance(circuit_dto.component_requirements.inductor_requirements[0], InductorRequirements):
                # due to mypy checker
                raise TypeError("Loaded capacitor requirements have wrong type.")

            # Add inductor requirements to the list
            inductor_requirements_list = inductor_requirements_list + circuit_dto.component_requirements.inductor_requirements

        return inductor_requirements_list

    def get_transformer_requirements(self) -> list[TransformerRequirements]:
        """Get the transformer requirements.

        :return: Transformer Requirements
        :rtype: TransformerRequirements
        """
        # Variable declaration
        transformer_requirements_list: list[TransformerRequirements]

        # Dummy access with self to satisfy ruff
        if self._sbc_config is None:
            transformer_requirements_list = []
        else:
            transformer_requirements_list = []

        return transformer_requirements_list

    @staticmethod
    def get_circuit_plot_data(act_study_data: StudyData) -> PlotData:
        """Provide the circuit data to plot.

        :param act_study_data: Information about the circuit study name and study path
        :type  act_study_data: StudyData
        :return: Plot data and legend
        :rtype: PlotData
        """
        # Load circuit study data
        df_circuit = SbcCircuitOptimization.study_to_df(act_study_data)

        # Convert from list[Series[Any]] to list[list[float]]
        circuit_x_values_list: list[list[float]] = [df_circuit["values_0"].to_list()]
        circuit_y_values_list: list[list[float]] = [df_circuit["values_1"].to_list()]

        # Extract circuit plot data from data frame
        circuit_plot_data: PlotData = (
            PlotData(x_values_list=circuit_x_values_list, y_values_list=circuit_y_values_list,
                     color_list=[gps.colors()["black"]], alpha_list=[0.5],
                     x_label=r'$T_\mathrm{period}$ / s', y_label=r'$P_\mathrm{tran}$ / W',
                     label_list=[None], fig_name_path=act_study_data.study_name))

        return circuit_plot_data

    def get_output_power(self):
        """Get the sampling data output power.

        :return: Transformer requirements
        :rtype: TransformerRequirements
        """
        pass

    def get_weights(self):
        """Get the sampling data weights.

        :return: Transformer requirements
        :rtype: TransformerRequirements
        """
        pass

    @staticmethod
    def get_number_of_required_capacitors() -> int:
        """Get the number of  required capacitors.

        :return: Number of capacitors required by the actual topology
        :rtype: int
        """
        return SbcCircuitOptimization._number_of_required_capacitors

    @staticmethod
    def get_number_of_required_inductors() -> int:
        """Get the number of  required inductors.

        :return: Number of inductors required by the actual topology
        :rtype: int
        """
        return SbcCircuitOptimization._number_of_required_inductors

    @staticmethod
    def get_number_of_required_transformers() -> int:
        """Get the number of  required transformers.

        :return: Number of transformers required by the actual topology
        :rtype: int
        """
        return SbcCircuitOptimization._number_of_required_transformers

    def generate_result_dtos(self, summary_data: StudyData, capacitor_selection_data_list: list[CapacitorConfiguration],
                             inductor_configuration_list: list[InductorConfiguration],
                             transformer_configuration_list: list[TransformerConfiguration],
                             df: pd.DataFrame, is_pre_summary: bool = True) -> None:
        """
        Generate the result dtos from a given (filtered) result dataframe.

        :param summary_data: Summary Data
        :type summary_data: StudyData
        :param capacitor_selection_data_list: List of capacitor selection data
        :type  capacitor_selection_data_list: list[CapacitorConfiguration]
        :param inductor_configuration_list: List of inductor study data
        :type  inductor_configuration_list: list[InductorConfiguration]
        :param transformer_configuration_list: List of transformer study data
        :type  transformer_configuration_list: list[TransformerConfiguration]
        :param df: dataframe to take the results from
        :type df: pd.DataFrame
        :param is_pre_summary: True for pre-summary, False for summary
        :type is_pre_summary: bool
        """
        """
        Generate the result dtos from a given (filtered) result dataframe.

        :param summary_data: Summary Data
        :type summary_data: StudyData
        :param capacitor_selection_data_list: List of capacitor selection data
        :type capacitor_selection_data_list: list[CapacitorConfiguration]
        :param inductor_configuration_list: List of inductor study data
        :type inductor_configuration_list: list[InductorConfiguration]
        :param transformer_study_data: List of transformer study data
        :type transformer_study_data: list[TransformerConfiguration]
        :param df: dataframe to take the results from
        :type df: pd.DataFrame
        :param is_pre_summary: True for pre-summary, False for summary
        :type is_pre_summary: bool
        """
        if is_pre_summary:
            # pre summary using reluctance model results (inductive components)
            inductor_result_directory = CIRCUIT_INDUCTOR_RELUCTANCE_LOSSES_FOLDER
            transformer_result_directory = CIRCUIT_TRANSFORMER_RELUCTANCE_LOSSES_FOLDER
        else:
            # final summary using FEM results (inductive components)
            inductor_result_directory = CIRCUIT_INDUCTOR_FEM_LOSSES_FOLDER
            transformer_result_directory = CIRCUIT_TRANSFORMER_FEM_LOSSES_FOLDER

        for _, row in df.iterrows():
            circuit_id = row['circuit_id']
            inductor_id = row['inductor_id_0']
            inductor_study_name = row['inductor_study_name_0']
            capacitor_1_id = row['capacitor_id_0']
            capacitor_1_study_name = row['capacitor_study_name_0']
            capacitor_2_id = row['capacitor_id_1']
            capacitor_2_study_name = row['capacitor_study_name_1']

            # load circuit DTO
            circuit_id_filepath = os.path.join(self.filter_data.filtered_list_pathname, f"{circuit_id}.pkl")
            with open(circuit_id_filepath, 'rb') as pickle_file_data:
                circuit_dto: s_dtos.SbcCircuitDTO = pickle.load(pickle_file_data)

            # load inductor DTO
            study_data = inductor_configuration_list[0].study_data
            inductor_study_results_filepath = os.path.join(study_data.optimization_directory, circuit_id,
                                                           inductor_study_name,
                                                           inductor_result_directory)

            inductor_id_filepath = os.path.join(inductor_study_results_filepath, f"{inductor_id}.pkl")
            with open(inductor_id_filepath, 'rb') as pickle_file_data:
                inductor_results = pickle.load(pickle_file_data)

            circuit_dto.inductor_results = inductor_results

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
            circuit_dto.capacitor_1_results = capacitor_1_results
            circuit_dto.capacitor_2_results = capacitor_2_results

            results_folder = os.path.join(summary_data.optimization_directory, SUMMARY_COMBINATION_FOLDER)
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)

            d_sets.HandleSbcDto.save(circuit_dto,
                                     f"c{circuit_id}_i{inductor_id}_cap1_{capacitor_1_id}_cap2_{capacitor_2_id}",
                                     directory=results_folder, timestamp=False)

    def add_time_domain_simulations(self) -> None:
        """Add time domain simulations to the existing circuit DTOs."""
        pass

    @staticmethod
    def plot_compare_waveforms(dto_directory: str, output_filepath: str) -> None:
        """
        Compare calculated waveforms with simulated waveforms (GeckoCIRCUITS).

        :param dto_directory: Folder of circuit DTOs to read the values from
        :type dto_directory: str
        :param output_filepath: filepath of output plot
        :type output_filepath: str
        """
        print(dto_directory)
        print(output_filepath)

    @staticmethod
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
    def get_inductor_conf_dict(component_id: int = 0) -> dict:
        """
        Provide dictionary for inductor default configuration.

        :param component_id: component id
        :type component_id: int
        :return: Dictionary of inductor default configuration
        :rtype:  dict
        """
        # Variable declaration
        inductor_conf_dict: dict = {}
        material_data_sources: dict = {}

        # Enter configuration values
        material_data_sources["permeability_datasource"] = "MagNet"
        material_data_sources["permittivity_datasource"] = "LEA_MTB"
        # Add to return dictionary
        inductor_conf_dict["material_data_sources"] = material_data_sources

        return inductor_conf_dict

    @staticmethod
    def get_template(template_type: TemplateType) -> tuple[bool, str]:
        """
        Provide the template source for default configuration of the topology.

        :param template_type: Type of the template (circuit or general)
        :type  template_type: TemplateType
        :return: validity indicator and template source data
        :rtype:  tuple[bool, str]
        """
        # Variable declaration
        template_file_path: str = ""
        template_source: str = ""
        is_valid: bool = True

        # Check for template type
        if template_type is TemplateType.circuit:
            template_file_path = str(os.path.join(PCDT_ROOT, SBC_TOPOLOGY_FOLDER, TOMPL_TEMPLATE_FOLDER_NAME, "Sbc"+CIRCUIT_CONF_TPL))
        elif template_type is TemplateType.general:
            template_file_path = str(os.path.join(PCDT_ROOT, SBC_TOPOLOGY_FOLDER, TOMPL_TEMPLATE_FOLDER_NAME, "Sbc"+GENERAL_CONF_TPL))

        # Load template
        # Check if template file is available
        if os.path.isfile(template_file_path):
            with open(template_file_path, "r", encoding="utf-8") as f:
                template_source = f.read()
        else:
            is_valid = False
            logger.error(f"Data corruption: Missing template file {template_file_path}")

        return is_valid, template_source
