"""Unit tests for class mainctl."""

# python libraries
import os
import stat
import sys
import shutil
import tempfile
import logging
import threading
import time
import datetime
import copy
from typing import Any
from enum import Enum

# 3rd party libraries
import pytest
from _pytest.logging import LogCaptureFixture
from _pytest.capture import CaptureFixture
import zipfile
import json


# own libraries
import pcdt
from pcdt.topology.dab.dab_circuit_topology import DabCircuitOptimization as CircuitOptimization
from pcdt.components.summary_processing import SummaryProcessing
import pcdt.toml_checker as tc
from pcdt import server_ctl_dtos as srv_ctl_dtos
from pcdt.server_ctl_dtos import RunTimeMeasurement as RunTime
from pcdt.mainctl import MainCtl
import pcdt.datasets_dtos as d_dtos
from pcdt.circuit_enums import CalcModeEnum

# Enable logger
pytestlogger = logging.getLogger(__name__)

class TestCase(Enum):
    """Enum of test types."""

    # Valid test case
    LowerBoundary = 0           # Test value at lower boundary
    UpperBoundary = 1           # Test value at lower boundary
    InBetween = 2               # Test value in between
    # Failure test case
    BoundaryInconsistent = 3    # Test when minimum > maximum ( Only for minimum maximum list)
    TooLessEntries = 4          # Test when the list has got too less entries ( Only for minimum maximum list)
    TooMuchEntries = 5          # Test when the list has got too much entries ( Only for minimum maximum list)
    ExceedLowerLimit = 6        # Test when the lower limit is exceeded
    ExceedUpperLimit = 7        # Test when the lower limit is exceeded
    # Special test of additional point list: Valid test case
    APt_NumberOfEntries = 8  # Test when Number of entries in additional point list is inconsistent
    # Special test of additional point list: Valid test case
    SpecialTestNumberOfEntries = 8  # Test when Number of entries in additional point list is inconsistent


# FlowControl base parameter set
test_FlowControl_base: tc.FlowControl = tc.FlowControl(
    general=tc.General(project_directory="Hallo",
                       topology="dab"),
    breakpoints=tc.Breakpoints(circuit_pareto="no",
                               circuit_filtered="no",
                               capacitor="no",
                               inductor="no",
                               transformer="no",
                               heat_sink="no",
                               pre_summary="no",
                               summary="no"),
    conditional_breakpoints=tc.CondBreakpoints(
        circuit=1,
        inductor=2,
        transformer=3,
        heat_sink=1),
    circuit=tc.Circuit(number_of_trials=5,
                       calculation_mode="continue",
                       subdirectory="dummy"),
    capacitor=tc.Capacitor(
        calculation_modes=["new", "new"],
        subdirectory="dummy"),
    inductor=tc.Inductor(numbers_of_trials=[10, 20],
                         calculation_modes=["continue", "continue"],
                         subdirectory="dummy"),
    transformer=tc.Transformer(numbers_of_trials=[30, 40],
                               calculation_modes=["continue", "continue"],
                               subdirectory="dummy"),
    heat_sink=tc.HeatSink(number_of_trials=50,
                          calculation_mode="continue",
                          subdirectory="dummy"),
    pre_summary=tc.PreSummary(calculation_mode="new",
                              subdirectory="dummy"),
    summary=tc.Summary(calculation_mode="new",
                       subdirectory="dummy"),
    data_generation=tc.DataGeneration(calculation_mode_visualization="new",
                                      calculation_mode_manufacture="new",
                                      subdirectory="dummy"),

    configuration_data_files=tc.ConfigurationDataFiles(
        topology_files=["dummy1", "dummy2"],
        capacitor_configuration_files=["dummy1", "dummy2"],
        inductor_configuration_files=["dummy", "dummy2"],
        transformer_configuration_files=["dummy", "dummy2"],
        heat_sink_configuration_file="dummy",
        summary_configuration_file="dummy")
)

#########################################################################################################
# test of load_toml_file
#########################################################################################################

# Test results
exp_config = {'integer_value': {'test_value': 10}, 'string': {'test_string': 'Hello'}}

# Invalid toml file
test_invalid_toml = """
string
test_string = "Hello"

[integer_value]
test_value = 10
"""
# Valid toml file
test_valid_toml = """
[string]
test_string = "Hello"

[integer_value]
test_value = 10
"""

# test parameter list
@pytest.mark.parametrize("test_toml_data, is_path_existing, is_exp_result_valid, exp_toml_object, exp_message_id", [
    # --invalid inputs----
    # Path does not exist
    (test_valid_toml, False, False, {}, 1),
    # Input file does not exist
    ("", True, False, {}, 2),
    # Input file, with wrong format
    (test_invalid_toml, True, False, {}, 3),
    # --valid inputs----
    # input file, with valid format
    (test_valid_toml, True, True, exp_config, 0)
])
# Unit test function
def test_load_toml_file(caplog: LogCaptureFixture, test_toml_data: str, is_path_existing: bool,
                        is_exp_result_valid: bool, exp_toml_object: dict, exp_message_id: int) -> None:
    """Test the method load_toml_file.

    :param caplog: class instance for logger data
    :type  caplog: LogCaptureFixture
    :param test_toml_data: Test data of the toml file
    :type  test_toml_data: str
    :param is_path_existing: Flag to indicate, if the path exists for the test
    :type  is_path_existing: bool
    :param is_exp_result_valid: Flag to indicate, if the expected result is valid
    :type  is_exp_result_valid: bool
    :param exp_toml_object: Configuration data load from the toml-file
    :type  exp_toml_object: dict
    :param exp_message_id: List index of the expected message
    :type  exp_message_id: int
    """
    # Variable declaration
    # Create the instance
    test_pcdt: MainCtl = MainCtl()
    invalid_path_name = ""

    with tempfile.TemporaryDirectory() as tmpdir:
        valid_filepath = os.path.join(tmpdir, "config.toml")
        # Check for invalid path flag
        with open(valid_filepath, "w") as f:
            f.write(test_toml_data)
        invalid_filepath = os.path.join(tmpdir, "not_exist.toml")
        # Check for empty content
        if not test_toml_data == "":
            if not is_path_existing:
                invalid_path_name = os.path.join(tmpdir, "invalid_path")
                filepath = os.path.join(invalid_path_name, "config.toml")
            else:
                filepath = valid_filepath
        else:
            filepath = invalid_filepath

        # Perform the test
        with caplog.at_level(logging.INFO):
            is_result_valid, toml_object = test_pcdt.load_toml_file(filepath)

            # Expected messages
            expected_message = ["",
                                f"Path {invalid_path_name} does not exists!",
                                f"File {filepath} does not exists!",
                                "toml-file is not conform to toml-format:\nExpected '=' after a key in a key/value pair (at line 2, column 7)"]

            # Evaluate the result
            if len(caplog.records) > 0:
                assert caplog.records[0].message == expected_message[exp_message_id]
            else:
                assert "" == expected_message[exp_message_id]

            assert is_result_valid == is_exp_result_valid
            assert toml_object == exp_toml_object


#########################################################################################################
# test of load_generate_logging_config
#########################################################################################################

# Valid toml file
test_valid_logger_toml = """
[loggers]
keys=root,pcdt,transistordatabase,femmt,hct

[handlers]
keys=console

[formatters]
keys=simple

[logger_root]
level=INFO
handlers=console
qualname=

[logger_pcdt]
level=INFO
handlers=console
qualname=pcdt
propagate=1

[logger_transistordatabase]
level=INFO
handlers=console
qualname=transistordatabase
propagate=0

[logger_femmt]
level=INFO
handlers=console
qualname=femmt
propagate=0

[logger_hct]
level=INFO
handlers=console
qualname=hct
propagate=0

[handler_console]
class=StreamHandler
level=NOTSET
formatter=simple
args=(sys.stdout,)

[formatter_simple]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
datefmt=%Y-%m-%d %H:%M:%S
"""

# Invalid toml file
test_invalid_logger_toml = """
[nix]
keys=root,pcdt,transistordatabase,femmt,hct

[nor]
keys=console

[formatters]
keys=simple

[logger_bad_key]
level=INFO
handlers=console
"""

# test parameter list
@pytest.mark.parametrize("test_toml_data, is_path_existing, is_file_existing, is_logger_to_restore, exp_message_id", [
    # --invalid input parameter----
    # Path does not exist
    (test_valid_logger_toml, False, False, False, 1),
    # Input file, with wrong format
    (test_invalid_logger_toml, True, True, False, 3),
    # Input file does not exist
    (test_valid_logger_toml, True, False, False, 2),
    # --valid input parameter----
    # Valid input file exists
    (test_valid_logger_toml, True, True, True, 0)
])
# Unit test function
def test_load_generate_logging_config(caplog: LogCaptureFixture, test_toml_data: str, is_path_existing: bool, is_file_existing: bool,
                                      is_logger_to_restore: bool, exp_message_id: int) -> None:
    """Test method load_generate_logging_config(logging_config_file: str) -> None: according values.

    :param caplog: class instance for logger data
    :type  caplog: LogCaptureFixture
    :param test_toml_data: Test data to load by the method
    :type  test_toml_data: str
    :param is_path_existing: Flag to indicate, if the path exists for the test
    :type  is_path_existing: bool
    :param is_file_existing: Flag to indicate, if the file exists for the test
    :type  is_file_existing: bool
    :param is_logger_to_restore: Flag to indicate, if the logger needs to be restored after test
    :type  is_logger_to_restore: bool
    :param exp_message_id: List index of the expected message
    :type  exp_message_id: int
    """
    # Variable declaration
    # Create the instance
    test_pcdt: MainCtl = MainCtl()

    # Prepare the setup
    # Create path
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create the file-path name
        valid_filepath = os.path.join(tmpdir, "logger_config.toml")
        # Check for valid file flag
        if is_file_existing:
            # Store the logger configuration data
            with open(valid_filepath, "w") as f:
                f.write(test_toml_data)
            # Check for valid path flag
        if is_path_existing:
            filepath = valid_filepath
        else:
            filepath = os.path.join(tmpdir, "not_existing_folder", "not_existing_file.toml")

        # Perform the test
        with caplog.at_level(logging.INFO):
            if is_logger_to_restore:
                test_pcdt.load_generate_logging_config(filepath)
                caplog.set_level(logging.INFO)
            else:
                test_pcdt.load_generate_logging_config(filepath)

            # Create wrong path string
            wrong_path = os.path.join(tmpdir, "not_existing_folder")
            # Expected messages
            expected_message = ["",
                                f"Path {wrong_path} does not exists!",
                                "Generate a new logging.conf file.",
                                f"Logging configuration file {valid_filepath} is inconsistent."]

            if len(caplog.records) > 0:
                assert caplog.records[0].message == expected_message[exp_message_id]
            else:
                assert "" == expected_message[exp_message_id]


#########################################################################################################
# test of delete_study_content
#########################################################################################################

# Valid toml file
test_ascii_file_data = "This is a test file"

# test parameter list
@pytest.mark.parametrize("is_path_existing,  is_folder_list_empty, is_folder_empty, is_delete_all_flag, exp_message_id", [
    # --invalid input parameter----
    # Path does not exist
    (False, False, False, False, 1),
    (False, False, False, True, 1),
    # --valid input parameter----
    # Empty folder or empty folder list
    (True, False, True, True, 2),
    (True, True, True, False, 3),
    (True, True, False, False, 3),
    # Valid input folder contains data
    (True, False, True, False, 0),
    (True, True, False, True, 0),
    (True, True, False, True, 0)
])
# Unit test function
def test_delete_study_content(caplog: LogCaptureFixture, is_path_existing: bool, is_folder_list_empty: bool,
                              is_folder_empty: bool, is_delete_all_flag: bool, exp_message_id: int) -> None:
    """Test method load_generate_logging_config(logging_config_file: str) -> None: according values.

    :param caplog: class instance for logger data
    :type  caplog: LogCaptureFixture
    :param is_path_existing: Flag to indicate, if the path exists for the test
    :type  is_path_existing: bool
    :param is_folder_list_empty: Flag to indicate, if the folder list is empty
    :type  is_folder_list_empty: bool
    :param is_folder_empty: Flag to indicate, if the folder is empty
    :type  is_folder_empty: bool
    :param is_delete_all_flag: Flag to indicate, if the complete optimization folder will be deleted
    :type  is_delete_all_flag: bool
    :param exp_message_id: List index of the expected message
    :type  exp_message_id: int
    """
    # Variable declaration
    # Create the instance
    test_pcdt: MainCtl = MainCtl()

    # Initialize sub folder list
    sub_folder_list = ["sub_folder1", "sub_folder2", "sub_folder3"]

    # Prepare the setup
    # Create path
    with tempfile.TemporaryDirectory() as tmpdir:
        # Check for empty folder in mode is_all_flag
        if not (is_folder_empty and is_delete_all_flag):
            # Create filenames
            study_asc_name = "study_data.text"
            study_bin_name = "study_data.sqlite3"
            # Create the pathname
            valid_asc_source = os.path.join(tmpdir, study_asc_name)
            valid_bin_source = os.path.join(tmpdir, study_bin_name)
            # Create 2 files
            # text file
            with open(valid_asc_source, "w") as f:
                f.write(test_ascii_file_data)
            # Binary file (1 kilo byte random data)
            with open(valid_bin_source, "wb") as f:
                f.write(os.urandom(1024))
            # Create sub folders (3 levels) and store files within each level
            for sub_folder in sub_folder_list:
                sub_dir = os.path.join(tmpdir, sub_folder)
                os.makedirs(sub_dir, exist_ok=True)
                if not is_folder_empty:
                    for level in ["level1", "level2"]:
                        actual_dir = os.path.join(sub_dir, level)
                        os.makedirs(actual_dir, exist_ok=True)
                        # Copy test file
                        target_file = os.path.join(actual_dir, study_asc_name)
                        shutil.copy(valid_asc_source, target_file)
                        # Copy binary file
                        target_file = os.path.join(actual_dir, study_bin_name)
                        shutil.copy(valid_bin_source, target_file)

        # Assign pathname
        if is_path_existing:
            path_name = tmpdir
        else:
            path_name = os.path.join(tmpdir, "not_existing_folder")

        # Assign folder list
        if is_folder_list_empty:
            used_sub_folder_list = []
        else:
            used_sub_folder_list = sub_folder_list

        # Perform the test
        with caplog.at_level(logging.INFO):
            test_pcdt.delete_study_content(is_delete_all_flag, path_name, "level1", used_sub_folder_list)
            # Expected messages
            expected_message = ["",
                                f"Path {path_name} does not exists!",
                                f"Folder {path_name} is empty!",
                                "sub folder list is empty. Nothing is deleted!"]

            if len(caplog.records) > 0:
                assert caplog.records[0].message == expected_message[exp_message_id]
            else:
                assert "" == expected_message[exp_message_id]

            # Check, if the folders are deleted, if the file exists
            if is_path_existing:
                if is_delete_all_flag:
                    assert len(os.listdir(path_name)) == 0
                elif (is_folder_list_empty or is_folder_empty) is False:
                    for sub_folder in sub_folder_list:
                        sub_dir = os.path.join(tmpdir, sub_folder, "level1")
                        assert len(os.listdir(sub_dir)) == 0

#########################################################################################################
# test of check_study_data
#########################################################################################################

# test parameter list
@pytest.mark.parametrize("is_path_existing, is_file_existing, file_name, exp_message_id, exp_result", [
    # --invalid input parameter----
    # Path does not exist
    (False, False, "study_data.sqlite3", 1, False),
    # File does not exists
    (True, False, "study_data.sqlite3", 2, False),
    # Filename is wrong
    (True, True, "wrong_study_data.sqlite3", 2, False),
    # Filename extension is wrong
    (True, True, "study_data.text", 2, False),
    # --valid input parameter----
    # Input file does not exist
    (True, True, "study_data.sqlite3", 0, True),
])
# Unit test function
def test_check_study_data(caplog: LogCaptureFixture, is_path_existing: bool, is_file_existing: bool, file_name: str,
                          exp_message_id: int, exp_result: bool) -> None:
    """Test method load_generate_logging_config(logging_config_file: str) -> None: according values.

    :param caplog: class instance for logger data
    :type  caplog: LogCaptureFixture
    :param is_path_existing: Flag to indicate, if the path exists for the test
    :type  is_path_existing: bool
    :param is_file_existing: Flag to indicate, if the file exists for the test
    :type  is_file_existing: bool
    :param file_name: Name of the file, which shall be used
    :type  file_name: str
    :param exp_message_id: List index of the expected message
    :type  exp_message_id: int
    :param exp_result: Expected result of the method
    :type  exp_result: bool
    """
    # Variable declaration
    target_study_name = "study_data"

    # Prepare the setup
    # Create path
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create the file pathname
        valid_file_source = os.path.join(tmpdir, file_name)
        # Check, if file exists
        if is_file_existing:
            with open(valid_file_source, "wb") as f:
                f.write(os.urandom(1024))
        # Assign pathname
        if is_path_existing:
            path_name = tmpdir
        else:
            path_name = os.path.join(tmpdir, "not_existing_folder")

        path_file_name = os.path.join(path_name, f"{target_study_name}.sqlite3")

        # Perform the test
        with caplog.at_level(logging.INFO):
            # Create the instance (Later to move this test function to a separate unit test file
            test_d_dtos: d_dtos.StudyData = d_dtos.StudyData(target_study_name, path_name, CalcModeEnum.new_mode)
            target_study_name = "study_data"

            # Check test case for is_path_existing == False
            if not is_path_existing:
                # Delete not_existing_folder due to created by d_dtos.StudyData
                os.removedirs(path_name)

            test_result = test_d_dtos.check_study_data(path_name, target_study_name)
            # Expected messages
            expected_message = ["",
                                f"Path {path_name} does not exists!",
                                f"File {path_file_name} does not exists!"]

            if len(caplog.records) > 0:
                assert caplog.records[0].message == expected_message[exp_message_id]
            else:
                assert "" == expected_message[exp_message_id]

            # Check the result of the method
            assert test_result == exp_result

#########################################################################################################
# test of get_number_of_pkl_files
#########################################################################################################

# test parameter list
@pytest.mark.parametrize("is_path_existing, number_of_pkl_files, prefix, exp_message_id", [
    # --invalid input parameter----
    # Path does not exist
    (False, 0, "", 1),
    # --valid input parameter----
    # No pkl-file exists
    (True, 0, "", 0),
    # 10 pkl files without prefix
    (True, 10, "", 0),
    # 20 pkl files with prefix
    (True, 10, "C_", 0),

])
# Unit test function
def test_get_number_of_pkl_files(caplog: LogCaptureFixture, is_path_existing: bool, number_of_pkl_files: int, prefix: str, exp_message_id: int) -> None:
    """Test method load_generate_logging_config.

    :param caplog: class instance for logger data
    :type  caplog: LogCaptureFixture
    :param is_path_existing: Flag to indicate, if the path exists for the test
    :type  is_path_existing: bool
    :param number_of_pkl_files: Number of pkl-files
    :type  number_of_pkl_files: int
    :param prefix: File prefix
    :type  prefix: str
    :param exp_message_id: List index of the expected message
    :type  exp_message_id: int
    """
    # Variable declaration
    # Create the instance
    test_pcdt: MainCtl = MainCtl()
    target_study_name = "study_data"

    # Prepare the setup
    # Create path
    with tempfile.TemporaryDirectory() as tmpdir:
        # Check, if path is valid and minimum one file is requested
        if is_path_existing:
            path_name = tmpdir
            if number_of_pkl_files > 0:
                # Create requested number of files
                for file_number in range(number_of_pkl_files):
                    # Create the file pathname
                    file_path_name = os.path.join(path_name, f"{prefix}{file_number}.pkl")
                    # Create and store the file
                    with open(file_path_name, "wb") as f:
                        f.write(os.urandom(1024))
        else:
            path_name = os.path.join(tmpdir, "not_existing_folder")

        # Perform the test
        with caplog.at_level(logging.INFO):
            test_result = test_pcdt.get_number_of_pkl_files(path_name)
            # Expected messages
            expected_message = ["",
                                f"Path {path_name} does not exists!"]

            if len(caplog.records) > 0:
                assert caplog.records[0].message == expected_message[exp_message_id]
            else:
                assert "" == expected_message[exp_message_id]

            # Check the result of the method
            assert test_result == number_of_pkl_files

#########################################################################################################
# test of check_breakpoint
#########################################################################################################
# check_breakpoint(self, break_point_key: str, info: str)

# test parameter list
@pytest.mark.parametrize("keyword, message, keyboard_input, expected_message, is_program_exit, is_wait_for_input", [
    # --invalid use case----
    # Invalid keyword
    ("invalid_keyword", "Message not to return", "", "", False, False),
    # --valid use cases----
    # Valid keyword no - Breakpoint will not wait for an input
    ("no", "message string (no)", "", "", False, False),
    # Valid keyword stop - Breakpoint leads to terminate program
    ("stop", "message string (stop)", "", "Program stops cause by breakpoint at: 'message string (stop)'!", True, False),
    # Valid keyword pause - Breakpoint will leads to pause and continue with valid keyboard input
    ("pause", "message string (pause-c)", "c",
     "Active breakpoint at: 'message string (pause-c)'!\n\n'C'=continue, 'S'=stop the program. Please enter your choice", False, True),
    # Valid keyword pause - Breakpoint leads to pause and continue with valid keyboard input
    ("pause", "message string (pause-C)", "C",
     "Active breakpoint at: 'message string (pause-C)'!\n\n'C'=continue, 'S'=stop the program. Please enter your choice", False, True),
    # Valid keyword pause - Breakpoint leads to pause and terminated with valid keyboard input
    ("pause", "message string (pause-s)", "s",
     "Active breakpoint at: 'message string (pause-s)'!\n\n'C'=continue, 'S'=stop the program. Please enter your choice\nUser stops the program!", True, True),
    # Valid keyword pause - Breakpoint leads to pause and terminated with valid keyboard input
    ("pause", "message string (pause-S)", "S",
     "Active breakpoint at: 'message string (pause-S)'!\n\n'C'=continue, 'S'=stop the program. Please enter your choice\nUser stops the program!", True, True)
])
# Unit test function
def test_check_breakpoint(capsys: CaptureFixture[str], keyword: str, message: str, keyboard_input: str,
                          expected_message: str, is_program_exit: bool, is_wait_for_input: bool) -> None:
    """Test method load_generate_logging_config(logging_config_file: str) -> None: according values.

    :param capsys: class instance for print data
    :type  capsys: CaptureFixture
    :param keyword: keyword to control the method under test
    :type  keyword: str
    :param message: Input message, which is used to create the output message
    :type  message: str
    :param keyboard_input: String, which simulates the keyboard input
    :type  keyboard_input: str
    :param expected_message: expected message string
    :type  expected_message: str
    :param is_program_exit: Flag to indicate, if the program terminates
    :type  is_program_exit: bool
    :param is_wait_for_input: Flag to indicate, if the function waits for keyboard input
    :type  is_wait_for_input: bool
    """
    # Variable declaration
    test_thread: threading.Thread
    is_exit: bool = False
    exit_code: int = -1

    # Create the instance
    test_pcdt: MainCtl = MainCtl()

    def test_thread_container():
        nonlocal is_exit
        nonlocal exit_code
        try:
            test_pcdt.check_breakpoint(keyword, message)
        except SystemExit as e:
            is_exit = True
            exit_code = e.code

    # Setup the thread
    test_thread = threading.Thread(target=test_thread_container)
    test_thread.daemon = True

    # Perform the test
    test_thread.start()
    time.sleep(0.2)

    # Check if the method waits for input
    if is_wait_for_input:
        assert test_thread.is_alive()
        test_pcdt._key_input_string = keyboard_input
        test_thread.join(timeout=2)

    # Read out the print log
    print_data = capsys.readouterr().out
    # Test on finish thread
    assert not test_thread.is_alive()
    # Check if thread is finished
    if not test_thread.is_alive():
        assert print_data.strip() == expected_message

    # Check if program termination is requested
    if is_program_exit:
        assert is_exit
        assert exit_code is None
    else:
        assert not is_exit
        assert exit_code == -1

#########################################################################################################
# test of generate_zip_archive
#########################################################################################################

# def generate_zip_archive(self, toml_prog_flow: tc.FlowControl) -> None:

# test parameter list
@pytest.mark.parametrize("is_path_existing, exp_message_id", [
    # --invalid input parameter----
    # Path does not exist
    (False, 1),
    # --valid input parameter----
    # Path exists
    (True, 2)
])
# Unit test function
def test_generate_zip_archive(caplog: LogCaptureFixture, is_path_existing: bool, exp_message_id: int) -> None:
    """Test method load_generate_logging_config(logging_config_file: str) -> None: according values.

    :param caplog: class instance for logger data
    :type  caplog: LogCaptureFixture
    :param is_path_existing: Flag to indicate, if the path exists for the test
    :type  is_path_existing: bool
    :param exp_message_id: List index of the expected message
    :type  exp_message_id: int
    """
    # Variable declaration
    # Folder to exclude from zip
    exclude_level: str = "00_femmt_simulation"
    # Create the instance
    test_pcdt: MainCtl = MainCtl()

    # Constant values of FlowControl
    test_parameter_2: tc.FlowControl = copy.deepcopy(test_FlowControl_base)

    # Prepare the setup
    # Create path
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create base path
        base_path = os.path.join(tmpdir, "base_path")
        os.makedirs(base_path, exist_ok=True)
        # Create filenames
        study_asc_name = "study_data.text"
        study_bin_name = "study_data.sqlite3"
        # Create the pathname
        valid_asc_source = os.path.join(base_path, study_asc_name)
        valid_bin_source = os.path.join(base_path, study_bin_name)
        # Create 2 files
        # text file
        with open(valid_asc_source, "w") as f:
            f.write(test_ascii_file_data)
        # Binary file (1 kilo byte random data)
        with open(valid_bin_source, "wb") as f:
            f.write(os.urandom(1024))
        # Create sub folders (3 levels) and store files within each level
        actual_dir = base_path
        for level in ["level1", "level2", exclude_level]:
            actual_dir = os.path.join(actual_dir, level)
            os.makedirs(actual_dir, exist_ok=True)
            # Copy test file
            target_file = os.path.join(actual_dir, study_asc_name)
            shutil.copy(valid_asc_source, target_file)
            # Copy binary file
            target_file = os.path.join(actual_dir, study_bin_name)
            shutil.copy(valid_bin_source, target_file)
            os.makedirs(actual_dir+"a", exist_ok=True)
            # Copy test file
            target_file = os.path.join(actual_dir+"a", study_asc_name)
            shutil.copy(valid_asc_source, target_file)
            # Copy binary file
            target_file = os.path.join(actual_dir+"a", study_bin_name)
            shutil.copy(valid_bin_source, target_file)

        # Assign pathname
        if is_path_existing:
            path_name = base_path
        else:
            path_name = os.path.join(base_path, "not_existing_folder")

        # Overtake constant values of FlowControl and update project directory
        test_flow_control: tc.FlowControl = copy.deepcopy(test_FlowControl_base)
        test_flow_control.general.project_directory = path_name

        # Test procedure
        with caplog.at_level(logging.INFO):
            # Create approximated file name before test
            zip_file_name_before = f'{path_name}_archived_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")}.zip'
            # Perform the test
            test_pcdt.generate_zip_archive(test_flow_control)
            # Create approximated file name after test
            zip_file_name_after = f'{path_name}_archived_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")}.zip'
            # Get common string part
            common_zip_file_name_part = os.path.commonprefix([zip_file_name_before, zip_file_name_after])

            # Expected messages
            expected_message = ["",
                                f"Path {path_name} does not exists!",
                                f"Zip archive created:{common_zip_file_name_part}"]

            # Check, if the folders are deleted, if the file exists
            if is_path_existing:
                is_file_found: bool = False
                # Open the first zip-file (only on exists)
                for zip_file_path in os.listdir(tmpdir):
                    # Check name
                    if zip_file_path.endswith('.zip'):
                        zip_file_path = os.path.join(tmpdir, zip_file_path)
                        is_file_found = True
                        break

                # Check if file is found
                assert is_file_found

                # Read the content of the zip-archive
                with zipfile.ZipFile(zip_file_path, 'r') as zf:
                    zip_contents = zf.namelist()

                # Restore actual path to relative path
                actual_dir = ""

                # Check content on upper level
                assert study_asc_name in zip_contents
                assert study_bin_name in zip_contents
                # Check content (folder and files)
                for level in ["level1", "level2"]:
                    actual_dir = os.path.join(actual_dir, level)
                    # Check text file
                    target_file = os.path.join(actual_dir, study_asc_name)
                    assert target_file in zip_contents
                    # Check binary file
                    target_file = os.path.join(actual_dir, study_bin_name)
                    assert target_file in zip_contents
                    # Check text file in additional folder
                    target_file = os.path.join(actual_dir + "a", study_asc_name)
                    assert target_file in zip_contents
                    # Check binary file in additional folder
                    target_file = os.path.join(actual_dir + "a", study_bin_name)
                    assert target_file in zip_contents

                # Check if the excluded folder is really excluded
                actual_dir = os.path.join(actual_dir, exclude_level)
                target_file = os.path.join(actual_dir, study_asc_name)
                target_file_bin = os.path.join(actual_dir, study_bin_name)
                # Loop over content list
                for name in zip_contents:
                    # Check text file
                    assert target_file not in name
                    # Check binary file
                    assert target_file_bin not in name

                # Check if text file in additional folder still exists
                target_file = os.path.join(actual_dir + "a", study_asc_name)
                assert target_file in zip_contents
                # Check binary file in additional folder
                target_file = os.path.join(actual_dir + "a", study_bin_name)
                assert target_file in zip_contents

            if len(caplog.records) > 0:
                assert caplog.records[0].message.startswith(expected_message[exp_message_id])
            else:
                assert "" == expected_message[exp_message_id]

#########################################################################################################
# test of _is_skippable
#########################################################################################################

# def _is_skippable(act_study_data: StudyData, complete_file_name: str, is_sqlite_check_enabled: bool = False,
#                   index_list: list[str] | None = None) -> tuple[bool, str]:

# test parameter list
@pytest.mark.parametrize("is_sqlite_check_enabled, is_sqlite_stored, is_index_list, is_processing_complete, exp_message_id", [
    # --parameter set which leads to an error----
    # sqlite check fails
    (True, False, True, True, 1),
    (True, False, False, True, 1),
    (True, False, True, False, 1),
    (True, False, False, False, 1),
    # --Result _is_skippable is False----
    # sqlite check is positive and processing complete condition is not fulfilled
    (True, True, True, False, 2),
    (True, True, False, False, 2),
    # --Result _is_skippable is True----
    (True, True, True, True, 0),
    (True, True, False, True, 0)
])
# Unit test function
def test__is_skippable(is_sqlite_check_enabled: bool, is_sqlite_stored: bool,
                       is_index_list: bool, is_processing_complete: bool, exp_message_id: int) -> None:
    """Test method def _is_skippable(act_study_data: StudyData, ...) -> tuple[bool, str] according values.

    :param is_sqlite_check_enabled: Flag to indicate, if sqlite test is enabled
    :type  is_sqlite_check_enabled: bool
    :param is_sqlite_stored: Flag to indicate, if the sqlite test fails
    :type  is_sqlite_stored: bool
    :param is_index_list: Flag to indicate, if a list is available
    :type  is_index_list: bool
    :param is_processing_complete: Flag to indicate, if the processing complete condition is fulfilled
    :type  is_processing_complete: bool
    :param exp_message_id: List index of the expected message
    :type  exp_message_id: int
    """
    # Variable declaration
    target_study_name = "test_study_name"
    string_test_values: list[str] = ["A", "c9", "Test123", "x_Y_z890", "6Secure_Pass01", "Alpha_Bravo_2024", "z_Z9y_Y8_x_X7", "7Long_String_Test999"]
    processing_complete_test_file: str = "processing_complete_test_file.json"
    processing_complete_path_test_file: str
    index_list: list[str] | None = None
    first_pkl_path_name: str
    json_file_list: list[str] = []

    # Prepare the setup
    # Create path
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create the study
        test_study_data: d_dtos.StudyData = d_dtos.StudyData(
            study_name=target_study_name,
            optimization_directory=os.path.join(tmpdir, "component_folder", target_study_name),
            calculation_mode=CalcModeEnum.continue_mode
        )
        # Check for index list
        if is_index_list:
            index_list = string_test_values
            # Create sqlite files for each index
            for index in index_list:
                # Create folder and add index list
                path_name = os.path.join(test_study_data.optimization_directory, index, test_study_data.study_name)
                # Check for is_processing_complete
                if is_processing_complete:
                    pkl_path_name = os.path.join(path_name, test_study_data.study_name)
                    os.makedirs(pkl_path_name, exist_ok=True)
                    # Store dummy files
                    for pkl_file in string_test_values:
                        pkl_path_file_name = os.path.join(pkl_path_name, pkl_file+".pkl")
                        with open(pkl_path_file_name, "wb") as f:
                            f.write(os.urandom(1024))
                        # Store file path  for json-list
                        json_file_list.append(pkl_path_file_name)
                else:
                    os.makedirs(path_name, exist_ok=True)

                if is_sqlite_stored:
                    file_name = os.path.join(path_name, f"{target_study_name}.sqlite3")
                    with open(file_name, "wb") as f:
                        f.write(os.urandom(1024))

            # Set the name of first element
            first_pkl_path_name = os.path.join(test_study_data.optimization_directory, index_list[0], test_study_data.study_name)

            # Check if processing complete file is store
            if is_processing_complete:
                # Create processing complete file
                processing_complete_path_test_file = os.path.join(test_study_data.optimization_directory, processing_complete_test_file)
                with open(processing_complete_path_test_file, "w", encoding="utf-8") as file_handle:
                    json.dump(json_file_list, file_handle, indent=2)

        else:
            # Check if sqlite file is to store
            if is_sqlite_stored:
                # Store the sqlite file
                file_name = os.path.join(test_study_data.optimization_directory, f"{target_study_name}.sqlite3")
                with open(file_name, "wb") as f:
                    f.write(os.urandom(1024))

            # Check for is_processing_complete
            if is_processing_complete:
                pkl_path_name = os.path.join(test_study_data.optimization_directory, test_study_data.study_name)
                os.makedirs(pkl_path_name, exist_ok=True)
                # Store dummy files
                for pkl_file in string_test_values:
                    pkl_path_file_name = os.path.join(pkl_path_name, pkl_file + ".pkl")
                    with open(pkl_path_file_name, "wb") as f:
                        f.write(os.urandom(1024))
                    # Store file path  for json-list
                    json_file_list.append(pkl_path_file_name)

                # Create processing complete file
                processing_complete_path_test_file = os.path.join(test_study_data.optimization_directory, processing_complete_test_file)
                with open(processing_complete_path_test_file, "w", encoding="utf-8") as file_handle:
                    json.dump(json_file_list, file_handle, indent=2)

            # Set the name of first element
            first_pkl_path_name = test_study_data.optimization_directory

        # Expected messages
        expected_message = ["",
                            f"Study {test_study_data.study_name} in path {first_pkl_path_name} "
                            "does not exist or file path is wrong. No sqlite3-database found!",
                            f"File {processing_complete_test_file} in path {test_study_data.optimization_directory} does not exists!\n"]

        # Perform the test
        # Check if sqlite is enabled
        if is_sqlite_check_enabled:
            # Check if sqlite test fails
            if not is_sqlite_stored:
                # Perform the test
                with pytest.raises(ValueError) as error_message:
                    is_skippable, issue_report = MainCtl._is_skippable(test_study_data, processing_complete_test_file,
                                                                       is_sqlite_check_enabled, index_list)
                assert expected_message[exp_message_id] in str(error_message.value)
            else:
                # Perform the test
                is_skippable, issue_report = MainCtl._is_skippable(test_study_data,
                                                                   processing_complete_test_file,
                                                                   is_sqlite_check_enabled, index_list)
                # Verify the result
                assert is_skippable == is_processing_complete
                assert issue_report == expected_message[exp_message_id]

#########################################################################################################
# _set_processing_complete
#########################################################################################################

# def _set_processing_complete(base_directory: str, subdirectory: str, complete_file_name: str,
#                              index_list: list[str] | None = None) -> None:

# test parameter list
@pytest.mark.parametrize("is_json_path_existing, is_pkl_path_existing, is_any_pkl_file_existing, is_list_enabled, exp_message_id", [
    # -- Parameter set which leads to an error ----
    # (False, False, False, False, 1),
    (True, False, False, False, 2),
    # (False, False, False, True, 1),
    (True, False, False, True, 2),
    # -- Parameter set which does not lead to an error ----
    # (True, True, False, False, 0),
    # (True, True, True, False, 0),
    # (True, True, False, True, 0),
    # (True, True, True, True, 0)
])
# Unit test function
def test__set_processing_complete(is_json_path_existing: bool, is_pkl_path_existing: bool,
                                  is_any_pkl_file_existing: bool, is_list_enabled: bool, exp_message_id: int) -> None:
    """Test method def _set_processing_complete(base_directory: str, subdirectory: str...) -> None: according values.

    :param is_json_path_existing: Flag to indicate, if the path for json-file exists
    :type  is_json_path_existing: bool
    :param is_pkl_path_existing: Flag to indicate, if the path for pkl-files exist
    :type  is_pkl_path_existing: bool
    :param is_any_pkl_file_existing: Flag to indicate, if the any pkl-file is stored in pkl-file path
    :type  is_any_pkl_file_existing: bool
    :param is_list_enabled: Flag to indicate, if the list is not empty
    :type  is_list_enabled: bool
    :param exp_message_id: List index of the expected message
    :type  exp_message_id: int
    """
    # Variable declaration
    string_test_values: list[str] = ["A", "c9", "Test123", "x_Y_z890", "6Secure_Pass01", "Alpha_Bravo2024", "z_Z9y_Y8_x_X7", "7Long_String_Test999"]
    processing_complete_test_file: str = "processing_complete_test_file.json"
    json_path_name: str = "test_json_path"
    subdirectory: str = "pkl_subdirectory"
    index_list: list[str] = []
    json_file_list_cmp: list[str] = []
    json_path: str
    pkl_path_name: str = ""

    # Prepare the setup
    # Create path
    with tempfile.TemporaryDirectory() as tmpdir:
        # Assemble json path
        json_path = os.path.join(tmpdir, json_path_name)
        # Check if json-file path exists
        if is_json_path_existing:
            # Create json path
            os.makedirs(json_path, exist_ok=True)
            # Check if pkl-file path exists
            if is_pkl_path_existing:
                #  Check if list mode is enabled
                if is_list_enabled:
                    # Generate pkl-files according the index list
                    index_list = string_test_values
                    # Create pkl-files for each index
                    for index in index_list:
                        # Create pkl-folder and add index list
                        pkl_path_name = os.path.join(json_path, index, subdirectory)
                        os.makedirs(pkl_path_name, exist_ok=True)
                        # Check if pkl-files are to create
                        if is_any_pkl_file_existing:
                            # Store dummy files
                            for pkl_file in string_test_values:
                                pkl_path_file_name = os.path.join(pkl_path_name, pkl_file+".pkl")
                                with open(pkl_path_file_name, "wb") as f:
                                    f.write(os.urandom(1024))
                                # Store file path  for json-list
                                json_file_list_cmp.append(pkl_path_file_name)
                else:
                    # Create pkl-folder
                    pkl_path_name = os.path.join(json_path, subdirectory)
                    os.makedirs(pkl_path_name, exist_ok=True)
                    # Store dummy files
                    for pkl_file in string_test_values:
                        pkl_path_file_name = os.path.join(pkl_path_name, pkl_file + ".pkl")
                        with open(pkl_path_file_name, "wb") as f:
                            f.write(os.urandom(1024))
                        # Store file path  for json-list
                        json_file_list_cmp.append(pkl_path_file_name)
            else:
                #  Check if list mode is enabled
                if is_list_enabled:
                    # Generate pkl-files according the index list
                    index_list = string_test_values
                    # Create pkl-files for each index
                    for index in index_list:
                        # Create only index folders without sub-directory
                        index_folder = os.path.join(json_path, index)
                        os.makedirs(index_folder, exist_ok=True)
                    # Assemble pkl-path name for first index
                    pkl_path_name = os.path.join(json_path, string_test_values[0], subdirectory)
                else:
                    # Assemble pkl-path name
                    pkl_path_name = os.path.join(json_path, subdirectory)

        # Expected messages
        expected_message = ["",
                            f"Path {json_path} does not exists!",
                            f"pkl-file path {pkl_path_name} does not exists!"]

        # Perform the test
        if not is_json_path_existing or not is_pkl_path_existing:
            # Perform the test
            with pytest.raises(ValueError) as error_message:
                MainCtl._set_processing_complete(json_path, subdirectory, processing_complete_test_file, index_list)
            assert expected_message[exp_message_id] in str(error_message.value)
        else:
            # Perform the test
            MainCtl._set_processing_complete(json_path, subdirectory, processing_complete_test_file, index_list)
            # Load the pkl_file_list from file
            json_path = os.path.join(tmpdir, json_path_name, processing_complete_test_file)
            with open(json_path, "r", encoding="utf-8") as file_handle:
                pkl_file_list = json.load(file_handle)
            # Verify the result
            assert sorted(json_file_list_cmp) == sorted(pkl_file_list)


#########################################################################################################
# _is_processing_complete
#########################################################################################################

# def _is_processing_complete(base_directory: str, act_complete_file_name: str) -> tuple[bool, str]:


# test parameter list
@pytest.mark.parametrize("is_path_existing, is_file_existing, is_file_empty, is_processing_complete, is_list_enabled, exp_message_id", [
    # --parameter set which leads to an error----
    (False, False, False, False, False, 1),
    # -- Negative result----
    (True, False, False, False, False, 2),
    (True, True, True, False, False, 3),
    (True, True, True, False, True, 3),
    (True, True, False, False, False, 4),
    (True, True, False, False, True, 4),
    # -- Positive result----
    (True, True, False, True, False, 0),
    (True, True, False, True, True, 0)]
)
# Unit test function
def test__is_processing_complete(is_path_existing: bool, is_file_existing: bool, is_file_empty: bool,
                                 is_processing_complete: bool, is_list_enabled: bool, exp_message_id: int) -> None:
    """Test method def _set_processing_complete(base_directory: str, subdirectory: str...) -> None: according values.

    :param is_path_existing: Flag to indicate, if the path for json-file exists
    :type  is_path_existing: bool
    :param is_file_existing: Flag to indicate, if the json-file exists
    :type  is_file_existing: bool
    :param is_file_empty: Flag to indicate, if the json-file is empty
    :type  is_file_empty: bool
    :param is_processing_complete: Flag to indicate, if the processing is complete
    :type  is_processing_complete: bool
    :param is_list_enabled: Flag to indicate, if the list is not empty
    :type  is_list_enabled: bool
    :param exp_message_id: List index of the expected message
    :type  exp_message_id: int
    """
    # Variable declaration
    target_study_name = "test_study_name"
    string_test_values: list[str] = ["A", "c9", "Test123", "x_Y_z890", "6Secure_Pass01", "Alpha_Bravo2024", "z_Z9y_Y8_x_X7", "7Long_String_Test999"]
    processing_complete_test_file: str = "processing_complete_test_file.json"
    processing_complete_path_test_file: str
    first_pkl_path_file_name: str = ""
    json_file_list: list[str] = []

    # Prepare the setup
    # Create path
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create the study
        test_study_data: d_dtos.StudyData = d_dtos.StudyData(
            study_name=target_study_name,
            optimization_directory=os.path.join(tmpdir, "component_folder", target_study_name),
            calculation_mode=CalcModeEnum.continue_mode
        )

        # Check if file is existing
        if is_path_existing:
            #  Check if list is available
            if is_list_enabled:
                # Generate pkl-files according the index list
                index_list: list[str] = string_test_values
                # Create pkl-files for each index
                for index in index_list:
                    # Create folder and add index list
                    pkl_path_name = os.path.join(test_study_data.optimization_directory, index, test_study_data.study_name)
                    os.makedirs(pkl_path_name, exist_ok=True)
                    # Store dummy files
                    for pkl_file in string_test_values:
                        pkl_path_file_name = os.path.join(pkl_path_name, pkl_file+".pkl")
                        with open(pkl_path_file_name, "wb") as f:
                            f.write(os.urandom(1024))
                        # Store file path  for json-list
                        json_file_list.append(pkl_path_file_name)

                # Set the name of first element
                first_pkl_path_file_name = os.path.join(test_study_data.optimization_directory, index_list[0],
                                                        test_study_data.study_name, string_test_values[1]+".pkl")

                # Check if processing complete file is store
                if is_file_existing:
                    # Create processing complete file
                    processing_complete_path_test_file = os.path.join(test_study_data.optimization_directory,
                                                                      processing_complete_test_file)
                    with open(processing_complete_path_test_file, "w", encoding="utf-8") as file_handle:
                        json.dump(json_file_list, file_handle, indent=2)

            else:
                # Create folder and add index list
                pkl_path_name = os.path.join(test_study_data.optimization_directory, test_study_data.study_name)
                os.makedirs(pkl_path_name, exist_ok=True)
                # Store dummy files
                for pkl_file in string_test_values:
                    pkl_path_file_name = os.path.join(pkl_path_name, pkl_file + ".pkl")
                    with open(pkl_path_file_name, "wb") as f:
                        f.write(os.urandom(1024))
                    # Store file path  for json-list
                    json_file_list.append(pkl_path_file_name)

                # Set the name of first element
                first_pkl_path_file_name = os.path.join(test_study_data.optimization_directory,
                                                        test_study_data.study_name, string_test_values[0]+".pkl")

                # Check if processing complete file is store
                if is_file_existing:
                    # Create processing complete file
                    processing_complete_path_test_file = os.path.join(test_study_data.optimization_directory,
                                                                      processing_complete_test_file)
                    with open(processing_complete_path_test_file, "w", encoding="utf-8") as file_handle:
                        json.dump(json_file_list, file_handle, indent=2)

            # Check if is_processing_complete is not fulfilled
            if not is_processing_complete:
                # Delete one pkl-file
                os.remove(first_pkl_path_file_name)

            # Check for condition 'is empty file'
            if is_file_empty:
                # Create an empty processing complete file
                MainCtl._set_processing_complete(test_study_data.optimization_directory, "",
                                                 processing_complete_test_file)

        else:
            # Generate a non existing path
            test_study_data.optimization_directory = os.path.join(test_study_data.optimization_directory, "not_existing_folder")

        # Expected messages
        expected_message = ["",
                            f"Path {test_study_data.optimization_directory} does not exists!",
                            f"File {processing_complete_test_file} in path {test_study_data.optimization_directory} does not exists!\n",
                            f"List in file {processing_complete_test_file} is empty!",
                            f"File {first_pkl_path_file_name} does not exists!\n"]

        # Perform the test
        if not is_path_existing:
            # Perform the test
            with pytest.raises(ValueError) as error_message:
                is_complete_result, issue_report = MainCtl._is_processing_complete(test_study_data.optimization_directory,
                                                                                   processing_complete_test_file)
            assert expected_message[exp_message_id] in str(error_message.value)
        else:
            # Perform the test
            is_complete_result, issue_report = (MainCtl._is_processing_complete(test_study_data.optimization_directory,
                                                                                processing_complete_test_file))
            # Verify the result
            assert is_complete_result == is_processing_complete
            assert issue_report == expected_message[exp_message_id]

#########################################################################################################
# _delete_processing_complete
#########################################################################################################

# def _delete_processing_complete(base_directory: str, act_complete_file_name: str) -> bool:

# test parameter list
@pytest.mark.parametrize("test_id, is_path_existing, is_file_existing, is_file_protected, exp_message_id", [
    # --parameter set which leads to an error----
    (0, False, False, False, 1),
    # -- Negative result----
    (1, True, True, True, 2),
    # -- Positive result----
    (2, True, True, False, 0),
    (3, True, False, False, 0)]
)
# Unit test function
def test__delete_processing_complete(caplog: LogCaptureFixture, test_id: int, is_path_existing: bool, is_file_existing: bool,
                                     is_file_protected: bool, exp_message_id: int) -> None:
    """Test method def _set_processing_complete(base_directory: str, subdirectory: str...) -> None: according values.

    :param caplog: class instance for logger data
    :type  caplog: LogCaptureFixture
    :param test_id: Index of the test to change test file name
    :type  test_id: int
    :param is_path_existing: Flag to indicate, if the path exists
    :type  is_path_existing: bool
    :param is_file_existing: Flag to indicate, if the file exists
    :type  is_file_existing: bool
    :param is_file_protected: Flag to indicate, if the file is 'write protected'
    :type  is_file_protected: bool
    :param exp_message_id: List index of the expected message
    :type  exp_message_id: int
    """
    # Variable declaration
    string_test_values: list[str] = ["A", "c9", "Test123", "x_Y_z890", "6Secure_Pass01", "Alpha_Bravo2024", "z_Z9y_Y8_x_X7", "7Long_String_Test999"]
    test_folder_name: str = "test_folder"
    test_file_name: str = ""
    test_path_file_name: str = ""

    # Prepare the setup
    # Create path
    with tempfile.TemporaryDirectory() as tmpdir:
        # Assemble json path
        test_path = os.path.join(tmpdir, test_folder_name)
        # Check if json-file path exists
        if is_path_existing:
            # Create json path
            os.makedirs(test_path, exist_ok=True)
            # Assemble file path name
            test_file_name = string_test_values[test_id]+".json"
            test_path_file_name = os.path.join(test_path, test_file_name)
            # Check if pkl-file path exists
            if is_file_existing:
                #  Create file
                with open(test_path_file_name, "wb") as f:
                    f.write(os.urandom(1024))
                #  Check if file is write protected
                if is_file_protected:
                    # Generate pkl-files according the index list
                    os.chmod(test_path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                    # (stat.S_IWRITE | stat.S_IWGRP | stat.S_IWOTH)

        # Expected messages
        expected_message = ["",
                            f"Path {test_path} does not exists!",
                            f"File {test_path_file_name} is write protected!"]

        # Perform the test
        with caplog.at_level(logging.INFO):
            if not is_path_existing:
                # Perform the test
                with pytest.raises(ValueError) as error_message:
                    MainCtl._delete_processing_complete(test_path, test_file_name)
                assert expected_message[exp_message_id] in str(error_message.value)
            else:
                # Perform the test
                is_deleted = MainCtl._delete_processing_complete(test_path, test_file_name)

                # Check write protection
                if is_file_protected:
                    # Undo write protection
                    os.chmod(test_path, stat.S_IWRITE | stat.S_IWGRP | stat.S_IWOTH | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

                # Verify the result
                assert os.path.isfile(test_path_file_name) != is_deleted
                if len(caplog.records) > 0:
                    assert caplog.records[0].message.startswith(expected_message[exp_message_id])
                else:
                    assert "" == expected_message[exp_message_id]


#########################################################################################################
# _update_calculation_mode, _get_calculation_mode
#########################################################################################################

# def _update_calculation_mode(circuit_calculation_mode: CalcModeEnum, dependent_study_data: StudyData) -> None:
# def _get_calculation_mode(calculation_mode_value: str) -> CalcModeEnum:

# test parameter list
@pytest.mark.parametrize(
    "mode_value, study_mode, expected_result_get, exp_result_update, is_no_error_raise", [
        # --invalid entries ----
        ("x_y_x_d", CalcModeEnum.skip_mode, CalcModeEnum.skip_mode, CalcModeEnum.skip_mode, False),
        ("", CalcModeEnum.new_mode, CalcModeEnum.skip_mode, CalcModeEnum.skip_mode, False),
        # --valid entries----
        ("new", CalcModeEnum.new_mode, CalcModeEnum.new_mode, CalcModeEnum.new_mode, True),
        ("continue", CalcModeEnum.new_mode, CalcModeEnum.continue_mode, CalcModeEnum.new_mode, True),
        ("skip", CalcModeEnum.new_mode, CalcModeEnum.skip_mode, CalcModeEnum.new_mode, True),
        ("new", CalcModeEnum.continue_mode, CalcModeEnum.new_mode, CalcModeEnum.new_mode, True),
        ("continue", CalcModeEnum.continue_mode, CalcModeEnum.continue_mode, CalcModeEnum.continue_mode, True),
        ("skip", CalcModeEnum.continue_mode, CalcModeEnum.skip_mode, CalcModeEnum.continue_mode, True),
        ("new", CalcModeEnum.skip_mode, CalcModeEnum.new_mode, CalcModeEnum.new_mode, True),
        ("continue", CalcModeEnum.skip_mode, CalcModeEnum.continue_mode, CalcModeEnum.continue_mode, True),
        ("skip", CalcModeEnum.skip_mode, CalcModeEnum.skip_mode, CalcModeEnum.skip_mode, True)
    ])
# Unit test function
def test__get_calculation_mode_and_update_calculation_mode(mode_value: str, study_mode: CalcModeEnum,
                                                           expected_result_get: CalcModeEnum,
                                                           exp_result_update: CalcModeEnum, is_no_error_raise: bool) -> None:
    """Test method def _get_calculation_mode and _update_calculation_mode.

    :param mode_value: string of mode value
    :type  mode_value: str
    :param study_mode: Mode of the study to update
    :type  study_mode
    :param expected_result_get: expected result of get calculation mode function
    :param expected_result_get: CalcModeEnum
    :param exp_result_update: expected result of update calculation mode function
    :param exp_result_update: CalcModeEnum
    :param is_no_error_raise: Indicates if the test is performed without raising error
    :type  is_no_error_raise: bool
    """
    # Variable declaration
    target_study_name: str = "test_study_name"
    exp_error_message: str = (f"Enum value {mode_value} is invalid!\n"
                              f"only {CalcModeEnum.new_mode.value}, {CalcModeEnum.continue_mode.value}, "
                              f"and {CalcModeEnum.skip_mode.value} are valid!\n")

    # Prepare the setup
    # Create path
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create the study
        test_study_data: d_dtos.StudyData = d_dtos.StudyData(
            study_name=target_study_name,
            optimization_directory=os.path.join(tmpdir),
            calculation_mode=study_mode
        )

        # Check for valid result
        if is_no_error_raise:
            # Perform the test of _get_calculation_mode
            test_result_get = MainCtl._get_calculation_mode(mode_value)
            # Verify the result
            assert test_result_get == expected_result_get
            # Perform the test of _update_calculation_mode
            MainCtl._update_calculation_mode(test_result_get, test_study_data)
            assert test_study_data.calculation_mode == exp_result_update

        else:
            # Perform the test
            with pytest.raises(ValueError) as error_message:
                test_result_get = MainCtl._get_calculation_mode(mode_value)
            assert exp_error_message in str(error_message.value)

#########################################################################################################
# test of get_initialization_queue_data
# test of _get_page_main_data
# test of _get_page_detail_data
#########################################################################################################

# def _get_page_main_data(self) -> srv_ctl_dtos.QueueMainData:
# test parameter list (counter)
@pytest.mark.parametrize("test_index", list(range(10)))
# Unit test function
def test__get_page_main_data(test_index: int) -> None:
    """Test the method load_toml_file.

    :param test_index: Test index of performed test is used as start index for test lists
    :type  test_index: int
    """
    # Variable declaration
    # Minimal value, in between value and maximal value
    int_test_values: list[int] = [0, 1, -1, 42, sys.maxsize, -sys.maxsize - 1]
    int_test_len = len(int_test_values)

    # Minimal value, in between value and maximal value (Definition range 0<x<max_integer)
    pos_int_test_values: list[int] = [1, 42, 298, sys.maxsize]
    pos_int_test_len = len(pos_int_test_values)

    float_test_values: list[float] = [-sys.float_info.max, -3.2, -1.2, -3.4e-12, -sys.float_info.min, 0]
    float_test_len = len(float_test_values)

    string_test_values: list[str] = ["A", "b9", "Test123", "x_Y_z890", "6Secure//Pass01", "Alpha//Bravo2024", "z_Z9y_Y8_x_X7", "7Long_String_Test999"]
    str_test_len = len(string_test_values)

    string_test_arrays: list[list[str]] = [
        ["A", "b9"],
        ["Test123", "x_Y_z890"],
        ["6Secure_Pass01", "Alpha_Bravo2024"],
        ["z_Z9y_Y8_x_X7", "7Long_String_Test999"],
        ["A", "7Long_String//Test999"],
        ["Long_String_Test999", "B9"],
        ["Alpha//Bravo2024", "6Secure_Pass01"],
        ["Test_123", "7Long_String//Test_999"]]
    # Length of string_test_arrays
    str_list_len = len(string_test_arrays)

    # Constant values of FlowControl
    test_parameter_1: tc.FlowControl = copy.deepcopy(test_FlowControl_base)

    # Initialize the statistical data
    start_progress_data: pcdt.ProgressData = pcdt.ProgressData(
        run_time=0, number_of_filtered_points=0,
        progress_status=pcdt.ProgressStatus.Idle)

    progress_data_circuit: pcdt.ProgressData = pcdt.ProgressData(
        run_time=float_test_values[test_index % float_test_len],
        number_of_filtered_points=pos_int_test_values[test_index % pos_int_test_len],
        progress_status=pcdt.ProgressStatus.Done)

    progress_data_heat_sink: pcdt.ProgressData = pcdt.ProgressData(
        run_time=float_test_values[(test_index + 1) % float_test_len],
        number_of_filtered_points=pos_int_test_values[(test_index + 1) % pos_int_test_len],
        progress_status=pcdt.ProgressStatus.Done)

    progress_data_summary: pcdt.ProgressData = pcdt.ProgressData(
        run_time=float_test_values[(test_index + 2) % float_test_len],
        number_of_filtered_points=pos_int_test_values[(test_index + 2) % pos_int_test_len],
        progress_status=pcdt.ProgressStatus.Done)

    filtered_point_list_data: list[tuple[str, int] | Any] = [["Hallo", pos_int_test_values[(test_index + 5) % pos_int_test_len]]]

    # Initialize test_parameter 1 and expected results
    # Circuit
    test_parameter_1.configuration_data_files.topology_files[1] = string_test_values[test_index % str_test_len]
    test_parameter_1.circuit.number_of_trials = int_test_values[test_index % int_test_len]
    exp_result_circuit: list[srv_ctl_dtos.ConfigurationDataEntryDto] = [srv_ctl_dtos.ConfigurationDataEntryDto(
        configuration_name=string_test_values[test_index % str_test_len],
        number_of_trials=int_test_values[test_index % int_test_len],
        progress_data=copy.deepcopy(start_progress_data))]

    # Inductor
    test_parameter_1.configuration_data_files.inductor_configuration_files[0] = string_test_values[(test_index + 1) % str_test_len]
    test_parameter_1.inductor.numbers_of_trials[0] = int_test_values[(test_index + 1) % int_test_len]
    exp_result_inductor_main: list[srv_ctl_dtos.MagneticDataEntryDto] = [srv_ctl_dtos.MagneticDataEntryDto(
        magnetic_configuration_name=string_test_values[(test_index + 1) % str_test_len],
        number_calculations=0, number_performed_calculations=0,
        progress_data=copy.deepcopy(start_progress_data))]
    test_parameter_1.configuration_data_files.inductor_configuration_files[1] = string_test_values[(test_index + 2) % str_test_len]
    test_parameter_1.inductor.numbers_of_trials[1] = int_test_values[(test_index + 2) % int_test_len]
    exp_result_inductor_main.append(srv_ctl_dtos.MagneticDataEntryDto(
        magnetic_configuration_name=string_test_values[(test_index + 2) % str_test_len],
        number_calculations=0, number_performed_calculations=0,
        progress_data=copy.deepcopy(start_progress_data)))
    # Inductor list (entry per configuration)
    exp_result_inductor: list[srv_ctl_dtos.ConfigurationDataEntryDto] = [srv_ctl_dtos.ConfigurationDataEntryDto(
        configuration_name=string_test_values[(test_index + 1) % str_test_len],
        number_of_trials=int_test_values[(test_index + 1) % int_test_len],
        progress_data=copy.deepcopy(start_progress_data))]
    exp_result_inductor.append(srv_ctl_dtos.ConfigurationDataEntryDto(
        configuration_name=string_test_values[(test_index + 2) % str_test_len],
        number_of_trials=int_test_values[(test_index + 2) % int_test_len],
        progress_data=copy.deepcopy(start_progress_data)))
    # Transformer
    test_parameter_1.configuration_data_files.transformer_configuration_files[0] = string_test_values[(test_index + 2) % str_test_len]
    test_parameter_1.transformer.numbers_of_trials[0] = int_test_values[(test_index + 2) % int_test_len]
    exp_result_transformer_main: list[srv_ctl_dtos.MagneticDataEntryDto] = [srv_ctl_dtos.MagneticDataEntryDto(
        magnetic_configuration_name=string_test_values[(test_index + 2) % str_test_len],
        number_calculations=0, number_performed_calculations=0,
        progress_data=copy.deepcopy(start_progress_data))]
    test_parameter_1.configuration_data_files.transformer_configuration_files[1] = string_test_values[(test_index + 3) % str_test_len]
    test_parameter_1.transformer.numbers_of_trials[1] = int_test_values[(test_index + 3) % int_test_len]
    exp_result_transformer_main.append(srv_ctl_dtos.MagneticDataEntryDto(
        magnetic_configuration_name=string_test_values[(test_index + 3) % str_test_len],
        number_calculations=0, number_performed_calculations=0,
        progress_data=copy.deepcopy(start_progress_data)))
    # Transformer data (List per configuration)
    exp_result_transformer: list[srv_ctl_dtos.ConfigurationDataEntryDto] = [srv_ctl_dtos.ConfigurationDataEntryDto(
        configuration_name=string_test_values[(test_index + 2) % str_test_len],
        number_of_trials=int_test_values[(test_index + 2) % int_test_len],
        progress_data=copy.deepcopy(start_progress_data))]
    exp_result_transformer.append(srv_ctl_dtos.ConfigurationDataEntryDto(
        configuration_name=string_test_values[(test_index + 3) % str_test_len],
        number_of_trials=int_test_values[(test_index + 3) % int_test_len],
        progress_data=copy.deepcopy(start_progress_data)))

    # Heat_sink data (List per configuration)
    test_parameter_1.configuration_data_files.heat_sink_configuration_file = string_test_values[(test_index + 3) % str_test_len]
    test_parameter_1.heat_sink.number_of_trials = int_test_values[(test_index + 3) % int_test_len]
    exp_result_heat_sink: list[srv_ctl_dtos.ConfigurationDataEntryDto] = [srv_ctl_dtos.ConfigurationDataEntryDto(
        configuration_name=string_test_values[(test_index + 3) % str_test_len],
        number_of_trials=int_test_values[(test_index + 3) % int_test_len],
        progress_data=copy.deepcopy(start_progress_data))]

    # Summary data (List per circuit configuration)
    exp_result_summary: list[srv_ctl_dtos.SummaryDataEntryDto] = [srv_ctl_dtos.SummaryDataEntryDto(
        configuration_name=exp_result_circuit[0].configuration_name,
        number_of_combinations=0,
        progress_data=copy.deepcopy(start_progress_data))]

    # Queue main data
    exp_result_queue_main_data: srv_ctl_dtos.QueueMainData = srv_ctl_dtos.QueueMainData(
        circuit_list=copy.deepcopy(exp_result_circuit),
        inductor_main_list=copy.deepcopy(exp_result_inductor_main),
        transformer_main_list=copy.deepcopy(exp_result_transformer_main),
        heat_sink_list=copy.deepcopy(exp_result_heat_sink),
        summary_list=copy.deepcopy(exp_result_summary),
        total_process_time=0,
        break_point_notification=string_test_values[(test_index + 4) % str_test_len])

    # Queue detail data
    exp_result_queue_detail_data: srv_ctl_dtos.QueueDetailData = srv_ctl_dtos.QueueDetailData(
        circuit_data=srv_ctl_dtos.CircuitConfigurationDataDto(
            configuration_name=exp_result_circuit[0].configuration_name,
            number_of_trials=exp_result_circuit[0].number_of_trials,
            filtered_points_name_list=(
                [[string_test_arrays[test_index % str_list_len][0], 0],
                 [string_test_arrays[test_index % str_list_len][1], 1]]),
            progress_data=copy.deepcopy(progress_data_circuit)),
        inductor_list=copy.deepcopy(exp_result_inductor),
        transformer_list=copy.deepcopy(exp_result_transformer),
        heat_sink_list=copy.deepcopy(exp_result_heat_sink),
        summary_data=copy.deepcopy(exp_result_summary[0]),
        conf_process_time=1,
        break_point_notification=string_test_values[(test_index + 4) % str_test_len])

    # Update progress data of circuit and heat sink
    exp_result_queue_main_data.circuit_list[0].progress_data = copy.deepcopy(progress_data_circuit)
    exp_result_queue_main_data.heat_sink_list[0].progress_data = copy.deepcopy(progress_data_heat_sink)
    exp_result_queue_main_data.summary_list[0].progress_data = copy.deepcopy(progress_data_summary)
    exp_result_queue_detail_data.heat_sink_list[0].progress_data = copy.deepcopy(progress_data_heat_sink)
    exp_result_queue_detail_data.summary_data.progress_data = copy.deepcopy(progress_data_summary)

    # Create the instance
    test_pcdt: MainCtl = MainCtl()
    # Allocate pcdt timer members
    test_pcdt._total_time = RunTime()
    test_pcdt._total_time.reset_start_trigger()
    # Breakpoint notification
    test_pcdt._break_point_message = string_test_values[(test_index + 4) % str_test_len]
    # Filtered point name list
    test_pcdt._filtered_list_files = string_test_arrays[test_index % str_list_len]
    # Allocate pcdt optimization objects
    test_pcdt._heat_sink_optimization = pcdt.HeatSinkOptimization()
    test_pcdt._circuit_optimization = CircuitOptimization()
    test_pcdt._summary_processing = SummaryProcessing()
    # Perform the test of get_initialization_queue_data
    (test_pcdt._circuit_list, test_pcdt._inductor_main_list, test_pcdt._inductor_list, test_pcdt._transformer_main_list,
     test_pcdt._transformer_list, test_pcdt._heat_sink_list, test_pcdt._summary_list) = test_pcdt.get_initialization_queue_data(test_parameter_1)

    # Set circuit- and heat sink optimization progress data and summary progress data
    test_pcdt._circuit_optimization._progress_data = progress_data_circuit
    test_pcdt._heat_sink_optimization._progress_data = progress_data_heat_sink
    test_pcdt._summary_processing._progress_data = progress_data_summary
    # Stop timers
    test_pcdt._total_time.stop_trigger()

    # Overtake the result to expected time data
    exp_result_queue_main_data.total_process_time = test_pcdt._total_time.get_runtime()
    exp_result_queue_detail_data.conf_process_time = test_pcdt._total_time.get_runtime()

    # Check results of get_initialization_queue_data
    assert test_pcdt._circuit_list == exp_result_circuit
    assert test_pcdt._inductor_main_list == exp_result_inductor_main
    assert test_pcdt._inductor_list == exp_result_inductor
    assert test_pcdt._transformer_main_list == exp_result_transformer_main
    assert test_pcdt._transformer_list == exp_result_transformer
    assert test_pcdt._heat_sink_list == exp_result_heat_sink
    assert test_pcdt._summary_list == exp_result_summary

    # Perform the test of _get_page_main_data
    result_main_data: srv_ctl_dtos.QueueMainData = test_pcdt._get_page_main_data()

    # Perform the test of _get_page_detailed_data
    result_detail_data: srv_ctl_dtos.QueueDetailData = test_pcdt._get_page_detail_data(0)

    # Check results of _get_page_main_data
    assert result_main_data == exp_result_queue_main_data
    assert result_detail_data == exp_result_queue_detail_data

#########################################################################################################
# test of internal class RunTime with methods
# reset_start_trigger
# continue_trigger
# stop_trigger
# is_timer_active
# get_runtime
#########################################################################################################

# test parameter list with timer index configurations
@pytest.mark.parametrize("timer_id_list", [
    # --invalid input parameter----
    # Timer order list
    ([0, 1, 2, 3, 4, 5, 6]),
    ([6, 5, 4, 3, 2, 1, 0]),
    ([0, 6, 5, 3, 1, 4, 2]),
    ([3, 6, 2, 5, 1, 4, 0]),
    ([6, 5, 4, 0, 1, 2, 3]),
    ([4, 1, 3, 6, 2, 0, 5])])
# Unit test function
def test_runtime_class(timer_id_list: list[int]) -> None:
    """Test method load_generate_logging_config(logging_config_file: str) -> None: according values.

    :param timer_id_list: List of timer order to test
    :type  timer_id_list: list[int]
    """
    # Variable declaration
    # List of timer objects of type pcdt.MainCtl.RunTime
    test_timer_list: list[RunTime] = []

    # Create the instance
    test_pcdt: MainCtl = MainCtl()

    # Allocate pcdt timer members
    for test_timer_id in range(len(timer_id_list)):
        # Create timer
        test_timer: RunTime = RunTime()
        # Add to list
        test_timer_list.append(test_timer)
        assert not test_timer_list[test_timer_id].is_timer_active()
        assert test_timer_list[test_timer_id].get_runtime() == 0
        assert not test_timer_list[test_timer_id].is_timer_active()

    # Test all timers by run 1 timer
    for test_timer_id in timer_id_list:
        # Start timer
        test_timer_list[test_timer_id].reset_start_trigger()
        assert test_timer_list[test_timer_id].is_timer_active()
        # Delay for 100ms
        time.sleep(0.1)
        # Get reference value 1
        minimal_time_offset = time.perf_counter()
        # (Re)-Start timer
        test_timer_list[test_timer_id].reset_start_trigger()
        # Get reference value 2
        maximal_time_offset = time.perf_counter()
        # Check remaining timers
        for test_remaining_timer_id in timer_id_list:
            if not test_remaining_timer_id == test_timer_id:
                assert test_timer_list[test_remaining_timer_id].get_runtime() == 0
                assert not test_timer_list[test_remaining_timer_id].is_timer_active()
                # Check running timer
                assert test_timer_list[test_timer_id].is_timer_active()
                minimal_time_sample = time.perf_counter() - maximal_time_offset
                test_time_sample = test_timer_list[test_timer_id].get_runtime()
                maximal_time_sample = time.perf_counter() - minimal_time_offset
                assert test_time_sample >= minimal_time_sample
                assert test_time_sample <= maximal_time_sample
        # Stop running timer
        test_timer_list[test_timer_id].stop_trigger()
        assert not test_timer_list[test_timer_id].is_timer_active()
        # Set runtime back to 0
        test_timer_list[test_timer_id]._run_time = 0.0
        assert test_timer_list[test_timer_id].get_runtime() == 0
    # Delay for 100ms before next test
    time.sleep(0.1)

    # Test all timers by run almost all timers
    # Start all timers
    # Get reference value 1
    minimal_time_offset = time.perf_counter()
    # Start almost all timers
    for test_timer_id in timer_id_list:
        assert not test_timer_list[test_timer_id].is_timer_active()
        test_timer_list[test_timer_id].reset_start_trigger()
        assert test_timer_list[test_timer_id].is_timer_active()
    # Get reference value 2
    maximal_time_offset = time.perf_counter()
    # Calculate start time for reconfigure stopped timer
    start_time_refresh = (minimal_time_offset + maximal_time_offset)/2

    for test_timer_id in timer_id_list:
        # Stop one timer
        minimal_time_sample = time.perf_counter() - maximal_time_offset
        test_timer_list[test_timer_id].stop_trigger()
        maximal_time_sample = time.perf_counter() - minimal_time_offset
        assert test_timer_list[test_timer_id].get_runtime() >= minimal_time_sample
        assert test_timer_list[test_timer_id].get_runtime() <= maximal_time_sample
        # Check remaining timers
        for test_remaining_timer_id in timer_id_list:
            if not test_remaining_timer_id == test_timer_id:
                # Check running timer
                assert test_timer_list[test_remaining_timer_id].is_timer_active()
                minimal_time_sample = time.perf_counter() - maximal_time_offset
                test_time_sample = test_timer_list[test_remaining_timer_id].get_runtime()
                maximal_time_sample = time.perf_counter() - minimal_time_offset
                assert test_time_sample >= minimal_time_sample
                assert test_time_sample <= maximal_time_sample
        # Reconfigure stopped timer
        test_timer_list[test_timer_id].continue_trigger()
        assert test_timer_list[test_timer_id].is_timer_active()
        # Refresh start time
        test_timer_list[test_timer_id]._start_time = start_time_refresh
        # Delay for 100ms
        time.sleep(0.05)

    # Test all timers by run all timers and stop them one by one
    # Start all timers
    # Get reference value 1
    minimal_time_offset = time.perf_counter()
    # Restart all timers
    for test_timer_id in timer_id_list:
        assert test_timer_list[test_timer_id].is_timer_active()
        test_timer_list[test_timer_id].reset_start_trigger()
        assert test_timer_list[test_timer_id].is_timer_active()
    # Get reference value 2
    maximal_time_offset = time.perf_counter()
    # Delay for 100ms to increase runtime for all timers
    time.sleep(0.05)

    # Stop one by one timer loop
    for test_timer_list_index in range(len(timer_id_list)):
        # Stop the next timer
        minimal_time_sample = time.perf_counter() - maximal_time_offset
        test_timer_list[timer_id_list[test_timer_list_index]].stop_trigger()
        maximal_time_sample = time.perf_counter() - minimal_time_offset
        assert test_timer_list[timer_id_list[test_timer_list_index]].get_runtime() >= minimal_time_sample
        assert test_timer_list[timer_id_list[test_timer_list_index]].get_runtime() <= maximal_time_sample
        # Delay for 100ms to let other timer continue running
        time.sleep(0.05)
        # Check remaining timers
        for test_timer_remaining_list_index in range(len(timer_id_list)):
            if test_timer_remaining_list_index > test_timer_list_index:
                # Check running timer
                assert test_timer_list[timer_id_list[test_timer_remaining_list_index]].is_timer_active()
                minimal_time_sample = time.perf_counter() - maximal_time_offset
                test_time_sample = test_timer_list[timer_id_list[test_timer_remaining_list_index]].get_runtime()
                maximal_time_sample = time.perf_counter() - minimal_time_offset
                assert test_time_sample >= minimal_time_sample
                assert test_time_sample <= maximal_time_sample
            elif test_timer_remaining_list_index < test_timer_list_index:
                assert not test_timer_list[timer_id_list[test_timer_remaining_list_index]].is_timer_active()
                runtime_value1 = test_timer_list[timer_id_list[test_timer_remaining_list_index]].get_runtime()
                runtime_value2 = test_timer_list[timer_id_list[test_timer_list_index]].get_runtime()
                assert runtime_value1 < runtime_value2

#########################################################################################################
# test of _request_pareto_front
#########################################################################################################

#########################################################################################################
# test of _srv_response_queue
#########################################################################################################

#########################################################################################################
# test of run_optimization_from_toml_configuration
#########################################################################################################
