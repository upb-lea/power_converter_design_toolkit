"""Unit tests for circuit optimization."""

# python libraries
import logging
import copy
from enum import Enum
import os
import tempfile

# 3rd party libraries
import pytest
from _pytest.logging import LogCaptureFixture
from numpy.testing import assert_array_equal

# own libraries
from pcdt.topology.dab.dab_circuit_topology import DabCircuitOptimization as ClassUnderTest
import pcdt.topology.dab.dab_circuit_topology_dtos as d_dtos
import pcdt.topology.dab.dab_datasets as d_set
import pcdt.topology.dab.dab_toml_checker as dab_tc
import transistordatabase as tdb
from pcdt.circuit_enums import SamplingEnum
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
    # Data are not initialized
    DataNotInitialized = 8  # Test when Number of entries in additional point list is inconsistent

#########################################################################################################
# Functions to setup test configurations
#########################################################################################################

def generate_additional_point_list(number_of_points: int, boundary_list: list[float]) -> list[list[float]]:
    """Generate a list with additional points.

    This method generates a list of additional point lists. The property of the generated additional point lists
    is like follow: Values at lower boundary | values at upper boundary | values in between | values in between |
                    values in between | values exceed the lower limit | values exceed the upper limit

    :param number_of_points: Number of additional points
    :type  number_of_points: int
    :param boundary_list: List with minimum and maximum value of the points
    :type  boundary_list: list[float]
    :return: List of lists of additional points
    :rtype: list[list[float]
    """
    # Variable declaration
    difference = boundary_list[1] - boundary_list[0]
    delta = difference / number_of_points / 20
    step = (difference - (4 * delta)) / number_of_points
    lower_boundary: list[float] = []
    upper_boundary: list[float] = []
    in_between: list[float] = []
    exceed_lower_boundary: list[float] = []
    exceed_upper_boundary: list[float] = []
    for item in range(number_of_points):
        lower_boundary.append(boundary_list[0] + (item * delta))
        upper_boundary.append(boundary_list[1] - (item * delta))
        in_between.append(boundary_list[0] + delta + (item * step))
        exceed_lower_boundary.append(boundary_list[0] - (item * delta))
        exceed_upper_boundary.append(boundary_list[1] + delta + (item * step))

    # Create new objects
    in_between1 = copy.deepcopy(in_between)
    in_between2 = copy.deepcopy(in_between)
    in_between3 = copy.deepcopy(in_between)

    result_list: list[list[float]] = (
        [lower_boundary, upper_boundary, in_between, in_between1, in_between2, in_between3, exceed_lower_boundary, exceed_upper_boundary])

    return result_list

def generate_weighting_point_list(number_of_points: int) -> list[list[float]]:
    """Generate a list with additional weighting points.

    This method generates a list of additional weighting point lists. The property of the generated additional point lists
    is like follow: Values at lower boundary | values at upper boundary | values in between | values in between |
                    values in between | values exceed the lower limit | values exceed the upper limit

    :param number_of_points: Number of additional weighting points
    :type  number_of_points: int
    :return: List of lists of additional weighting points
    :rtype: list[list[float]
    """
    # Variable declaration
    delta = 1 / (number_of_points)
    lower_boundary: list[float] = []
    upper_boundary: list[float] = []
    in_between: list[float] = []
    exceed_lower_boundary: list[float] = []
    exceed_upper_boundary: list[float] = []
    for _ in range(number_of_points):
        lower_boundary.append(0)
        upper_boundary.append(delta)
        in_between.append(delta * 0.9)
        exceed_lower_boundary.append(delta * 0.9)
        exceed_upper_boundary.append(delta * 1.1)

    # Set the first value below limit (0)
    exceed_lower_boundary[0] = -0.01

    # Create new objects
    in_between1 = copy.deepcopy(in_between)
    in_between2 = copy.deepcopy(in_between)
    in_between3 = copy.deepcopy(in_between)

    result_list: list[list[float]] = (
        [lower_boundary, upper_boundary, in_between, in_between1, in_between2, in_between3, exceed_lower_boundary, exceed_upper_boundary])

    return result_list

#########################################################################################################
# test of calculate_fixed_parameters
#########################################################################################################

@pytest.mark.parametrize(
    "sampling_method, v1_additional_user_point_list, v2_additional_user_point_list, p_additional_user_point_list, "
    "additional_user_weighting_point_list, expected_exception, result_weighting, expected_message_id_list",
    [
        # user-given operating points
        (SamplingEnum.latin_hypercube, [700], [200], [1000], [0.5], None, [[[0.1], [0.1], [0.1], [0.1], [0.1], [0.5]]], [2]),
        # no user-given operating points
        (SamplingEnum.latin_hypercube, [], [], [], [], None, [[[0.2], [0.2], [0.2], [0.2], [0.2]]], [3]),
        # meshgrid, no user-given operating points. Internal algorithm increases sampling points from 5 to 8, so weighting is 0.125
        (SamplingEnum.meshgrid, [], [], [], [], None, [[[0.125, 0.125], [0.125, 0.125]], [[0.125, 0.125], [0.125, 0.125]]], [0, 1, 4]),
        # meshgrid, with user-given operating points (will be ignored). Internal algorithm increases sampling points from 5 to 8, so weighting is 0.125
        (SamplingEnum.meshgrid, [700], [200], [1000], [0.5], None, [[[0.125, 0.125], [0.125, 0.125]], [[0.125, 0.125], [0.125, 0.125]]], [0, 1, 4]),
        # value error expected
        (SamplingEnum.latin_hypercube, [700], [200], [1000], [1.5], ValueError, None, []),
    ]
)
def test_calculate_fix_parameters(caplog: LogCaptureFixture, sampling_method: SamplingEnum, v1_additional_user_point_list: list[float],
                                  v2_additional_user_point_list: list[float],
                                  p_additional_user_point_list: list[float], additional_user_weighting_point_list: list[float],
                                  expected_exception: type, result_weighting: list[float], expected_message_id_list: dict) -> None:
    """
    Unit test to check the fix parameters, especially the sampling.

    :param caplog: pytest feature to read logger messages
    :param sampling_method: sampling method
    :param v1_additional_user_point_list: user-given operating points for v1
    :param v2_additional_user_point_list: user-given operating points for v2
    :param p_additional_user_point_list: user-given operating points for power
    :param additional_user_weighting_point_list: user-given operating point weighting
    :param expected_exception: expected exception
    :param result_weighting: unit test results
    :param expected_message_id_list: expected message IDs in a list
    """
    # set logging level to warning
    caplog.set_level(logging.INFO)

    # Create the instance
    test_object: ClassUnderTest = ClassUnderTest()

    # Initialize configuration parameter
    design_space = d_dtos.CircuitParetoDesignSpace(
        f_s_min_max_list=[100_000, 200_000],
        l_s_min_max_list=[10e-6, 10e-3],
        l_1_min_max_list=[10e-6, 10e-3],
        l_2__min_max_list=[10e-6, 10e-3],
        n_min_max_list=[1, 3],
        transistor_1_name_list=['CREE_C3M0065100J', 'CREE_C3M0120100J'],
        transistor_2_name_list=['CREE_C3M0060065J', 'CREE_C3M0120065J'],
        c_par_1=1e-12,
        c_par_2=1e-12,
        t_dead_1_max=100e-9,
        t_dead_2_max=100e-9)

    output_range = d_dtos.CircuitOutputRange(
        v1_min_max_list=[690, 710],
        v2_min_max_list=[175, 295],
        p_min_max_list=[-2000, 2000])

    filter = d_dtos.CircuitFilter(
        number_filtered_designs=1,
        difference_percentage=5)

    info_message_dict = {0: "Number of sampling points has been updated from 5 to 8.",
                         1: "Note: meshgrid sampling does not take user-given operating points into account",
                         2: "Auto-weight given for all other 5 operating points: 0.1",
                         3: "Auto-weight given for all other 5 operating points: 0.2",
                         4: "Auto-weight given for all other 8 operating points: 0.125"}

    # set up sampling
    sampling = d_dtos.CircuitSampling(
        sampling_method=sampling_method,
        sampling_points=5,
        sampling_random_seed=1,
        v1_additional_user_point_list=v1_additional_user_point_list,
        v2_additional_user_point_list=v2_additional_user_point_list,
        p_additional_user_point_list=p_additional_user_point_list,
        additional_user_weighting_point_list=additional_user_weighting_point_list
    )

    # set up input configuration
    dab_config = d_dtos.CircuitParetoDabDesign(
        circuit_study_name="test",
        project_directory="",
        design_space=design_space,
        output_range=output_range,
        sampling=sampling,
        filter=filter)

    # set up file to store transistor losses
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create temporary directory
        valid_filepath = os.path.join(tmpdir, "circuit_folder")
        os.makedirs(valid_filepath, exist_ok=True)
        # Set the path
        d_set.HandleDabDto.set_c_oss_storage_directory(valid_filepath)

        if expected_exception:
            with pytest.raises(expected_exception):
                ClassUnderTest.calculate_fixed_parameters(dab_config)
        else:
            output = ClassUnderTest.calculate_fixed_parameters(dab_config)
            assert_array_equal(result_weighting, output.mesh_weights)

            for info_message_id in expected_message_id_list:
                assert info_message_dict[info_message_id] in caplog.text


#########################################################################################################
# test of test_load_and_verify_general_parameters
#########################################################################################################

# test parameter list (counter)
@pytest.mark.parametrize("test_index, test_type, u_points_test_index, u_points_test_type", [
    # Valid test case
    # Test value at lower boundary
    (0, TestCase.LowerBoundary, -1, TestCase.InBetween),
    # Test value at lower boundary
    (1, TestCase.UpperBoundary, -1, TestCase.InBetween),
    # Test value in between
    (2, TestCase.InBetween, -1, TestCase.InBetween),
    # Failure test case
    # Test when minimum > maximum ( Only for minimum maximum list)
    (3, TestCase.BoundaryInconsistent, -1, TestCase.InBetween),
    # Test when the list has got too few entries ( Only for minimum maximum list)
    (4, TestCase.TooLessEntries, -1, TestCase.InBetween),
    # Test when the list has got too many entries ( Only for minimum maximum list)
    (5, TestCase.TooMuchEntries, -1, TestCase.InBetween),
    # Test when the lower limit is exceeded
    (6, TestCase.ExceedLowerLimit, -1, TestCase.InBetween),
    # Test when the lower limit is exceeded
    (7, TestCase.ExceedUpperLimit, -1, TestCase.InBetween),
    # Test of additional user points
    # Additional user points: Test value at lower boundary
    (2, TestCase.InBetween, 0, TestCase.LowerBoundary),
    # Additional user points: Test value at lower boundary
    (2, TestCase.InBetween, 1, TestCase.UpperBoundary),
    # Additional user points: Test value in between
    (2, TestCase.InBetween, 2, TestCase.InBetween),
    # Failure test case
    # Additional user points: Test when inconsistent number of user points in v1
    (2, TestCase.InBetween, 3, TestCase.BoundaryInconsistent),
    # Additional user points: Test when inconsistent number of user points in v2
    (2, TestCase.InBetween, 4, TestCase.BoundaryInconsistent),
    # Additional user points: Test when inconsistent number of user points in p
    (2, TestCase.InBetween, 5, TestCase.BoundaryInconsistent),
    # Additional user points: Test when the lower limit is exceeded
    (2, TestCase.InBetween, 6, TestCase.ExceedLowerLimit),
    # Additional user points: Test when the lower limit is exceeded
    (2, TestCase.InBetween, 7, TestCase.ExceedUpperLimit)
])
# Unit test function
def test_load_and_verify_general_parameters(test_index: int, test_type: TestCase,
                                            u_points_test_index: int, u_points_test_type: TestCase) -> None:
    """Test the method load_toml_file.

    :param test_index: Test index of the used list element (not valid for additional point list)
    :type  test_index: int
    :param test_type: Type of performed test (not valid for additional point list test)
    :type  test_type: TestCase
    :param u_points_test_index: Test index of the used list element of the additional point list
    :type  u_points_test_index: int
    :param u_points_test_type: Type of performed test for additional point list
    :type  u_points_test_type: TestCase
    """
    # Variable declaration
    test_object: ClassUnderTest = ClassUnderTest()

    # Called only on time while parametric test
    # List entries for values and list (exception *in between for values):
    # at lower boundary | at upper boundary | in between | minimum > maximum* | too few entries*
    # too many entries* | exceed the lower limit | exceed the upper limit
    float_min_max_list_configuration_gt0_lt1500: list[list[float]] = (
        [[1e-18, 1e-18], [1499.9, 1499.9], [1000, 1300], [1000, 300], [500], [600, 1000, 1300], [-10, 1200], [40.5, 1500]])
    float_min_max_list_configuration_gtm100kw_lt100kw: list[list[float]] = (
        [[-9.9999, -9.9999], [9.999e4, 9.9999e4], [2000, 5e4], [2e2, 50], [2000], [2000, 2.5e4, 3e4], [-1.01e5, 3222], [2000, -1.01e5]])
    int_value_gt0 = [1, 181877627, 1111, 4332, 14332, 34544, 0, 10000]
    int_value_ge0 = [0, 181877627, 1111, 4332, 4889393, 334544, -1, 10000]

    # Initialize the general parameters
    test_general_parameter: dab_tc.TomlDabGeneral = dab_tc.TomlDabGeneral(
        output_range=dab_tc.TomlDabOutputRange(
            v1_min_max_list=float_min_max_list_configuration_gt0_lt1500[test_index],
            v2_min_max_list=float_min_max_list_configuration_gt0_lt1500[test_index],
            p_min_max_list=float_min_max_list_configuration_gtm100kw_lt100kw[test_index]),
        sampling=dab_tc.TomlDabSampling(
            sampling_method=SamplingEnum.meshgrid,
            sampling_points=int_value_gt0[test_index],
            sampling_random_seed=int_value_ge0[test_index],
            v1_additional_user_point_list=[],
            v2_additional_user_point_list=[],
            p_additional_user_point_list=[],
            additional_user_weighting_point_list=[]),
    )

    # Check if number of user points>0
    if u_points_test_index >= 0:
        # at lower boundary | at upper boundary | in between | inconsistent number of entries V1 | inconsistent number of entries V2
        # inconsistent number of entries p | exceed the lower limit | exceed the upper limit
        v1_additional_point_list: list[list[float]] = generate_additional_point_list(3, test_general_parameter.output_range.v1_min_max_list)
        v2_additional_point_list: list[list[float]] = generate_additional_point_list(3, test_general_parameter.output_range.v2_min_max_list)
        p_additional_point_list: list[list[float]] = generate_additional_point_list(3, test_general_parameter.output_range.p_min_max_list)
        w_point_list: list[list[float]] = generate_weighting_point_list(3)
        # Manipulate list for inconsistency by deleting one value
        del v1_additional_point_list[3][0]
        del v2_additional_point_list[4][0]
        del p_additional_point_list[5][0]
        # Set additional user point parameters
        test_general_parameter.sampling.v1_additional_user_point_list = v1_additional_point_list[u_points_test_index]
        test_general_parameter.sampling.v2_additional_user_point_list = v2_additional_point_list[u_points_test_index]
        test_general_parameter.sampling.p_additional_user_point_list = p_additional_point_list[u_points_test_index]
        test_general_parameter.sampling.additional_user_weighting_point_list = w_point_list[u_points_test_index]
        # In case of additional user point test, the test type for remaining parameter must be in_between
        assert test_type == TestCase.InBetween

    # Perform the test for the general parameters
    # Create boundary list from minimum-maximum list with assigned parameters
    min_max_list_name_list_general: list[str] = ["v1_min_max_list", "v2_min_max_list", "p_min_max_list"]
    value_name_list_general: list[str] = []
    value_name_low_limit_list_general: list[str] = ["sampling_points", "sampling_random_seed"]
    u_point_name_list: list[str] = (["v1_additional_user_point_list", "v2_additional_user_point_list",
                                     "p_additional_user_point_list", "additional_user_weighting_point_list"])

    # Convert general parameter class to a dict
    test_general_parameter_dict = test_general_parameter.model_dump()

    # Perform the test for the general parameters
    is_consistent, error_report = test_object.load_and_verify_general_parameters(test_general_parameter_dict)

    if test_type == TestCase.LowerBoundary or test_type == TestCase.UpperBoundary:
        # No error and empty report string
        assert error_report == ""
        assert is_consistent

    elif test_type == TestCase.InBetween:
        # Check additional point test type
        if u_points_test_type == TestCase.LowerBoundary or u_points_test_type == TestCase.UpperBoundary or u_points_test_type == TestCase.InBetween:
            # No error and empty report string
            assert error_report == ""
            assert is_consistent

        elif u_points_test_type == TestCase.BoundaryInconsistent:
            # Check if not any minimum-maximum list parameters is identified
            for parameter_name in min_max_list_name_list_general:
                assert parameter_name not in error_report

            # Check if not any value_name_list_general parameter is identified
            for parameter_name in value_name_list_general:
                assert parameter_name not in error_report

            # Check if all additional user point list parameters are identified
            for parameter_name in u_point_name_list:
                assert parameter_name in error_report

            # Error is indicated
            assert not is_consistent

        elif u_points_test_type == TestCase.ExceedLowerLimit:
            # Check if not any minimum-maximum list parameters is identified
            for parameter_name in min_max_list_name_list_general:
                assert parameter_name not in error_report

            # Check if not any value_name_list_general parameter is identified
            for parameter_name in value_name_list_general:
                assert parameter_name not in error_report

            # Check if all additional user point list parameters are identified
            for parameter_name in u_point_name_list:
                assert parameter_name in error_report
            # Error is indicated
            assert not is_consistent

        elif u_points_test_type == TestCase.ExceedUpperLimit:
            # Check if not any minimum-maximum list parameters is identified
            for parameter_name in min_max_list_name_list_general:
                assert parameter_name not in error_report

            # Check if not any value_name_list_general parameter is identified
            for parameter_name in value_name_list_general:
                assert parameter_name not in error_report

            # Check if all additional user point list parameters are identified
            for parameter_name in u_point_name_list:
                assert parameter_name in error_report
            # Error is indicated
            assert not is_consistent

    elif test_type == TestCase.ExceedUpperLimit:
        # Check if all minimum-maximum list parameters are identified
        for parameter_name in min_max_list_name_list_general:
            assert parameter_name in error_report

        # Check if all value_name_list_general parameters are identified
        for parameter_name in value_name_list_general:
            assert parameter_name in error_report

        # Check if not any value_name_low_limit_list_general parameter is identified
        for parameter_name in value_name_low_limit_list_general:
            assert parameter_name not in error_report
        # Error is indicated
        assert not is_consistent

    elif test_type == TestCase.ExceedLowerLimit:
        # Check if all minimum-maximum list parameters are identified
        for parameter_name in min_max_list_name_list_general:
            assert parameter_name in error_report

        # Check if all value_name_list_general list parameters are identified
        for parameter_name in value_name_list_general:
            assert parameter_name in error_report

        # Check if all value_name_low_limit_list_general parameters are identified
        for parameter_name in value_name_low_limit_list_general:
            assert parameter_name in error_report
        # Error is indicated
        assert not is_consistent

    elif test_type == TestCase.TooLessEntries:
        # Check if all minimum-maximum list parameters are identified
        for parameter_name in min_max_list_name_list_general:
            assert parameter_name in error_report

        # Check if not any value_name_list_general parameter is identified
        for parameter_name in value_name_list_general:
            assert parameter_name not in error_report

        # Check if not any value_name_low_limit_list_general parameter is identified
        for parameter_name in value_name_low_limit_list_general:
            assert parameter_name not in error_report
        # Error is indicated
        assert not is_consistent

    elif test_type == TestCase.TooMuchEntries:
        # Check if all minimum-maximum list parameters are identified
        for parameter_name in min_max_list_name_list_general:
            assert parameter_name in error_report

        # Check if not any value_name_list_general parameter is identified
        for parameter_name in value_name_list_general:
            assert parameter_name not in error_report

        # Check if not any value_name_low_limit_list_general parameter is identified
        for parameter_name in value_name_low_limit_list_general:
            assert parameter_name not in error_report
        # Error is indicated
        assert not is_consistent

    elif test_type == TestCase.BoundaryInconsistent:
        # Check if all minimum-maximum list parameters are identified
        for parameter_name in min_max_list_name_list_general:
            assert parameter_name in error_report

        # Check if not any value_name_list_general parameter is identified
        for parameter_name in value_name_list_general:
            assert parameter_name not in error_report

        # Check if not any value_name_low_limit_list_general parameter is identified
        for parameter_name in value_name_low_limit_list_general:
            assert parameter_name not in error_report
        # Error is indicated
        assert not is_consistent

#########################################################################################################
# test of load_and_verify_circuit_parameters
#########################################################################################################


""" Information
## Circuit #####################################################################
Implemented boundary
[design_space]
f_s_min_max_list: list[int] -> 1000<range<1e7
l_s_min_max_list: list[float] -> 0<range<1
l_1_min_max_list: list[float] -> 0<range<1
l_2__min_max_list: list[float] -> 0<range<1
n_min_max_list: list[float] -> 0<range<100
transistor_1_name_list: list[str] -> All entries are key names of the transistor database
transistor_2_name_list: list[str] -> All entries are key names of the transistor database
c_par_1: float -> 0<value<1e-3
c_par_2: float -> 0<value<1e-3
[output_range]
v1_min_max_list: list[float] -> 0<range<1500
v2_min_max_list: list[float] -> 0<range<1500
p_min_max_list: list[float] -> -100kW<range<100kW
[sampling]
sampling_method is checked by toml checker)
sampling_points: int -> 0<value
sampling_random_seed: int | Literal["random"] -> All values allowed (Type is checked by toml checker)
v1_additional_user_point_list: list[float] -> Boundary according configuration of v1_min_max_list
v2_additional_user_point_list: list[float] -> -> Boundary according configuration of v2_min_max_list
p_additional_user_point_list: list[float] -> -> Boundary according configuration of p_min_max_list
additional_user_weighting_point_list: list[float] -> 0=<val and sum<=1
[filter_distance]
number_filtered_designs: int -> value >= 1
difference_percentage: float -> 0.01<value<=100
"""

@pytest.fixture(scope="module")
def get_transistor_name_list() -> list[str]:
    """Get transistor names from transistor database.

    :return: List of transistor names
    :rtype: list[str]
    """
    # Variable declaration

    # Create dictionary from transistor database list
    transistor_database = tdb.DatabaseManager()
    transistor_database.set_operation_mode_json()
    keyword_list: list[str] = transistor_database.get_transistor_names_list()
    return keyword_list

# test parameter list (counter)
@pytest.mark.parametrize("test_index, test_type", [
    # Valid test case
    # Test value at lower boundary
    (0, TestCase.LowerBoundary),
    # Test value at lower boundary
    (1, TestCase.UpperBoundary),
    # Test value in between
    (2, TestCase.InBetween),
    # Failure test case
    # Test when minimum > maximum ( Only for minimum maximum list)
    (3, TestCase.BoundaryInconsistent),
    # Test when the list has got too few entries ( Only for minimum maximum list)
    (4, TestCase.TooLessEntries),
    # Test when the list has got too many entries ( Only for minimum maximum list)
    (5, TestCase.TooMuchEntries),
    # Test when the lower limit is exceeded
    (6, TestCase.ExceedLowerLimit),
    # Test when the upper limit is exceeded
    (7, TestCase.ExceedUpperLimit),
])
# Unit test function
def test_load_and_verify_circuit_parameters(get_transistor_name_list: list[str], test_index: int, test_type: TestCase) -> None:
    """Test the method load_toml_file.

    :param get_transistor_name_list: List of transistor names
    :type  get_transistor_name_list: list[str]
    :param test_index: Test index of the used list element (not valid for additional point list)
    :type  test_index: int
    :param test_type: Type of performed test (not valid for additional point list test)
    :type  test_type: TestCase
    """
    # Variable declaration
    test_object: ClassUnderTest = ClassUnderTest()

    # Called only on time while parametric test
    transistor_name_list = get_transistor_name_list
    t_list_len = len(transistor_name_list)
    assert t_list_len > 0
    one_t_name = transistor_name_list[t_list_len//2]
    t_sub_list = copy.deepcopy(transistor_name_list)
    t_sub_list = t_sub_list[:len(t_sub_list)//2]
    t_first_name_wrong_list = copy.deepcopy(transistor_name_list)
    t_first_name_wrong_list[0] = one_t_name+"first_name_wrong"
    t_last_name_wrong_list = copy.deepcopy(transistor_name_list)
    t_last_name_wrong_list[t_list_len-1] = one_t_name+"last_name_wrong"
    # List entries for the list:
    # All names | one name | half of the list | empty list
    # one wrong name | first name wrong | last name wrong
    transistor_name_list_configuration: list[list[str]] = [transistor_name_list, [one_t_name], t_sub_list, ["1", "2", "3"],
                                                           [], [one_t_name+"wrong"], t_first_name_wrong_list,
                                                           t_last_name_wrong_list]

    # List entries for values and list (exception *in between for values):
    # at lower boundary | at upper boundary | in between | minimum > maximum* | too few entries*
    # too many entries* | exceed the lower limit | exceed the upper limit
    int_min_max_list_configuration_gt1000_lt1e7: list[list[int]] = (
        [[1001, 1001], [999999, 999999], [2000, 500000], [5000, 2500], [2000], [2000, 2500, 3000], [10, 3222], [2000, 13400000]])
    float_min_max_list_configuration_gt0_lt1: list[list[float]] = (
        [[1e-17, 1e-17], [0.9991, 0.9991], [0.34, 0.77], [0.74, 0.73], [0.33], [0.33, 0.66, 0.99], [-0.1, 0.88], [0.55, 1.0]])
    float_min_max_list_configuration_gt0_lt100: list[list[float]] = (
        [[1e-18, 1e-18], [99.8, 99.8], [34, 77], [90, 67], [33], [33, 66, 99], [0, 88], [55, 100]])
    float_value_gt0_lt1em3: list[float] = [1e-22, 9.999e-4, 3.55e-4, 4.55e-4, 8.98e-14, 6.6e-12, -1e-3, 1.2e-3]
    float_value_gt1em2_le100: list[float] = [0.011, 100, 55, 99, 45, 67, 9.9e-3, 100.15]
    float_value_list_configuration_gt0_le1em2xgt1_le100: list[list[float]] = (
        [[1e-17, 1.1], [0.01, 100], [0.0034, 73.78], [-1, -282.3], [0.0033], [0.0066, 22.76, 33], [0, 10], [0.011, 100.01]])
    int_value_gt0 = [1, 181877627, 1111, 4332, 14332, 34544, 0, 10000]

    # Initialize the circuit parameters
    test_circuit_parameter: dab_tc.TomlDabCircuitParetoDesign = dab_tc.TomlDabCircuitParetoDesign(
        design_space=dab_tc.TomlDabCircuitParetoDesignSpace(
            f_s_min_max_list=int_min_max_list_configuration_gt1000_lt1e7[test_index],
            l_s_min_max_list=float_min_max_list_configuration_gt0_lt1[test_index],
            l_1_min_max_list=float_min_max_list_configuration_gt0_lt1[test_index],
            l_2__min_max_list=float_min_max_list_configuration_gt0_lt1[test_index],
            n_min_max_list=float_min_max_list_configuration_gt0_lt100[test_index],
            transistor_1_name_list=transistor_name_list_configuration[test_index],
            transistor_2_name_list=transistor_name_list_configuration[test_index],
            c_par_1=float_value_gt0_lt1em3[test_index],
            c_par_2=float_value_gt0_lt1em3[test_index],
            t_dead_1_max=100e-9,
            t_dead_2_max=100e-9),
        thermal_data=dab_tc.TomlDabThermalResistanceData(
            transistor_b1_cooling=float_value_list_configuration_gt0_le1em2xgt1_le100[test_index],
            transistor_b2_cooling=float_value_list_configuration_gt0_le1em2xgt1_le100[test_index]),
        filter_distance=dab_tc.TomlDabCircuitFilterDistance(
            number_filtered_designs=int_value_gt0[test_index],
            difference_percentage=float_value_gt1em2_le100[test_index])
    )

    # Create boundary list from minimum-maximum list with assigned parameters
    min_max_list_name_list: list[str] = ["f_s_min_max_list", "l_s_min_max_list", "l_1_min_max_list", "l_2__min_max_list",
                                         "n_min_max_list", "transistor_b1_cooling", "transistor_b2_cooling"]
    value_name_list: list[str] = ["c_par_1", "c_par_2", "difference_percentage"]
    value_name_low_limit_list: list[str] = []

    # Convert circuit parameter class to a dict
    test_circuit_parameter_dict = test_circuit_parameter.model_dump()

    # Perform the test for the circuit
    is_circuit_consistent, error_report_circuit = test_object.load_and_verify_circuit_parameters(test_circuit_parameter_dict)

    if test_type == TestCase.LowerBoundary or test_type == TestCase.UpperBoundary:
        # No error and empty report string
        assert error_report_circuit == ""
        assert is_circuit_consistent

    elif test_type == TestCase.ExceedUpperLimit:
        # Check if all minimum-maximum list parameters are identified
        for parameter_name in min_max_list_name_list:
            assert parameter_name in error_report_circuit

        # Check if all value_name_list parameters are identified
        for parameter_name in value_name_list:
            assert parameter_name in error_report_circuit

        # Check if not any value_name_low_limit_list parameter is identified
        for parameter_name in value_name_low_limit_list:
            assert parameter_name not in error_report_circuit
        # Error is indicated
        assert not is_circuit_consistent

    elif test_type == TestCase.ExceedLowerLimit:
        # Check if all minimum-maximum list parameters are identified
        for parameter_name in min_max_list_name_list:
            assert parameter_name in error_report_circuit

        # Check if all value_name_list list parameters are identified
        for parameter_name in value_name_list:
            assert parameter_name in error_report_circuit

        # Check if all value_name_low_limit_list parameters are identified
        for parameter_name in value_name_low_limit_list:
            assert parameter_name in error_report_circuit
        # Error is indicated
        assert not is_circuit_consistent

    elif test_type == TestCase.TooLessEntries:
        # Check if all minimum-maximum list parameters are identified
        for parameter_name in min_max_list_name_list:
            assert parameter_name in error_report_circuit

        # Check if not any value_name_list parameter is identified
        for parameter_name in value_name_list:
            assert parameter_name not in error_report_circuit

        # Check if not any value_name_low_limit_list parameter is identified
        for parameter_name in value_name_low_limit_list:
            assert parameter_name not in error_report_circuit
        # Error is indicated
        assert not is_circuit_consistent

    elif test_type == TestCase.TooMuchEntries:
        # Check if all minimum-maximum list parameters are identified
        for parameter_name in min_max_list_name_list:
            assert parameter_name in error_report_circuit

        # Check if not any value_name_list parameter is identified
        for parameter_name in value_name_list:
            assert parameter_name not in error_report_circuit

        # Check if not any value_name_low_limit_list parameter is identified
        for parameter_name in value_name_low_limit_list:
            assert parameter_name not in error_report_circuit
        # Error is indicated
        assert not is_circuit_consistent

    elif test_type == TestCase.BoundaryInconsistent:
        # Check if all minimum-maximum list parameters are identified
        for parameter_name in min_max_list_name_list:
            assert parameter_name in error_report_circuit

        # Check if not any value_name_list parameter is identified
        for parameter_name in value_name_list:
            assert parameter_name not in error_report_circuit

        # Check if not any value_name_low_limit_list parameter is identified
        for parameter_name in value_name_low_limit_list:
            assert parameter_name not in error_report_circuit
        # Error is indicated
        assert not is_circuit_consistent
################

#########################################################################################################
# test of initialize_circuit_optimization
#########################################################################################################

# initialize_circuit_optimization(self, toml_circuit: dab_tc.TomlDabCircuitParetoDesign, toml_prog_flow: tc.FlowControl) -> bool:
# test parameter list (counter)
@pytest.mark.parametrize("test_index, test_type, is_error", [
    # Valid test case
    # Test value at lower boundary
    (0, TestCase.LowerBoundary, False),
    # Test value at upper boundary
    (1, TestCase.UpperBoundary, False),
    # Test value in between
    (2, TestCase.InBetween, False),
    # Failure test case
    # Test when the circuit is not loaded
    (3, TestCase.DataNotInitialized, True),
])
# Unit test function
def test_initialize_circuit_optimization(get_transistor_name_list: list[str], test_index: int, test_type: TestCase, is_error: bool) -> None:
    """Test the method initialize_circuit_optimization.

    :param get_transistor_name_list: List of transistor names
    :type  get_transistor_name_list: list[str]
    :param test_index: Test index of the used list element
    :type  test_index: int
    :param test_type: Type of performed test
    :type  test_type: TestCase
    :param is_error: Indicates, if the function exits with error
    :type  is_error: bool
    """
    # Variable declaration

    # Called only on time while parametric test
    transistor_name_list = get_transistor_name_list
    t_list_len = len(transistor_name_list)
    assert t_list_len > 0
    one_t_name = transistor_name_list[t_list_len//2]
    t_sub_list = copy.deepcopy(transistor_name_list)
    t_sub_list = t_sub_list[:len(t_sub_list)//2]
    t_first_name_wrong_list = copy.deepcopy(transistor_name_list)
    t_first_name_wrong_list[0] = one_t_name+"first_name_wrong"
    t_last_name_wrong_list = copy.deepcopy(transistor_name_list)
    t_last_name_wrong_list[t_list_len-1] = one_t_name+"last_name_wrong"
    # List entries for the list:
    # All names | one name | half of the list | empty list
    # one wrong name | first name wrong | last name wrong
    transistor_name_list_configuration: list[list[str]] = [transistor_name_list, [one_t_name], t_sub_list,
                                                           [one_t_name+"wrong"]]

    # List entries for values and list (exception *in between for values):
    # at lower boundary | at upper boundary | in between | exceed the lower or upper limit
    int_min_max_list_configuration_gt1000_lt1e7: list[list[int]] = (
        [[1001, 2001], [399999, 999999], [2000, 500000], [10, 3222]])
    float_min_max_list_configuration_gt0_lt1: list[list[float]] = (
        [[1e-17, 8.1e-17], [0.1991, 0.9991], [0.34, 0.77], [-0.1, 0.88]])
    float_min_max_list_configuration_gt0_lt100: list[list[float]] = (
        [[1e-18, 7e-18], [69.8, 99.8], [34, 77], [0, 88]])
    float_min_max_list_configuration_gt0_lt1500: list[list[float]] = (
        [[1e-18, 2.2e-18], [10, 1499.9], [1000, 1300], [-10, 1200]])
    float_min_max_list_configuration_gtm100kw_lt100kw: list[list[float]] = (
        [[-9.9999, 0], [22, 9.9999e4], [2000, 5e4], [-1.01e5, 3222]])
    float_value_gt0_lt1em3: list[float] = [1e-22, 9.999e-4, 3.55e-4, -1e-3]
    float_value_gt1em2_le100: list[float] = [0.011, 100, 55, 9.9e-3]
    float_value_list_configuration_gt0_le1em2xgt1_le100: list[list[float]] = (
        [[1e-17, 1.1], [0.01, 100], [0.0034, 73.78], [0, 282.3]])
    int_value_gt0 = [1, 181877627, 1111, 10000]
    int_value_ge0 = [0, 181877627, 1111, 10000]

    # Initialize the general parameters
    test_general_parameter: dab_tc.TomlDabGeneral = dab_tc.TomlDabGeneral(
        output_range=dab_tc.TomlDabOutputRange(
            v1_min_max_list=float_min_max_list_configuration_gt0_lt1500[test_index],
            v2_min_max_list=float_min_max_list_configuration_gt0_lt1500[test_index],
            p_min_max_list=float_min_max_list_configuration_gtm100kw_lt100kw[test_index]),
        sampling=dab_tc.TomlDabSampling(
            sampling_method=SamplingEnum.meshgrid,
            sampling_points=int_value_gt0[test_index],
            sampling_random_seed=int_value_ge0[test_index],
            v1_additional_user_point_list=[],
            v2_additional_user_point_list=[],
            p_additional_user_point_list=[],
            additional_user_weighting_point_list=[])
    )

    # Initialize the circuit parameters
    test_circuit_parameter: dab_tc.TomlDabCircuitParetoDesign = dab_tc.TomlDabCircuitParetoDesign(
        design_space=dab_tc.TomlDabCircuitParetoDesignSpace(
            f_s_min_max_list=int_min_max_list_configuration_gt1000_lt1e7[test_index],
            l_s_min_max_list=float_min_max_list_configuration_gt0_lt1[test_index],
            l_1_min_max_list=float_min_max_list_configuration_gt0_lt1[test_index],
            l_2__min_max_list=float_min_max_list_configuration_gt0_lt1[test_index],
            n_min_max_list=float_min_max_list_configuration_gt0_lt100[test_index],
            transistor_1_name_list=transistor_name_list_configuration[test_index],
            transistor_2_name_list=transistor_name_list_configuration[test_index],
            c_par_1=float_value_gt0_lt1em3[test_index],
            c_par_2=float_value_gt0_lt1em3[test_index],
            t_dead_1_max=100e-9,
            t_dead_2_max=100e-9),
        thermal_data=dab_tc.TomlDabThermalResistanceData(
            transistor_b1_cooling=float_value_list_configuration_gt0_le1em2xgt1_le100[test_index],
            transistor_b2_cooling=float_value_list_configuration_gt0_le1em2xgt1_le100[test_index]),
        filter_distance=dab_tc.TomlDabCircuitFilterDistance(
            number_filtered_designs=int_value_gt0[test_index],
            difference_percentage=float_value_gt1em2_le100[test_index])
    )

    # at lower boundary | at upper boundary | in between | exceed the upper limit
    v1_additional_point_list: list[list[float]] = generate_additional_point_list(5, test_general_parameter.output_range.v1_min_max_list)
    v2_additional_point_list: list[list[float]] = generate_additional_point_list(5, test_general_parameter.output_range.v2_min_max_list)
    p_additional_point_list: list[list[float]] = generate_additional_point_list(5, test_general_parameter.output_range.p_min_max_list)
    w_point_list: list[list[float]] = generate_weighting_point_list(5)
    # Set additional user point parameters
    test_general_parameter.sampling.v1_additional_user_point_list = v1_additional_point_list[test_index]
    test_general_parameter.sampling.v2_additional_user_point_list = v2_additional_point_list[test_index]
    test_general_parameter.sampling.p_additional_user_point_list = p_additional_point_list[test_index]
    test_general_parameter.sampling.additional_user_weighting_point_list = w_point_list[test_index]

    string_test_values: list[str] = ["A", "b9", "Test123", "x_Y_z890"]
    str_test_len = len(string_test_values)

    # Create the instance
    test_object: ClassUnderTest = ClassUnderTest()

    # Initialize the second test parameter
    test_string_circuit = string_test_values[(test_index + 1) % str_test_len]

    # Convert general parameter class to a dict
    test_general_parameter_dict = test_general_parameter.model_dump()
    # Convert circuit parameter class to a dict
    test_circuit_parameter_dict = test_circuit_parameter.model_dump()

    # Create path
    with tempfile.TemporaryDirectory() as tmpdir:
        # Assemble project directory path
        project_directory = os.path.join(tmpdir, string_test_values[test_index % str_test_len])
        # Init study and path information
        test_object.init_study_information(test_string_circuit, project_directory,
                                           string_test_values[(test_index + 1) % str_test_len],
                                           CalcModeEnum.new_mode)

        # Check if no error is expected
        if not is_error:
            # Load the data from dict
            test_object.load_and_verify_general_parameters(test_general_parameter_dict)
            test_object.load_and_verify_circuit_parameters(test_circuit_parameter_dict)
            # Perform the test
            is_initialized = test_object.initialize_circuit_optimization()

            assert test_object._dab_config is not None

            # Check valid result
            assert test_object._dab_config.design_space.f_s_min_max_list == test_circuit_parameter.design_space.f_s_min_max_list
            assert test_object._dab_config.design_space.l_1_min_max_list == test_circuit_parameter.design_space.l_1_min_max_list
            assert test_object._dab_config.design_space.l_2__min_max_list == test_circuit_parameter.design_space.l_2__min_max_list
            assert test_object._dab_config.design_space.l_s_min_max_list == test_circuit_parameter.design_space.l_s_min_max_list
            assert test_object._dab_config.design_space.n_min_max_list == test_circuit_parameter.design_space.n_min_max_list
            assert test_object._dab_config.design_space.transistor_1_name_list == test_circuit_parameter.design_space.transistor_1_name_list
            assert test_object._dab_config.design_space.transistor_2_name_list == test_circuit_parameter.design_space.transistor_2_name_list
            assert test_object._dab_config.design_space.c_par_1 == test_circuit_parameter.design_space.c_par_1
            assert test_object._dab_config.design_space.c_par_2 == test_circuit_parameter.design_space.c_par_2
            assert test_object._dab_config.output_range.v1_min_max_list == test_general_parameter.output_range.v1_min_max_list
            assert test_object._dab_config.output_range.v2_min_max_list == test_general_parameter.output_range.v2_min_max_list
            assert test_object._dab_config.output_range.p_min_max_list == test_general_parameter.output_range.p_min_max_list
            assert test_object._dab_config.sampling.sampling_method == test_general_parameter.sampling.sampling_method
            assert test_object._dab_config.sampling.sampling_points == test_general_parameter.sampling.sampling_points
            assert test_object._dab_config.sampling.v1_additional_user_point_list == test_general_parameter.sampling.v1_additional_user_point_list
            assert test_object._dab_config.sampling.v2_additional_user_point_list == test_general_parameter.sampling.v2_additional_user_point_list
            assert test_object._dab_config.sampling.p_additional_user_point_list == test_general_parameter.sampling.p_additional_user_point_list
            assert test_object._dab_config.filter.number_filtered_designs == test_circuit_parameter.filter_distance.number_filtered_designs
            assert test_object._dab_config.filter.difference_percentage == test_circuit_parameter.filter_distance.difference_percentage
            assert test_object._dab_config.project_directory == project_directory
            assert test_object._dab_config.circuit_study_name == test_string_circuit
            assert is_initialized
        else:
            with pytest.raises(ValueError) as error_message:
                # Perform the test
                is_initialized = test_object.initialize_circuit_optimization()
            assert "Serious programming error 'multiple allocation failure'. Please write an issue!" in str(error_message.value)
