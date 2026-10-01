"""Unit tests for sbc datasets."""

# python libraries
import logging
from enum import Enum

# 3rd party libraries
import pytest
import numpy as np

# own libraries
from pcdt.topology.sbc.sbc_datasets import HandleTransistorDto
import pcdt.topology.sbc.sbc_datasets_dtos as d_dtos
import transistordatabase as tdb

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

#########################################################################################################
# test of class HandleTransistorDto: tdb_to_transistor_dto
#########################################################################################################

@pytest.mark.parametrize(
    "transistor_name, expected_exception, expected_message_id",
    [
        # Invalid transistor names
        ("invalid_transistor_name", ValueError, 1),
        # Valid transistor names
        ("CREE_C3M0065100J", None, 0),
    ]
)
def test_tdb_to_transistor_dto(transistor_name: str, expected_exception: type,
                               expected_message_id: int) -> None:
    """
    Unit test to check the loading transistor out of transistor database.

    :param transistor_name: Name of the transistor to load
    :type  transistor_name: str
    :param expected_exception: Indicated, if the call causes an error
    :type  expected_exception: type
    :param expected_message_id: expected message IDs in a list
    :type  expected_message_id: int
    """
    info_message: list[str] = ["",
                               "Transistor type: invalid_transistor_name is of registered in transistor database!",
                               f"Transistor 1: {transistor_name} is of non-allowed type. "
                               f"Allowed types are MOSFET, SiC-MOSFET."]

    if expected_exception is not None:
        with pytest.raises(expected_exception) as error_message:
            HandleTransistorDto.tdb_to_transistor_dto(transistor_name)
        assert info_message[expected_message_id] in str(error_message.value)
    else:
        HandleTransistorDto.tdb_to_transistor_dto(transistor_name)

#########################################################################################################
# test of class HandleTransistorDto: calculate_2D_grid
#########################################################################################################

# Test with following curves y=voltage, x=current
# y         1	2	3	4	5	x-values
# 0	    	0   0   0       0	Extended curve
# 20		2				10  Curve 1
# 40			16	36		    Curve 2
#
# y    	0   1	2	3	4	5	x-values
# 0	    0 	0   0   0  (0)  0	Extended curve
# 20	0	2	4	6  (8)  10	Curve 1 (After interpolation)
# 40	0	8	16	36 (56) 76	Curve 2 (After interpolation)
#
# y  Expected values	at y
# 10	0	1	2	3	4	5
# 30	0	5	10	21  32	43

@pytest.mark.parametrize(
    "expected_exception, expected_message_id",
    [   # Invalid values
        # Valid values
        (None, 0),
    ]
)
def test_calculate_2D_grid(expected_exception: type, expected_message_id: int) -> None:
    """
    Unit test to check interpolation method.

    :param expected_exception: Indicated, if the call causes an error
    :type  expected_exception: type
    :param expected_message_id: expected message IDs in a list
    :type  expected_message_id: int
    """
    # Create curves
    curve1 = np.array([[1, 5], [2, 10]])
    curve2 = np.array([[2, 3], [16, 36]])

    set1_arg = {"dataset_type": "graph_i_e", "t_j": 25, "v_supply": 20, "v_g": 15, "graph_i_e": curve1}
    set2_arg = {"dataset_type": "graph_i_e", "t_j": 25, "v_supply": 40, "v_g": 15, "graph_i_e": curve2}

    switch_set1: tdb.SwitchEnergyData = tdb.SwitchEnergyData(set1_arg)
    switch_set2: tdb.SwitchEnergyData = tdb.SwitchEnergyData(set2_arg)

    # Initialize expected result
    exp_result: d_dtos.LossDataGrid = d_dtos.LossDataGrid(voltage_parameter=np.array([0, 20, 40]),
                                                          loss_data=np.array([[0, 0, 0, 0, 0],
                                                                              [0, 2, 4, 6, 10],
                                                                              [0, 8, 16, 36, 76]]),
                                                          current_data=np.array([0, 1, 2, 3, 5]))
    # Create loss list from valid sets
    switch_loss_list: list[tdb.SwitchEnergyData] = [switch_set1, switch_set2]

    info_message: list[str] = ["",
                               "'NoneType' object has no attribute 'type'",
                               "Transistor 1: is of non-allowed type. "
                               "Allowed types are MOSFET, SiC-MOSFET."]

    if expected_exception is not None:
        with pytest.raises(expected_exception) as error_message:
            # result_loss_data: d_dtos.LossDataGrid = HandleTransistorDto.calculate_2D_grid(switch_loss_invalid_list)
            pass
        assert info_message[expected_message_id] in str(error_message.value)
    else:
        result_loss_data: d_dtos.LossDataGrid = HandleTransistorDto.calculate_2D_grid(switch_loss_list)
        assert np.array_equal(result_loss_data.voltage_parameter, exp_result.voltage_parameter)
        assert np.array_equal(result_loss_data.loss_data, exp_result.loss_data)
        assert np.array_equal(result_loss_data.current_data, exp_result.current_data)


#########################################################################################################
# test of class HandleTransistorDto: transistor_conduction_loss
#########################################################################################################

#########################################################################################################
# test of class HandleTransistorDto: transistor_switch_loss
#########################################################################################################

#########################################################################################################
# test of class HandleSbcDto: init_config
#########################################################################################################

#########################################################################################################
# test of class HandleSbcDto: get_c_oss_from_tdb
#########################################################################################################

#########################################################################################################
# test of class HandleSbcDto: _integrate_c_oss
#########################################################################################################

#########################################################################################################
# test of class HandleSbcDto: save
#########################################################################################################

#########################################################################################################
# test of class HandleSbcDto: load_from_file
#########################################################################################################

#########################################################################################################
# test of class HandleSbcDto: get_max_peak_waveform_inductor
#########################################################################################################

#########################################################################################################
# test of class HandleSbcDto: add_inductor_results
#########################################################################################################
