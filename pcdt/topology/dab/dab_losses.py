"""Calculate the transistor losses."""
# python libraries
import logging

# 3rd party libraries
import numpy as np
import transistordatabase as tdb

# own modules
from pcdt.topology.dab import dab_datasets_dtos as d_dtos
from pcdt.topology.dab.dab_functions_waveforms import double_waveform

logger = logging.getLogger(__name__)

def transistor_conduction_loss(transistor_rms_current: float, transistor_dto: d_dtos.TransistorDTO) -> np.ndarray:
    """
    Calculate the transistor conduction losses.

    :param transistor_rms_current: transistor RMS current in A
    :type transistor_rms_current: float
    :param transistor_dto: transistor DTO (Data transfer object)
    :type transistor_dto: pcdt.TransistorDTO
    :return: transistor conduction loss in W
    :rtype: float
    """
    return transistor_dto.r_channel * transistor_rms_current ** 2


def calculate_turn_off_currents_single_operating_point(i_lc_full_time_current_waveform: np.ndarray, i_hf_full_time_current_waveform: np.ndarray,
                                                       tau_rad: np.ndarray) -> tuple[float, float]:
    """
    Transistor turn-off current estimation.

    The i_lc current (in the commutation inductance) is needed to estimate the switching point of the corresponding bridge.
    This function is independent of the bridge, so there is no _1 or _2 variable name index.

    :param i_lc_full_time_current_waveform: i_lc1 or i_lc2 in format [[time], [current]]
    :type i_lc_full_time_current_waveform: np.ndarray
    :param i_hf_full_time_current_waveform: i_hf1 or i_hf2 in format [[time], [current]]
    :type i_hf_full_time_current_waveform: np.ndarray
    :param tau_rad: control parameter tau in radiant to distinguish for triangular / trapezoidal current
    :type tau_rad: float
    :return: turn-off currents: current_switching_1, current_switching_2
    :rtype: tuple[float, float]
    """
    # figure out the maximum current points to determine the first switching event
    indexes_ilc_max = np.where(i_lc_full_time_current_waveform[1] == np.max(i_lc_full_time_current_waveform[1]))[0]
    first_switching_index = indexes_ilc_max[0]

    # always double the waveforms, as nothing can go wrong. This avoids a few if/else statements.
    i_lc_full_time_current_waveform_doubled = double_waveform(time_current_waveform=i_lc_full_time_current_waveform)
    i_hf_full_time_current_waveform_doubled = double_waveform(time_current_waveform=i_hf_full_time_current_waveform)

    # consider the first switching event
    if first_switching_index == 0:
        logger.debug("Curve at very beginning. Shift index.")
        # curve is at the very beginning. Integration will fail due to the shift.
        index_switching = len(i_lc_full_time_current_waveform_doubled[0]) - 1
        current_switching_1 = i_hf_full_time_current_waveform_doubled[1][index_switching]
    else:
        current_switching_1 = i_hf_full_time_current_waveform_doubled[1][first_switching_index]

    # in case of tau_rad is not 180°, two maximum in i_lc appear (three different voltage levels on the bridge output)
    # but i_hf has two different current values at the switching points. The integration must be done on the second switching point also.
    # Therefore, it must be checked which dead time is greater.
    current_switching_2 = current_switching_1
    if tau_rad != np.pi:
        # Note: It is important to take the second index, not the last.
        # In case of taking the last index, the current could be the same as the first (in case of the first index is the maximum),
        # as the waveform is symmetric. The second needs to be taken!
        # Update: It is important to choose the last index following after the first of the doubled waveform!
        indexes_ilc_doubled_max = np.where(i_lc_full_time_current_waveform_doubled[1] == np.max(i_lc_full_time_current_waveform_doubled[1]))[0]

        # get the index beginning from zero
        high = False
        for count in range(1, len(indexes_ilc_doubled_max)):
            if indexes_ilc_doubled_max[count - 1] == count - 1:
                high = True
            else:
                if high:
                    last_high_index = count
                else:
                    high = False

        second_switching_index = indexes_ilc_doubled_max[count]

        # consider the first switching event
        if second_switching_index == 0:
            logger.debug("Curve at very beginning. Shift index.")
            # curve is at the very beginning. Integration will fail due to the shift.
            index_switching = len(i_lc_full_time_current_waveform_doubled[0]) - 1
            current_switching_2 = i_hf_full_time_current_waveform_doubled[1][index_switching]
        else:
            current_switching_2 = i_hf_full_time_current_waveform_doubled[1][second_switching_index]

    return current_switching_1, current_switching_2

def transistor_turn_off_loss(transistor_turn_off_current: float, transistor_dto: d_dtos.TransistorDTO, v_dc: np.ndarray,
                             temperature: np.ndarray, f_s: float, c_par: float) -> float:
    """
    Calculate the transistor turn-off losses.

    In case of measurement data is available, the measurement fit factors are used. In case of no measurement data available, the
    data sheet switching loss curves are considered.

    :param transistor_turn_off_current: Current in A at turn-off
    :type transistor_turn_off_current: float
    :param transistor_dto: transistor DTO (Data transfer object)
    :type transistor_dto: pcdt.TransistorDTO
    :param v_dc: dc-link voltage in V
    :type v_dc: np.ndarray
    :param f_s: switching frequency in Hz
    :type f_s: float
    :param c_par: parallel PCB capacitance in F
    :type c_par: float
    :param temperature: Temperature in °C
    :type temperature: float
    :return: Switching losses in W
    :rtype: float
    """
    # in case of measurement data is available, use this
    if transistor_dto.turn_off_fit_factors is not None:
        logger.info(f"{transistor_dto.name} turn-on loss data source: fitted measurement data.")
        current_vec = np.linspace(0, transistor_dto.turn_off_fit_factors.current_max)
        energy_vec = tdb.Transistor.fit_function(
            (current_vec, v_dc, temperature), transistor_dto.turn_off_fit_factors.a_current,
            transistor_dto.turn_off_fit_factors.b_current, transistor_dto.turn_off_fit_factors.c_current, transistor_dto.turn_off_fit_factors.voltage_factor,
            transistor_dto.turn_off_fit_factors.voltage_exponent, transistor_dto.turn_off_fit_factors.ct_0, transistor_dto.turn_off_fit_factors.ct_1,
            transistor_dto.turn_off_fit_factors.ct_2)

        # correct data with the energy in c_oss
        energy_in_capacitance_at_dpt_voltage = np.interp(v_dc, transistor_dto.v_oss, transistor_dto.e_oss)

        energy_vec_corrected = energy_vec - energy_in_capacitance_at_dpt_voltage
    else:
        logger.info(f"{transistor_dto.name} turn-off loss data source: data sheet data")
        # use data sheet data, scale curve according to the dc link voltage
        current_vec = transistor_dto.turn_off_current_vec
        energy_vec_corrected = transistor_dto.turn_off_energy_vec * v_dc / transistor_dto.turn_off_voltage

    # clip unrealistic values smaller then zero
    turn_off_energy_corrected_energy_voltage_vec = energy_vec_corrected.clip(min=0)

    # interpolate the switching energy
    turn_off_energy = np.interp(np.abs(transistor_turn_off_current), current_vec, turn_off_energy_corrected_energy_voltage_vec)

    # estimate the switching loss
    turn_off_power = turn_off_energy * f_s

    return float(turn_off_power)
