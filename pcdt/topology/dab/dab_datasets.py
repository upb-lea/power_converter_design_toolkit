"""Describe the dataclasses how to store the input parameters and results."""

# python libraries
import os
import datetime
import logging
import pickle

# 3rd party libraries
import numpy as np
import transistordatabase as tdb
from matplotlib import pyplot as plt
from scipy.integrate import cumulative_trapezoid

# own libraries
from pcdt.constant_path import GECKO_PATH
from pcdt.topology.dab import dab_datasets_dtos as d_dtos
from pcdt.topology.dab import dab_functions_waveforms as d_waveforms
from pcdt.topology.dab import dab_mod_zvs as mod
from pcdt.topology.dab import dab_currents as pcdt_currents
from pcdt.topology.dab import dab_geckosimulation as pcdt_gecko
from pcdt.topology.dab import dab_losses as pcdt_loss
from pcdt.topology.dab.dab_circuit_topology_dtos import CircuitSampling
from pcdt.topology.dab.dab_functions_waveforms import (full_current_waveform_from_currents, full_angle_waveform_from_angles,
                                                       full_time_waveforms_from_angles_currents, double_waveform)
from pcdt.components.component_dtos import (CircuitThermal, CapacitorRequirements, InductorRequirements, TransformerRequirements,
                                            InductorResults, StackedTransformerResults, ComponentCooling)
from pcdt.components.heat_sink_optimization import ThermalCalcSupport
from pcdt.topology.dab.dab_losses import transistor_turn_off_loss, calculate_turn_off_currents_single_operating_point

logger = logging.getLogger(__name__)

class HandleDabDto:
    """Class to handle the DabDTO, e.g. save and load the files."""

    c_oss_storage_directory: str = ""

    @staticmethod
    def set_c_oss_storage_directory(act_c_oss_storage_directory: str) -> bool:
        """
        Set the transistor database path.

        :param act_c_oss_storage_directory: Directory for storage of transistor switch loss
        :type  act_c_oss_storage_directory: str
        :return: True, if the path exists, else false
        :rtype: bool
        """
        # Variable declaration
        valid_directory_flag: bool = False

        # Get absolute path name of the directory
        abs_dir = os.path.abspath(act_c_oss_storage_directory)

        # check path
        if os.path.exists(abs_dir):
            HandleDabDto.c_oss_storage_directory = abs_dir
            logger.debug(f"Directory {act_c_oss_storage_directory} exists.")
            # Set return value to True
            valid_directory_flag = True
        else:
            logger.warning(f"Directory {abs_dir} does not exist!")

        return valid_directory_flag

    @staticmethod
    def init_config(circuit_id: str, mesh_v1: np.ndarray, mesh_v2: np.ndarray, mesh_p: np.ndarray,
                    sampling: CircuitSampling, n: float, ls: float, lc1: float, lc2: float, fs: float,
                    transistor_dto_1: d_dtos.TransistorDTO, transistor_dto_2: d_dtos.TransistorDTO,
                    lossfilepath: str, c_par_1: float, c_par_2: float, t_dead_1_max: float, t_dead_2_max: float,
                    mesh_weights: np.ndarray) -> d_dtos.DabCircuitDTO:
        """
        Initialize the DAB structure.

        :param circuit_id: circuit ID
        :type circuit_id: str
        :param mesh_v1: mesh or hypercube sampling for v1
        :type mesh_v1: np.ndarray
        :param mesh_v2: mesh or hypercube sampling for v2
        :type mesh_v2: np.ndarray
        :param mesh_p: mesh or hypercube sampling for p
        :type mesh_p: np.ndarray
        :param sampling: Sampling parameters
        :type sampling: d_dtos.Sampling
        :param n: transformer transfer ratio
        :type n: float
        :param ls: series inductance
        :type ls: float
        :param lc1: Commutation inductance Lc1
        :type lc1: float
        :param lc2: Commutation inductance Lc2
        :type lc2: float
        :param fs: Switching frequency
        :type fs: float
        :param mesh_weights: weighted operating points
        :type mesh_weights: np.ndarray
        :param transistor_dto_1: Transistor DTO for transistor bridge 1. Must match with transistordatabase available transistors.
        :type transistor_dto_1: TransistorDTO
        :param transistor_dto_2: Transistor DTO for transistor bridge 2. Must match with transistordatabase available transistors.
        :type transistor_dto_2: TransistorDTO
        :param lossfilepath: Path to store the calculated transistor loss
        :type  lossfilepath: str
        :param c_par_1: Parasitic PCB capacitance per transistor footprint of bridge 1
        :type c_par_1: float
        :param c_par_2: Parasitic PCB capacitance per transistor footprint of bridge 2
        :type c_par_2: float
        :param t_dead_1_max: maximum dead time for bridge 1
        :type t_dead_1_max: float
        :param t_dead_2_max: maximum dead time for bridge 2
        :type t_dead_2_max: float
        :return: Configuration data for the actual design
        :rtype:  d_dtos.DabCircuitDTO
        """
        input_configuration = d_dtos.CircuitConfig(mesh_v1=mesh_v1,
                                                   mesh_v2=mesh_v2,
                                                   mesh_p=mesh_p,
                                                   mesh_weights=mesh_weights,
                                                   sampling=sampling,
                                                   n=np.array(n),
                                                   Ls=np.array(ls),
                                                   Lc1=np.array(lc1),
                                                   Lc2=np.array(lc2),
                                                   fs=np.array(fs),
                                                   transistor_dto_1=transistor_dto_1,
                                                   transistor_dto_2=transistor_dto_2,
                                                   lossfilepath=lossfilepath,
                                                   c_par_1=c_par_1,
                                                   c_par_2=c_par_2,
                                                   t_dead_1_max=t_dead_1_max,
                                                   t_dead_2_max=t_dead_2_max
                                                   )
        calc_config = HandleDabDto.calculate_from_configuration(config=input_configuration)
        modulation_parameters = HandleDabDto.calculate_modulation(input_configuration, calc_config)

        i_l_s_rms, i_l_1_rms, i_l_2_rms, angles_rad_sorted, i_l_s_sorted, i_l_1_sorted, i_l_2_sorted, angles_rad_unsorted = pcdt_currents.calc_rms_currents(
            input_configuration, modulation_parameters)

        i_hf_1_rms, i_hf_2_rms, i_hf_1_sorted, i_hf_2_sorted = pcdt_currents.calc_hf_currents(
            angles_rad_sorted, i_l_s_sorted, i_l_1_sorted, i_l_2_sorted, input_configuration.n)

        i_m1_rms = pcdt_currents.calc_transistor_rms_currents(i_hf_1_rms)
        i_m2_rms = pcdt_currents.calc_transistor_rms_currents(i_hf_2_rms)

        calc_currents = d_dtos.CalcCurrents(**{'i_l_s_rms': i_l_s_rms, 'i_l_1_rms': i_l_1_rms, 'i_l_2_rms': i_l_2_rms, 'angles_rad_sorted': angles_rad_sorted,
                                               'angles_rad_unsorted': angles_rad_unsorted, 'i_l_s_sorted': i_l_s_sorted, 'i_l_1_sorted': i_l_1_sorted,
                                               'i_l_2_sorted': i_l_2_sorted, 'i_hf_1_rms': i_hf_1_rms, 'i_hf_2_rms': i_hf_2_rms,
                                               'i_m1_rms': i_m1_rms, 'i_m2_rms': i_m2_rms, 'i_hf_1_sorted': i_hf_1_sorted, 'i_hf_2_sorted': i_hf_2_sorted})
        p_m1_cond = pcdt_loss.transistor_conduction_loss(i_m1_rms, transistor_dto_1)
        p_m2_cond = pcdt_loss.transistor_conduction_loss(i_m2_rms, transistor_dto_2)

        calc_losses = d_dtos.CalcLosses(**{'p_m1_conduction': p_m1_cond,
                                           'p_m2_conduction': p_m2_cond,
                                           'p_dab_conduction': 4 * (p_m1_cond + p_m2_cond),
                                           # switching loss
                                           'i_turn_off_1a': np.nan,
                                           'i_turn_off_1b': np.nan,
                                           'i_turn_off_2a': np.nan,
                                           'i_turn_off_2b': np.nan,
                                           'p_turn_off_1a': np.nan,
                                           'p_turn_off_1b': np.nan,
                                           'p_turn_off_2a': np.nan,
                                           'p_turn_off_2b': np.nan,
                                           'p_m1_switching': np.nan,
                                           'p_m2_switching': np.nan,
                                           'p_dab_switching': np.nan,
                                           # total loss
                                           'p_dab': np.nan})

        gecko_additional_params = d_dtos.GeckoAdditionalParameters(
            timestep=1e-9, number_sim_periods=2, timestep_pre=25e-9, number_pre_sim_periods=0,
            simfilepath=os.path.join(GECKO_PATH, 'DAB_MOSFET_Modulation_v8.ipes'),
            lossfilepath=lossfilepath)

        dab_dto = d_dtos.DabCircuitDTO(
            circuit_id=circuit_id,
            timestamp=None,
            metadata=None,
            input_config=input_configuration,
            calc_config=calc_config,
            calc_modulation=modulation_parameters,
            calc_currents=calc_currents,
            calc_dead_time=None,
            calc_losses=calc_losses,
            component_requirements=None,
            gecko_additional_params=gecko_additional_params,
            gecko_results=None,
            gecko_waveforms=None,
            capacitor_1_results=None,
            capacitor_2_results=None,
            inductor_results=None,
            stacked_transformer_results=None,
            circuit_thermal=None,
        )

        # add minimum dead time in case of design is useful
        if not ((np.any(np.isnan(dab_dto.calc_modulation.phi)) or np.any(np.isnan(dab_dto.calc_modulation.tau1)) \
                or np.any(np.isnan(dab_dto.calc_modulation.tau2)))):
            dab_dto = HandleDabDto.add_calculated_dead_time(dab_dto)

        return dab_dto

    @staticmethod
    def add_gecko_simulation_results(dab_dto: d_dtos.DabCircuitDTO, get_waveforms: bool = False) -> d_dtos.DabCircuitDTO:
        """
        Add GeckoCIRCUITS simulation results to the given DTO.

        :param dab_dto: DabDTO
        :param get_waveforms: Read back GeckoCIRCUITS simulation waveforms (high memory consumption). Default to False.
        :type get_waveforms: bool
        :return: DabDTO
        """
        if dab_dto.calc_dead_time is None:
            raise ValueError("Incomplete calculation as dead time is missing.")

        gecko_results, gecko_waveforms = pcdt_gecko.start_gecko_simulation(
            mesh_v1=dab_dto.input_config.mesh_v1, mesh_v2=dab_dto.input_config.mesh_v2,
            mesh_p=dab_dto.input_config.mesh_p, mod_phi=dab_dto.calc_modulation.phi,
            mod_tau1=dab_dto.calc_modulation.tau1, mod_tau2=dab_dto.calc_modulation.tau2,
            mesh_t_dead1=dab_dto.calc_dead_time.t_dead_1, mesh_t_dead2=dab_dto.calc_dead_time.t_dead_2,
            fs=dab_dto.input_config.fs, ls=dab_dto.input_config.Ls, lc1=dab_dto.input_config.Lc1,
            lc2=dab_dto.input_config.Lc2, n=dab_dto.input_config.n,
            t_j_1=dab_dto.calc_config.t_j_1, t_j_2=dab_dto.calc_config.t_j_2,
            simfilepath=dab_dto.gecko_additional_params.simfilepath,
            lossfilepath=dab_dto.gecko_additional_params.lossfilepath,
            timestep=dab_dto.gecko_additional_params.timestep,
            number_sim_periods=dab_dto.gecko_additional_params.number_sim_periods,
            timestep_pre=dab_dto.gecko_additional_params.timestep_pre,
            number_pre_sim_periods=dab_dto.gecko_additional_params.number_pre_sim_periods, geckoport=43036,
            c_par_1=dab_dto.input_config.c_par_1, c_par_2=dab_dto.input_config.c_par_2,
            transistor_1_name=dab_dto.input_config.transistor_dto_1.name,
            transistor_2_name=dab_dto.input_config.transistor_dto_2.name, get_waveforms=get_waveforms,
            i_ls_start=dab_dto.calc_currents.i_l_s_sorted[0],
            i_lc1_start=dab_dto.calc_currents.i_l_1_sorted[0],
            i_lc2_start=dab_dto.calc_currents.i_l_2_sorted[0])

        # add GeckoCIRCUITS simulation results to the result DTO.
        dab_dto.gecko_results = d_dtos.GeckoResults(**gecko_results)

        dab_dto.gecko_waveforms = d_dtos.GeckoWaveforms(**gecko_waveforms)
        return dab_dto

    @staticmethod
    def calculate_from_configuration(config: d_dtos.CircuitConfig) -> d_dtos.CalcFromCircuitConfig:
        """
        Calculate logical parameters which can be calculated from the input parameters.

        :param config: DAB configuration
        :type config: CircuitConfig
        :return: CalcFromConfig
        :rtype: CalcFromCircuitConfig
        """
        Lc2_ = config.Lc2 * config.n ** 2

        calc_from_config = d_dtos.CalcFromCircuitConfig(
            Lc2_=Lc2_,
            t_j_1=config.transistor_dto_1.t_j_max_op,
            t_j_2=config.transistor_dto_2.t_j_max_op,
            c_oss_par_1=config.transistor_dto_1.c_oss + config.c_par_1,
            c_oss_par_2=config.transistor_dto_2.c_oss + config.c_par_2,
            c_oss_1=config.transistor_dto_1.c_oss,
            c_oss_2=config.transistor_dto_2.c_oss,
            q_oss_1=config.transistor_dto_1.q_oss,
            q_oss_2=config.transistor_dto_2.q_oss
        )

        return calc_from_config

    @staticmethod
    def calculate_modulation(config: d_dtos.CircuitConfig, calc_config: d_dtos.CalcFromCircuitConfig) -> d_dtos.CalcModulation:
        """
        Calculate the modulation parameters like phi, tau1, tau, ...

        :param config: DAB input configuration
        :param calc_config: calculated parameters from the input configuration
        :return: Modulation parameters.
        """
        result_dict = mod.calc_modulation_params(config.n, config.Ls, config.Lc1, config.Lc2, config.fs, c_oss_1=calc_config.c_oss_par_1,
                                                 c_oss_2=calc_config.c_oss_par_2, v1=config.mesh_v1, v2=config.mesh_v2, power=config.mesh_p)

        return d_dtos.CalcModulation(**result_dict)

    @staticmethod
    def add_calculated_dead_time(dab_calc: d_dtos.DabCircuitDTO) -> d_dtos.DabCircuitDTO:
        """
        Add the required minimum dead time for bridge 1 and bridge 2 to the DabCircuitDTO.

        :param dab_calc: DAB circuit DTO
        :type dab_calc: d_dtos.DabCircuitDTO
        """
        # Calculate the required dead time
        t_dead_1 = np.full_like(dab_calc.calc_modulation.phi, np.nan)
        t_dead_2 = np.full_like(dab_calc.calc_modulation.phi, np.nan)
        is_zvs_1 = np.full_like(dab_calc.calc_modulation.phi, False)
        is_zvs_2 = np.full_like(dab_calc.calc_modulation.phi, False)
        for vec_vvp in np.ndindex(dab_calc.calc_modulation.phi.shape):
            i_lc1_time_current = np.asarray(full_time_waveforms_from_angles_currents(
                dab_calc.input_config.fs, np.transpose(dab_calc.calc_currents.angles_rad_sorted, (1, 2, 3, 0))[vec_vvp],
                np.transpose(dab_calc.calc_currents.i_l_1_sorted, (1, 2, 3, 0))[vec_vvp]))
            i_hf1_time_current = np.asarray(full_time_waveforms_from_angles_currents(
                dab_calc.input_config.fs, np.transpose(dab_calc.calc_currents.angles_rad_sorted, (1, 2, 3, 0))[vec_vvp],
                np.transpose(dab_calc.calc_currents.i_hf_1_sorted, (1, 2, 3, 0))[vec_vvp]))
            i_lc2_time_current = np.asarray(full_time_waveforms_from_angles_currents(
                dab_calc.input_config.fs, np.transpose(dab_calc.calc_currents.angles_rad_sorted, (1, 2, 3, 0))[vec_vvp],
                np.transpose(dab_calc.calc_currents.i_l_2_sorted, (1, 2, 3, 0))[vec_vvp]))
            i_hf2_time_current = np.asarray(full_time_waveforms_from_angles_currents(
                dab_calc.input_config.fs, np.transpose(dab_calc.calc_currents.angles_rad_sorted, (1, 2, 3, 0))[vec_vvp],
                np.transpose(dab_calc.calc_currents.i_hf_2_sorted, (1, 2, 3, 0))[vec_vvp]))
            is_zvs_1[vec_vvp], t_dead_1[vec_vvp] = HandleDabDto.calculate_dead_time(
                dab_calc.calc_modulation.q_ab_req1[vec_vvp] / 2, i_lc1_time_current, i_hf1_time_current, dab_calc.calc_modulation.tau1[vec_vvp])
            is_zvs_2[vec_vvp], t_dead_2[vec_vvp] = HandleDabDto.calculate_dead_time(
                dab_calc.calc_modulation.q_ab_req2[vec_vvp] / 2, i_lc2_time_current, i_hf2_time_current, dab_calc.calc_modulation.tau2[vec_vvp], is_plot=False)

        is_zvs = np.logical_and(is_zvs_1, is_zvs_2)
        zvs_coverage = np.count_nonzero(is_zvs) / np.size(is_zvs)

        dab_calc.calc_dead_time = d_dtos.CalcDeadTimes(t_dead_1=t_dead_1, t_dead_2=t_dead_2, is_zvs_1=is_zvs_1, is_zvs_2=is_zvs_2, zvs_coverage=zvs_coverage)
        return dab_calc

    @staticmethod
    def add_transistor_turn_off_currents_losses(dto: d_dtos.DabCircuitDTO) -> d_dtos.DabCircuitDTO:
        """
        Add the turn-off currents for the MOSFETs in bridge 1 and bridge 2 with the legs a and b.

        :param dto: DAB circuit DTO
        :type dto: d_dtos.DabCircuitDTO
        :return: DAB circuit DTO
        :rtype: d_dtos.DabCircuitDTO
        """
        # Calculate the required dead time
        i_turn_off_1a = np.full_like(dto.calc_modulation.phi, np.nan)
        i_turn_off_2a = np.full_like(dto.calc_modulation.phi, np.nan)
        i_turn_off_1b = np.full_like(dto.calc_modulation.phi, np.nan)
        i_turn_off_2b = np.full_like(dto.calc_modulation.phi, np.nan)
        p_turn_off_1a = np.full_like(dto.calc_modulation.phi, np.nan)
        p_turn_off_2a = np.full_like(dto.calc_modulation.phi, np.nan)
        p_turn_off_1b = np.full_like(dto.calc_modulation.phi, np.nan)
        p_turn_off_2b = np.full_like(dto.calc_modulation.phi, np.nan)
        for vec_vvp in np.ndindex(dto.calc_modulation.phi.shape):
            i_lc1_time_current = np.asarray(full_time_waveforms_from_angles_currents(
                dto.input_config.fs, np.transpose(dto.calc_currents.angles_rad_sorted, (1, 2, 3, 0))[vec_vvp],
                np.transpose(dto.calc_currents.i_l_1_sorted, (1, 2, 3, 0))[vec_vvp]))
            i_hf1_time_current = np.asarray(full_time_waveforms_from_angles_currents(
                dto.input_config.fs, np.transpose(dto.calc_currents.angles_rad_sorted, (1, 2, 3, 0))[vec_vvp],
                np.transpose(dto.calc_currents.i_hf_1_sorted, (1, 2, 3, 0))[vec_vvp]))
            i_lc2_time_current = np.asarray(full_time_waveforms_from_angles_currents(
                dto.input_config.fs, np.transpose(dto.calc_currents.angles_rad_sorted, (1, 2, 3, 0))[vec_vvp],
                np.transpose(dto.calc_currents.i_l_2_sorted, (1, 2, 3, 0))[vec_vvp]))
            i_hf2_time_current = np.asarray(full_time_waveforms_from_angles_currents(
                dto.input_config.fs, np.transpose(dto.calc_currents.angles_rad_sorted, (1, 2, 3, 0))[vec_vvp],
                np.transpose(dto.calc_currents.i_hf_2_sorted, (1, 2, 3, 0))[vec_vvp]))
            i_turn_off_1a[vec_vvp], i_turn_off_1b[vec_vvp] = calculate_turn_off_currents_single_operating_point(i_lc1_time_current, i_hf1_time_current,
                                                                                                                dto.calc_modulation.tau1[vec_vvp])
            i_turn_off_2a[vec_vvp], i_turn_off_2b[vec_vvp] = calculate_turn_off_currents_single_operating_point(i_lc2_time_current, i_hf2_time_current,
                                                                                                                dto.calc_modulation.tau2[vec_vvp])
            p_turn_off_1a[vec_vvp] = transistor_turn_off_loss(
                float(i_turn_off_1a[vec_vvp]), dto.input_config.transistor_dto_1, dto.input_config.mesh_v1[vec_vvp],
                dto.input_config.transistor_dto_1.t_j_max_op - 25, dto.input_config.fs, dto.input_config.c_par_1)
            p_turn_off_1b[vec_vvp] = transistor_turn_off_loss(
                float(i_turn_off_1b[vec_vvp]), dto.input_config.transistor_dto_1, dto.input_config.mesh_v1[vec_vvp],
                dto.input_config.transistor_dto_1.t_j_max_op - 25, dto.input_config.fs, dto.input_config.c_par_1)
            p_turn_off_2a[vec_vvp] = transistor_turn_off_loss(
                float(i_turn_off_2a[vec_vvp]), dto.input_config.transistor_dto_2, dto.input_config.mesh_v2[vec_vvp],
                dto.input_config.transistor_dto_2.t_j_max_op - 25, dto.input_config.fs, dto.input_config.c_par_2)
            p_turn_off_2b[vec_vvp] = transistor_turn_off_loss(
                float(i_turn_off_2b[vec_vvp]), dto.input_config.transistor_dto_2, dto.input_config.mesh_v2[vec_vvp],
                dto.input_config.transistor_dto_2.t_j_max_op - 25, dto.input_config.fs, dto.input_config.c_par_2)

        if dto.calc_losses is not None:  # mypy
            p_m1_cond = dto.calc_losses.p_m1_conduction
            p_m2_cond = dto.calc_losses.p_m2_conduction

            calc_losses = d_dtos.CalcLosses(**{'p_m1_conduction': p_m1_cond,
                                               'p_m2_conduction': p_m2_cond,
                                               'p_dab_conduction': 4 * (p_m1_cond + p_m2_cond),
                                               # switching loss
                                               'i_turn_off_1a': i_turn_off_1a,
                                               'i_turn_off_1b': i_turn_off_1b,
                                               'i_turn_off_2a': i_turn_off_2a,
                                               'i_turn_off_2b': i_turn_off_2b,
                                               'p_turn_off_1a': p_turn_off_1a,
                                               'p_turn_off_1b': p_turn_off_1b,
                                               'p_turn_off_2a': p_turn_off_2a,
                                               'p_turn_off_2b': p_turn_off_2b,
                                               'p_m1_switching': 2 * (p_turn_off_1a + p_turn_off_1b),
                                               'p_m2_switching': 2 * (p_turn_off_2a + p_turn_off_2b),
                                               'p_dab_switching': 2 * (p_turn_off_1a + p_turn_off_1b) + 2 * (p_turn_off_2a + p_turn_off_2b),
                                               # total loss
                                               'p_dab': 4 * (p_m1_cond + p_m2_cond) + 2 * (p_turn_off_1a + p_turn_off_1b) + \
                                               2 * (p_turn_off_2a + p_turn_off_2b)})

            dto.calc_losses = calc_losses

        return dto

    @staticmethod
    def _integrate_part_a_leftwards(q_ab_half_req: np.ndarray, time_high_resolution: np.ndarray, i_hf_high_resolution: np.ndarray,
                                    t_interp_index_switching: int, number_of_points: int, dead_time_resolution: float = 1e-9) -> tuple[bool, float]:
        """
        Integrate the current i_hf part A (from start of the dead time until the middle of the dead time) until the charge q_ab_half_req is reached.

        :param q_ab_half_req: Required charge Q_AB_required / 2 to charge/discharge for half of the DC voltage
        :type q_ab_half_req: np.ndarray
        :param time_high_resolution: time vector
        :type time_high_resolution: np.ndarray
        :param i_hf_high_resolution: i_hf vector
        :type i_hf_high_resolution: np.ndarray
        :param t_interp_index_switching: time index when the switching event happens
        :type t_interp_index_switching: int
        :param number_of_points: sampling points for the high resolution
        :type number_of_points: int
        :param dead_time_resolution: dead time resolution
        :type dead_time_resolution: float
        """
        is_zvs = True
        # integrate part A (from switching point backwards to get Q_A_req). Therefore, the array is flipped.
        part_a_shifted_time = np.linspace(0, time_high_resolution[-1], number_of_points)
        part_a_shifted_current = np.flip(np.roll(i_hf_high_resolution, -t_interp_index_switching))
        current_sign_at_switching_point = np.sign(part_a_shifted_current[0])
        for count, time_value in enumerate(part_a_shifted_time):
            current_vector_to_integrate = part_a_shifted_current[0:count]
            q_a = dead_time_resolution * np.trapezoid(current_vector_to_integrate)
            current_sign_at_dead_time = np.sign(part_a_shifted_current[count])
            dead_time_part_a = time_value
            if np.abs(q_a) >= q_ab_half_req:
                break
            elif current_sign_at_dead_time != current_sign_at_switching_point:
                is_zvs = False
                break
        return is_zvs, dead_time_part_a

    @staticmethod
    def _integrate_part_b_rightwards(q_ab_req_half: np.ndarray, time_high_resolution: np.ndarray, i_hf_high_resolution: np.ndarray,
                                     t_interp_index_switching: int, number_of_points: int, dead_time_resolution: float = 1e-9) -> tuple[bool, float]:
        """
        Integrate the current i_hf part B (from middle of the dead time until the end of the dead time) until the charge q_ab_half_req is reached.

        :param q_ab_half_req: Required charge Q_AB_required / 2 to charge/discharge for half of the DC voltage
        :type q_ab_half_req: np.ndarray
        :param time_high_resolution: time vector
        :type time_high_resolution: np.ndarray
        :param i_hf_high_resolution: i_hf vector
        :type i_hf_high_resolution: np.ndarray
        :param t_interp_index_switching: time index when the switching event happens
        :type t_interp_index_switching: int
        :param number_of_points: sampling points for the high resolution
        :type number_of_points: int
        :param dead_time_resolution: dead time resolution
        :type dead_time_resolution: float
        """
        is_zvs = True
        # integrate part B (from switching point to get Q_B_req)
        part_b_shifted_time = np.linspace(0, time_high_resolution[-1], number_of_points)
        part_b_shifted_current = np.roll(i_hf_high_resolution, -t_interp_index_switching)
        current_sign_at_switching_point = np.sign(part_b_shifted_current[0])
        for count, time_value in enumerate(part_b_shifted_time):
            current_vector_to_integrate = part_b_shifted_current[0:count]
            q_b = dead_time_resolution * np.trapezoid(current_vector_to_integrate)
            current_sign_at_dead_time = np.sign(part_b_shifted_current[count])
            dead_time_part_b = time_value
            if np.abs(q_b) >= q_ab_req_half:
                break
            elif current_sign_at_dead_time != current_sign_at_switching_point:
                is_zvs = False
                break
        return is_zvs, dead_time_part_b

    @staticmethod
    def _index_of_nearest_value(array: np.ndarray, value: float) -> int:
        """
        Search for the nearest value in an array and return its index.

        :param array: Array to search for the closest value
        :type array: np.ndarray
        :param value: value to search for (which element in array is closest)
        :type value: float
        :return: index of closest value
        """
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return int(idx)

    @staticmethod
    def calculate_dead_time(q_ab_half_req: np.ndarray, i_lc_full_time_current_waveform: np.ndarray, i_hf_full_time_current_waveform: np.ndarray,
                            tau_rad: np.ndarray, is_plot: bool = False) -> tuple[bool, float]:
        """
        Minimum dead time estimation based on required charge Q_AB_req and i_hf currents.

        The i_lc current is needed to estimate the switching point of the corresponding bridge.
        This function is independent of the bridge, so there is no _1 or _2 variable name index.
        :param q_ab_half_req: required charge in Q. We need the half of Q_AB_req -> Q_A_req or Q_B_req
        :type q_ab_half_req: float
        :param i_lc_full_time_current_waveform: i_lc1 or i_lc2 in format [[time], [current]]
        :type i_lc_full_time_current_waveform: np.ndarray
        :param i_hf_full_time_current_waveform: i_hf1 or i_hf2 in format [[time], [current]]
        :type i_hf_full_time_current_waveform: np.ndarray
        :param tau_rad: control parameter tau in radiant to distinguish for triangular / trapezoidal current
        :type tau_rad: float
        :param is_plot: True to show a plot for debugging
        :type is_plot: bool
        """
        # figure out the maximum current points to determine the first switching event
        indexes_ilc_max = np.where(i_lc_full_time_current_waveform[1] == np.max(i_lc_full_time_current_waveform[1]))[0]
        first_switching_index = indexes_ilc_max[0]

        # always double the waveforms, as nothing can go wrong. This avoids a few if/else statements.
        i_lc_full_time_current_waveform_doubled = double_waveform(time_current_waveform=i_lc_full_time_current_waveform)
        i_hf_full_time_current_waveform_doubled = double_waveform(time_current_waveform=i_hf_full_time_current_waveform)

        # generate high resolution current waveforms for integration
        # linspace is used, as it considers the end point
        dead_time_resolution = 1e-9
        number_of_points = int((i_hf_full_time_current_waveform_doubled[0][-1] - i_hf_full_time_current_waveform_doubled[0][0]) / dead_time_resolution + 1)
        time_high_resolution = np.linspace(i_hf_full_time_current_waveform_doubled[0][0], i_hf_full_time_current_waveform_doubled[0][-1], number_of_points)
        i_hf_high_resolution = np.interp(time_high_resolution, i_hf_full_time_current_waveform_doubled[0], i_hf_full_time_current_waveform_doubled[1])

        # consider the first switching event
        if first_switching_index == 0:
            logger.debug("Curve at very beginning. Shift index.")
            # curve is at the very beginning. Integration will fail due to the shift.
            index_switching = len(i_lc_full_time_current_waveform_doubled[0]) - 1
            t_switching_1 = i_lc_full_time_current_waveform_doubled[0][index_switching]
        else:
            t_switching_1 = i_lc_full_time_current_waveform_doubled[0][first_switching_index]

        t_interp_index_switching = HandleDabDto._index_of_nearest_value(time_high_resolution, t_switching_1)

        is_zvs_1_a, time_a_first_switching_event = HandleDabDto._integrate_part_a_leftwards(
            q_ab_half_req, time_high_resolution, i_hf_high_resolution, t_interp_index_switching, number_of_points, dead_time_resolution)
        is_zvs_1_b, time_b_first_switching_event = HandleDabDto._integrate_part_b_rightwards(
            q_ab_half_req, time_high_resolution, i_hf_high_resolution, t_interp_index_switching, number_of_points, dead_time_resolution)

        # check if ZVS condition is for switching event 1 fulfilled
        is_zvs_1 = is_zvs_1_a & is_zvs_1_b

        minimum_dead_time_first_switching_event = time_a_first_switching_event + time_b_first_switching_event

        # in case of tau_rad is not 180°, two maximum in i_lc appear (three different voltage levels on the bridge output)
        # but i_hf has two different current values at the switching points. The integration must be done on the second switching point also.
        # In the end, it must be checked which dead time is greater.
        is_zvs_2 = True
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
                t_switching_2 = i_lc_full_time_current_waveform_doubled[0][index_switching]
            else:
                t_switching_2 = i_lc_full_time_current_waveform_doubled[0][second_switching_index]

            t_interp_index_switching = HandleDabDto._index_of_nearest_value(time_high_resolution, t_switching_2)

            is_zvs_2_a, time_a_second_switching_event = HandleDabDto._integrate_part_a_leftwards(
                q_ab_half_req, time_high_resolution, i_hf_high_resolution, t_interp_index_switching, number_of_points, dead_time_resolution)
            is_zvs_2_b, time_b_second_switching_event = HandleDabDto._integrate_part_b_rightwards(
                q_ab_half_req, time_high_resolution, i_hf_high_resolution, t_interp_index_switching, number_of_points, dead_time_resolution)

            # check if ZVS condition is for switching event 2 fulfilled
            is_zvs_2 = is_zvs_2_a & is_zvs_2_b

            minimum_dead_time_second_switching_event = time_a_second_switching_event + time_b_second_switching_event
            logger.debug(f"{minimum_dead_time_second_switching_event=}")

            # overwrite dead time for the first switching event with greater secondary event switching time
            if minimum_dead_time_second_switching_event > minimum_dead_time_first_switching_event:
                logger.debug(
                    f"Second switching time {minimum_dead_time_second_switching_event} > first switching time {minimum_dead_time_first_switching_event}.")
                minimum_dead_time_first_switching_event = minimum_dead_time_second_switching_event

        is_zvs = is_zvs_1 & is_zvs_2

        if is_plot:
            # triangular shape
            fig, axs = plt.subplots(2, 1, sharex=True)
            axs[0].plot(i_lc_full_time_current_waveform_doubled[0], i_lc_full_time_current_waveform_doubled[1], label="i_lc", linestyle="--")
            # plot first switching event
            axs[0].plot([t_switching_1 - time_a_first_switching_event, t_switching_1 - time_a_first_switching_event],
                        [-1.1 * np.max(i_lc_full_time_current_waveform_doubled[1]), 1.1 * np.max(i_lc_full_time_current_waveform_doubled[1])],
                        linestyle="--", color="red")
            axs[0].plot([t_switching_1, t_switching_1], [-1.1 * np.max(i_lc_full_time_current_waveform_doubled[1]),
                                                         1.1 * np.max(i_lc_full_time_current_waveform_doubled[1])], color="red")
            axs[0].plot([t_switching_1 + time_b_first_switching_event, t_switching_1 + time_b_first_switching_event],
                        [-1.1 * np.max(i_lc_full_time_current_waveform_doubled[1]), 1.1 * np.max(i_lc_full_time_current_waveform_doubled[1])],
                        linestyle="--", color="red")

            if tau_rad != np.pi:
                # plot second switching event
                axs[0].plot([t_switching_2 - time_a_second_switching_event, t_switching_2 - time_a_second_switching_event],
                            [-1.1 * np.max(i_lc_full_time_current_waveform_doubled[1]), 1.1 * np.max(i_lc_full_time_current_waveform_doubled[1])],
                            linestyle="--", color="gray")
                axs[0].plot([t_switching_2, t_switching_2],
                            [-1.1 * np.max(i_lc_full_time_current_waveform_doubled[1]), 1.1 * np.max(i_lc_full_time_current_waveform_doubled[1])], color="gray")
                axs[0].plot([t_switching_2 + time_b_second_switching_event, t_switching_2 + time_b_second_switching_event],
                            [-1.1 * np.max(i_lc_full_time_current_waveform_doubled[1]), 1.1 * np.max(i_lc_full_time_current_waveform_doubled[1])],
                            linestyle="--", color="gray")

            axs[1].plot(i_hf_full_time_current_waveform_doubled[0], i_hf_full_time_current_waveform_doubled[1], label="i_hf")
            axs[1].plot(time_high_resolution, i_hf_high_resolution, label="i_hf interpolated")

            # plot first switching event
            axs[1].plot([t_switching_1 - time_a_first_switching_event, t_switching_1 - time_a_first_switching_event],
                        [-1.1 * np.max(i_hf_full_time_current_waveform_doubled[1]), 1.1 * np.max(i_hf_full_time_current_waveform_doubled[1])],
                        linestyle="--", color="red")
            axs[1].plot([t_switching_1, t_switching_1], [-1.1 * np.max(i_hf_full_time_current_waveform_doubled[1]),
                                                         1.1 * np.max(i_hf_full_time_current_waveform_doubled[1])], color="red")
            axs[1].plot([t_switching_1 + time_b_first_switching_event, t_switching_1 + time_b_first_switching_event],
                        [-1.1 * np.max(i_hf_full_time_current_waveform_doubled[1]), 1.1 * np.max(i_hf_full_time_current_waveform_doubled[1])],
                        linestyle="--", color="red")

            if tau_rad != np.pi:
                # plot second switching event
                axs[1].plot([t_switching_2 - time_a_second_switching_event, t_switching_2 - time_a_second_switching_event],
                            [-1.1 * np.max(i_hf_full_time_current_waveform_doubled[1]), 1.1 * np.max(i_hf_full_time_current_waveform_doubled[1])],
                            linestyle="--", color="gray")
                axs[1].plot([t_switching_2, t_switching_2],
                            [-1.1 * np.max(i_hf_full_time_current_waveform_doubled[1]), 1.1 * np.max(i_hf_full_time_current_waveform_doubled[1])],
                            color="gray")
                axs[1].plot([t_switching_2 + time_b_second_switching_event, t_switching_2 + time_b_second_switching_event],
                            [-1.1 * np.max(i_hf_full_time_current_waveform_doubled[1]), 1.1 * np.max(i_hf_full_time_current_waveform_doubled[1])],
                            linestyle="--", color="gray")

            axs[0].legend()
            axs[0].grid()
            axs[1].legend()
            axs[1].grid()
            plt.title(f"{tau_rad=}")
            plt.show()

        return is_zvs, minimum_dead_time_first_switching_event

    @staticmethod
    def get_c_oss_from_tdb(transistor: tdb.Transistor, margin_factor: float = 1.0) -> tuple:
        """
        Import the transistor Coss(Vds) capacitance from the transistor database (TDB).

        Note we assume V_ds in Volt and C_oss in F. If this is not the case, scale your data accordingly!

        :param transistor: transistor database object
        :type transistor: transistordatabase.Transistor
        :param margin_factor: factor for margin. [1.0 = no margin], [1.2 = 20 % margin: DEFAULT]
        :type margin_factor: float
        :return: c_oss, q_oss (in 1 V steps)
        :rtype: tuple
        """
        csv_data = transistor.c_oss[0].graph_v_c.T

        # Maybe check if data is monotonically
        # Check if voltage is monotonically rising
        if not np.all(csv_data[1:, 0] >= csv_data[:-1, 0], axis=0):
            logger.warning("The voltage in csv file is not monotonically rising!")
        # Check if Coss is monotonically falling
        if not np.all(csv_data[1:, 1] <= csv_data[:-1, 1], axis=0):
            logger.warning("The C_oss in csv file is not monotonically falling!")

        # Rescale and interpolate the csv data to have a nice 1V step size from 0V to v_max
        # A first value with zero volt will be added
        v_max = int(np.round(csv_data[-1, 0]))
        v_interp = np.arange(v_max + 1)

        # The margin is considered here as a factor of the original capacitance value
        coss_interp = margin_factor * np.interp(v_interp, csv_data[:, 0], csv_data[:, 1])
        # Since we now have an evenly spaced vector where x correspond to the element-number of the vector
        # we don't have to store x (v_interp) with it.
        # To get Coss(V) just get the array element coss_interp[V]

        # return coss_interp
        c_oss = coss_interp
        q_oss = HandleDabDto._integrate_c_oss(coss_interp)
        e_oss = cumulative_trapezoid(v_interp * c_oss, v_interp, initial=0)
        return v_interp, c_oss, q_oss, e_oss

    @staticmethod
    def _integrate_c_oss(coss):
        """
        Integrate Coss for each voltage from 0 to V_max.

        :param coss: MOSFET Coss(Vds) curve from Vds=0V to >= V1_max. Just one row with Coss data and index = Vds.
        :return: Qoss(Vds) as one row of data and index = Vds.
        """

        # Integrate from 0 to v
        def integrate(v):
            v_interp = np.arange(v + 1)
            coss_v = np.interp(v_interp, np.arange(coss.shape[0]), coss)
            return np.trapezoid(coss_v)

        coss_int = np.vectorize(integrate)
        # get the qoss vector that has the resolution 1V from 0 to V_max
        v_vec = np.arange(coss.shape[0])
        qoss = coss_int(v_vec)

        return qoss

    @staticmethod
    def save(dab_dto: d_dtos.DabCircuitDTO, name: str, directory: str, timestamp: bool = True) -> None:
        """
        Save the DabDTO-class to a npz file.

        :param dab_dto: Class to store
        :type dab_dto: d_dtos.DabCircuitDTO
        :param name: Filename
        :type name: str
        :param directory: Directory to store the results
        :type directory: str
        :param timestamp: [True] to add a timestamp to the file name.
        :type timestamp: bool
        """
        # Add some descriptive data to the file
        # Adding a timestamp, it may be useful
        dab_dto.timestamp = np.asarray(datetime.datetime.now().isoformat())

        # Adding a timestamp to the filename if requested
        if timestamp:
            if name:
                filename = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "_" + name
            else:
                filename = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        else:
            if name:
                filename = name
            else:
                # set some default non-empty filename
                filename = "dab_dataset"

        if directory:
            directory = os.path.expanduser(directory)
            directory = os.path.expandvars(directory)
            directory = os.path.abspath(directory)
            if os.path.isdir(directory):
                file = os.path.join(directory, filename)
            else:
                logger.warning("Directory does not exist!", stacklevel=2)
                file = os.path.join(filename)
        else:
            file = os.path.join(filename)

        with open(f"{file}.pkl", 'wb') as output:
            pickle.dump(dab_dto, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_from_file(file: str) -> d_dtos.DabCircuitDTO:
        """
        Load everything from the given .npz file.

        :param file: a .nps filename or file-like object, string, or pathlib.Path
        :return: two objects with type DAB_Specification and DAB_Results
        """
        # Check for filename extension
        file_name, file_extension = os.path.splitext(file)
        if not file_extension:
            file += '.pkl'
        file = os.path.expanduser(file)
        file = os.path.expandvars(file)
        file = os.path.abspath(file)

        with open(file, 'rb') as pickle_file_data:
            loaded_circuit_dto = pickle.load(pickle_file_data)
            if not isinstance(loaded_circuit_dto, d_dtos.DabCircuitDTO):
                raise TypeError(f"Loaded pickle file {loaded_circuit_dto} not of type DabCircuitDTO.")
            return loaded_circuit_dto

    @staticmethod
    def get_max_peak_waveform_transformer(dab_dto: d_dtos.DabCircuitDTO, plot: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int]]:
        """
        Get the transformer waveform with the maximum current peak out of the three-dimensional simulation array (v_1, v_2, P).

        :param dab_dto: DAB data transfer object (DTO)
        :type dab_dto: d_dtos.DabCircuitDTO
        :param plot: True to plot the results, mostly for understanding and debugging
        :type plot: bool
        :return: sorted_max_angles, i_l_s_max_current_waveform, i_hf_2_max_current_waveform, vvp_index. All as a numpy array except vvp_index (tuple)
        :rtype: List[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int]]
        """
        i_hf_2_sorted = np.transpose(dab_dto.calc_currents.i_l_s_sorted * dab_dto.input_config.n - dab_dto.calc_currents.i_l_2_sorted, (1, 2, 3, 0))
        angles_rad_sorted = np.transpose(dab_dto.calc_currents.angles_rad_sorted, (1, 2, 3, 0))

        max_index = (0, 0, 0)
        for vec_vvp in np.ndindex(dab_dto.calc_modulation.phi.shape):
            max_index = vec_vvp if np.max(i_hf_2_sorted[vec_vvp]) > np.max(i_hf_2_sorted[max_index]) else max_index  # type: ignore
            if plot:
                plt.plot(d_waveforms.full_angle_waveform_from_angles(angles_rad_sorted[vec_vvp]),
                         d_waveforms.full_current_waveform_from_currents(i_hf_2_sorted[vec_vvp]), color='gray')

        i_hf_2_max_current_waveform = i_hf_2_sorted[max_index]
        i_l_s_max_current_waveform = np.transpose(dab_dto.calc_currents.i_l_s_sorted, (1, 2, 3, 0))[max_index]

        sorted_max_angles, unique_indices = np.unique(d_waveforms.full_angle_waveform_from_angles(angles_rad_sorted[max_index]), return_index=True)
        i_l_s_max_current_waveform = d_waveforms.full_current_waveform_from_currents(i_l_s_max_current_waveform)[unique_indices]
        i_hf_2_max_current_waveform = d_waveforms.full_current_waveform_from_currents(i_hf_2_max_current_waveform)[unique_indices]

        if plot:
            plt.plot(sorted_max_angles, i_hf_2_max_current_waveform, color='red', label='peak current full waveform')
            plt.grid()
            plt.xlabel('Angle in rad')
            plt.ylabel('Current in A')
            plt.legend()
            plt.show()

        return sorted_max_angles, i_l_s_max_current_waveform, i_hf_2_max_current_waveform, max_index

    @staticmethod
    def get_max_peak_waveform_inductor(dab_dto: d_dtos.DabCircuitDTO, plot: bool = False) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
        """
        Get the inductor waveform with the maximum current peak out of the three-dimensional simulation array (v_1, v_2, P).

        :param dab_dto: DAB data transfer object (DTO)
        :type dab_dto: d_dtos.DabCircuitDTO
        :param plot: True to plot the results, mostly for understanding and debugging
        :type plot: bool
        :return: [sorted_max_angles, i_l_max_current_waveform, vvp_max_index]. As a numpy array, index as tuple
        :rtype: List[np.ndarray, np.ndarray, tuple[int, int, int]]
        """
        i_l1_sorted = np.transpose(dab_dto.calc_currents.i_l_1_sorted, (1, 2, 3, 0))
        angles_rad_sorted = np.transpose(dab_dto.calc_currents.angles_rad_sorted, (1, 2, 3, 0))

        max_index = (0, 0, 0)
        for vec_vvp in np.ndindex(dab_dto.calc_modulation.phi.shape):
            max_index = vec_vvp if np.max(i_l1_sorted[vec_vvp]) > np.max(i_l1_sorted[max_index]) else max_index  # type: ignore
            if plot:
                plt.plot(d_waveforms.full_angle_waveform_from_angles(angles_rad_sorted[vec_vvp]),
                         d_waveforms.full_current_waveform_from_currents(i_l1_sorted[vec_vvp]), color='gray')

        i_l1_max_current_waveform = i_l1_sorted[max_index]

        sorted_max_angles, unique_indices = np.unique(d_waveforms.full_angle_waveform_from_angles(angles_rad_sorted[max_index]), return_index=True)
        i_l1_max_current_waveform = d_waveforms.full_current_waveform_from_currents(i_l1_max_current_waveform)[unique_indices]

        if plot:
            plt.plot(sorted_max_angles, i_l1_max_current_waveform, color='red', label='peak current full waveform')
            plt.grid()
            plt.xlabel('Angle in rad')
            plt.ylabel('Current in A')
            plt.legend()
            plt.show()

        return sorted_max_angles, i_l1_max_current_waveform, max_index

    @staticmethod
    def get_max_rms_waveform_capacitor_1(dab_dto: d_dtos.DabCircuitDTO, plot: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the capacitor waveform with the maximum RMS current out of the three-dimensional simulation array (v_1, v_2, P).

        :param dab_dto: DAB data transfer object (DTO)
        :type dab_dto: d_dtos.DabCircuitDTO
        :param plot: True to plot the results, mostly for understanding and debugging
        :type plot: bool
        :return: angle_vec, i_hf1_max_rms_current_waveform, All as a numpy array.
        :rtype: List[np.ndarray]
        """
        i_c1_sorted = np.transpose(dab_dto.calc_currents.i_hf_1_sorted, (1, 2, 3, 0))
        i_c1_rms = dab_dto.calc_currents.i_hf_1_rms
        angles_rad_sorted = np.transpose(dab_dto.calc_currents.angles_rad_sorted, (1, 2, 3, 0))

        max_index = (0, 0, 0)
        for vec_vvp in np.ndindex(dab_dto.calc_modulation.phi.shape):
            max_index = vec_vvp if np.max(i_c1_rms[vec_vvp]) > np.max(i_c1_rms[max_index]) else max_index  # type: ignore
            if plot:
                plt.plot(d_waveforms.full_angle_waveform_from_angles(angles_rad_sorted[vec_vvp]),
                         d_waveforms.full_current_waveform_from_currents(i_c1_sorted[vec_vvp]), color='gray')

        i_c1_max_rms_current_waveform = i_c1_sorted[max_index]

        sorted_max_rms_angles, unique_indices = np.unique(d_waveforms.full_angle_waveform_from_angles(angles_rad_sorted[max_index]), return_index=True)
        i_c1_max_rms_current_waveform = d_waveforms.full_current_waveform_from_currents(i_c1_max_rms_current_waveform)[unique_indices]

        if plot:
            plt.plot(sorted_max_rms_angles, i_c1_max_rms_current_waveform, color='red', label='peak current full waveform')
            plt.grid()
            plt.xlabel('Angle in rad')
            plt.ylabel('Current in A')
            plt.legend()
            plt.show()

        return sorted_max_rms_angles, i_c1_max_rms_current_waveform

    @staticmethod
    def get_max_rms_waveform_capacitor_2(dab_dto: d_dtos.DabCircuitDTO, plot: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the capacitor waveform with the maximum RMS current out of the three-dimensional simulation array (v_1, v_2, P).

        :param dab_dto: DAB data transfer object (DTO)
        :type dab_dto: d_dtos.DabCircuitDTO
        :param plot: True to plot the results, mostly for understanding and debugging
        :type plot: bool
        :return: angle_vec, i_hf1_max_rms_current_waveform, All as a numpy array.
        :rtype: List[np.ndarray]
        """
        i_c2_sorted = np.transpose(dab_dto.calc_currents.i_hf_2_sorted, (1, 2, 3, 0))
        i_c2_rms = dab_dto.calc_currents.i_hf_2_rms
        angles_rad_sorted = np.transpose(dab_dto.calc_currents.angles_rad_sorted, (1, 2, 3, 0))

        max_index = (0, 0, 0)
        for vec_vvp in np.ndindex(dab_dto.calc_modulation.phi.shape):
            max_index = vec_vvp if np.max(i_c2_rms[vec_vvp]) > np.max(i_c2_rms[max_index]) else max_index  # type: ignore
            if plot:
                plt.plot(d_waveforms.full_angle_waveform_from_angles(angles_rad_sorted[vec_vvp]),
                         d_waveforms.full_current_waveform_from_currents(i_c2_sorted[vec_vvp]), color='gray')

        i_c2_max_rms_current_waveform = i_c2_sorted[max_index]

        sorted_max_rms_angles, unique_indices = np.unique(d_waveforms.full_angle_waveform_from_angles(angles_rad_sorted[max_index]), return_index=True)
        i_c2_max_rms_current_waveform = d_waveforms.full_current_waveform_from_currents(i_c2_max_rms_current_waveform)[unique_indices]

        if plot:
            plt.plot(sorted_max_rms_angles, i_c2_max_rms_current_waveform, color='red', label='peak current full waveform')
            plt.grid()
            plt.xlabel('Angle in rad')
            plt.ylabel('Current in A')
            plt.legend()
            plt.show()

        return sorted_max_rms_angles, i_c2_max_rms_current_waveform

    @staticmethod
    def add_inductor_results(dab_dto: d_dtos.DabCircuitDTO, inductor_results: InductorResults) -> d_dtos.DabCircuitDTO:
        """Add inductor results to the DabCircuitDTO.

        :param dab_dto: Dual-active bridge DTO
        :type dab_dto: d_dtos.DabCircuitDTO
        :param inductor_results: inductor losses
        :type inductor_results: InductorResults
        :return: Dual-active bridge DTO including the inductor losses
        :rtype: d_dtos.DabCircuitDTO
        """
        dab_dto.inductor_results = inductor_results

        return dab_dto

    @staticmethod
    def add_stacked_transformer_results(dab_dto: d_dtos.DabCircuitDTO, transformer_results: StackedTransformerResults) -> d_dtos.DabCircuitDTO:
        """Add stacked transformer results to the DabCircuitDTO.

        :param dab_dto: Dual-active bridge DTO
        :type dab_dto: d_dtos.DabCircuitDTO
        :param transformer_results: stacked transformer results
        :type transformer_results: StackedTransformerResults
        :return: Dual-active bridge DTO including the inductor losses
        :rtype: d_dtos.DabCircuitDTO
        """
        dab_dto.stacked_transformer_results = transformer_results
        return dab_dto

    @staticmethod
    def tdb_to_transistor_dto(transistor_name: str, c_oss_margin_factor: float = 1.2) -> d_dtos.TransistorDTO:
        """
        Load a transistor from the transistor database and transfer the important parameters to a TransistorDTO.

        :param transistor_name: transistor name, must be same name as in the transistor database
        :type transistor_name: str
        :param c_oss_margin_factor: margin for c_oss
        :type c_oss_margin_factor: float
        :return: Transistor DTO
        :rtype: d_dtosTransistorDTO
        """
        db = tdb.DatabaseManager()
        db.set_operation_mode_json()

        # Check if a valid directory is not initialized
        if HandleDabDto.c_oss_storage_directory == "":
            raise ValueError("A valid transistor database directory is not set./n"+"Use method 'set_c_oss_storage_directory' for this purpose!")

        transistor: tdb.Transistor = db.load_transistor(transistor_name)

        if transistor.type != "MOSFET" and transistor.type != "SiC-MOSFET":
            raise ValueError(f"Transistor 1: {transistor.name} is of non-allowed type {transistor.type}. "
                             f"Allowed types are MOSFET, SiC-MOSFET.")

        t_j_recommended = transistor.switch.t_j_max - 25

        v_interp, c_oss, q_oss, e_oss = (HandleDabDto.get_c_oss_from_tdb(transistor, margin_factor=c_oss_margin_factor))

        # export c_oss files for GeckoCIRCUITS
        if not os.path.exists(os.path.join(HandleDabDto.c_oss_storage_directory, f"{transistor.name}_c_oss.nlc")):
            transistor.export_geckocircuits_coss(filepath=HandleDabDto.c_oss_storage_directory, margin_factor=c_oss_margin_factor)

        transistor.quickstart_wp()

        transistor_dto = d_dtos.TransistorDTO(
            name=transistor.name,
            t_j_max_op=t_j_recommended,
            v_oss=v_interp,
            c_oss=c_oss,
            q_oss=q_oss,
            e_oss=e_oss,
            r_th_jc=transistor.switch.thermal_foster.r_th_total,
            cooling_area=transistor.cooling_area,
            housing_area=transistor.housing_area,
            r_channel=transistor.wp.switch_r_channel,
            # switching loss data sheet curves
            turn_on_current_vec=transistor.wp.e_on.graph_i_e[0],
            turn_on_energy_vec=transistor.wp.e_on.graph_i_e[1],
            turn_on_temperature=transistor.wp.e_on.t_j,
            turn_on_voltage=transistor.wp.e_on.v_supply,
            turn_off_current_vec=transistor.wp.e_off.graph_i_e[0],
            turn_off_energy_vec=transistor.wp.e_off.graph_i_e[1],
            turn_off_temperature=transistor.wp.e_off.t_j,
            turn_off_voltage=transistor.wp.e_off.v_supply,
            # switching loss measurement fitted curves
            turn_on_fit_factors=transistor.wp.e_on_meas_fit,
            turn_off_fit_factors=transistor.wp.e_off_meas_fit
        )

        return transistor_dto

    @staticmethod
    def generate_capacitor_1_target_requirements(dab_dto: d_dtos.DabCircuitDTO) -> CapacitorRequirements:
        """Capacitor 1 requirements.

        :param dab_dto: DAB circuit DTO
        :type dab_dto: d_dtos.DabCircuitDTO
        :return: capacitor requirements
        :rtype: CapacitorRequirements
        """
        # get the single maximum operating point
        sorted_max_rms_angles, i_c1_max_rms_current_waveform = HandleDabDto.get_max_rms_waveform_capacitor_1(dab_dto, plot=False)

        # get all current waveforms for all operating points
        i_l1_sorted = np.transpose(dab_dto.calc_currents.i_l_1_sorted, (1, 2, 3, 0))
        angles_rad_sorted = np.transpose(dab_dto.calc_currents.angles_rad_sorted, (1, 2, 3, 0))

        dimension_0 = np.shape(i_l1_sorted)[0]
        dimension_1 = np.shape(i_l1_sorted)[1]
        dimension_2 = np.shape(i_l1_sorted)[2]
        dimension_3 = np.shape(i_l1_sorted)[3]

        time_array = np.full((dimension_0, dimension_1, dimension_2, dimension_3 + 5), np.nan)
        current_array = np.full((dimension_0, dimension_1, dimension_2, dimension_3 + 5), np.nan)

        for vec_vvp in np.ndindex(dab_dto.calc_modulation.phi.shape):
            time = full_angle_waveform_from_angles(angles_rad_sorted[vec_vvp]) / 2 / np.pi / dab_dto.input_config.fs

            # needs np.unique( , return_index=True)
            current = full_current_waveform_from_currents(i_l1_sorted[vec_vvp])  # [unique_indices]

            time_array[vec_vvp] = time
            current_array[vec_vvp] = current

        time_array_resorted = np.transpose(time_array, (0, 1, 2, 3))
        current_array_resorted = np.transpose(current_array, (0, 1, 2, 3))

        capacitor_1_requirements = CapacitorRequirements(
            current_vec=i_c1_max_rms_current_waveform,
            time_vec=sorted_max_rms_angles / (2 * np.pi * dab_dto.input_config.fs),
            time_array=time_array_resorted,
            current_array=current_array_resorted,
            v_dc_max=np.max(dab_dto.input_config.mesh_v1),
            study_name="",
            circuit_id=dab_dto.circuit_id,
            capacitor_number_in_circuit=0
        )
        return capacitor_1_requirements

    @staticmethod
    def generate_capacitor_2_target_requirements(dab_dto: d_dtos.DabCircuitDTO) -> CapacitorRequirements:
        """Capacitor 2 requirements.

        :param dab_dto: DAB circuit DTO
        :type dab_dto: d_dtos.DabCircuitDTO
        :return: capacitor requirements
        :rtype: CapacitorRequirements
        """
        # get the single maximum operating point
        sorted_max_rms_angles, i_c2_max_rms_current_waveform = HandleDabDto.get_max_rms_waveform_capacitor_2(dab_dto, plot=False)

        # get all current waveforms for all operating points
        i_hf2_sorted = np.transpose(dab_dto.calc_currents.i_hf_2_sorted, (1, 2, 3, 0))
        angles_rad_sorted = np.transpose(dab_dto.calc_currents.angles_rad_sorted, (1, 2, 3, 0))

        dimension_0 = np.shape(i_hf2_sorted)[0]
        dimension_1 = np.shape(i_hf2_sorted)[1]
        dimension_2 = np.shape(i_hf2_sorted)[2]
        dimension_3 = np.shape(i_hf2_sorted)[3]

        time_array = np.full((dimension_0, dimension_1, dimension_2, dimension_3 + 5), np.nan)
        current_array = np.full((dimension_0, dimension_1, dimension_2, dimension_3 + 5), np.nan)

        for vec_vvp in np.ndindex(dab_dto.calc_modulation.phi.shape):
            time = full_angle_waveform_from_angles(angles_rad_sorted[vec_vvp]) / 2 / np.pi / dab_dto.input_config.fs

            # needs np.unique( , return_index=True)
            current = full_current_waveform_from_currents(i_hf2_sorted[vec_vvp])  # [unique_indices]

            time_array[vec_vvp] = time
            current_array[vec_vvp] = current

        time_array_resorted = np.transpose(time_array, (0, 1, 2, 3))
        current_array_resorted = np.transpose(current_array, (0, 1, 2, 3))

        capacitor_2_requirements = CapacitorRequirements(
            current_vec=i_c2_max_rms_current_waveform,
            time_vec=sorted_max_rms_angles / (2 * np.pi * dab_dto.input_config.fs),
            time_array=time_array_resorted,
            current_array=current_array_resorted,
            v_dc_max=np.max(dab_dto.input_config.mesh_v1),
            study_name="",
            circuit_id=dab_dto.circuit_id,
            capacitor_number_in_circuit=1
        )
        return capacitor_2_requirements

    @staticmethod
    def generate_inductor_target_requirements(dab_dto: d_dtos.DabCircuitDTO) -> InductorRequirements:
        """Inductor requirements.

        :param dab_dto: DAB circuit DTO
        :type dab_dto: d_dtos.DabCircuitDTO
        :return: Inductor requirements
        :rtype: InductorRequirements
        """
        # get the single maximum operating point
        sorted_max_rms_angles, i_l_1_max_peak_current_waveform, vvp_max_index = HandleDabDto.get_max_peak_waveform_inductor(dab_dto, plot=False)

        # get all current waveforms for all operating points
        i_l_1_sorted = np.transpose(dab_dto.calc_currents.i_l_1_sorted, (1, 2, 3, 0))
        angles_rad_sorted = np.transpose(dab_dto.calc_currents.angles_rad_sorted, (1, 2, 3, 0))

        dimension_0 = np.shape(i_l_1_sorted)[0]
        dimension_1 = np.shape(i_l_1_sorted)[1]
        dimension_2 = np.shape(i_l_1_sorted)[2]
        dimension_3 = np.shape(i_l_1_sorted)[3]

        time_array = np.full((dimension_0, dimension_1, dimension_2, dimension_3 + 5), np.nan)
        current_array = np.full((dimension_0, dimension_1, dimension_2, dimension_3 + 5), np.nan)

        for vec_vvp in np.ndindex(dab_dto.calc_modulation.phi.shape):
            time = full_angle_waveform_from_angles(angles_rad_sorted[vec_vvp]) / 2 / np.pi / dab_dto.input_config.fs

            # needs np.unique( , return_index=True)
            current = full_current_waveform_from_currents(i_l_1_sorted[vec_vvp])  # [unique_indices]

            time_array[vec_vvp] = time
            current_array[vec_vvp] = current

        time_array_resorted = np.transpose(time_array, (0, 1, 2, 3))
        current_array_resorted = np.transpose(current_array, (0, 1, 2, 3))

        inductor_requirements = InductorRequirements(
            current_vec=i_l_1_max_peak_current_waveform,
            time_vec=sorted_max_rms_angles / (2 * np.pi * dab_dto.input_config.fs),
            vvp_index=vvp_max_index,
            time_array=time_array_resorted,
            current_array=current_array_resorted,
            study_name="",
            target_inductance=dab_dto.input_config.Lc1,
            circuit_id=dab_dto.circuit_id,
            inductor_number_in_circuit=0,
        )
        return inductor_requirements

    @staticmethod
    def generate_transformer_target_requirements(dab_dto: d_dtos.DabCircuitDTO) -> TransformerRequirements:
        """Generate transformer requirements.

        Note: the current counting system is adapted to FEMMT! The secondary current is counted negative!

        :param dab_dto: DAB circuit DTO
        :type dab_dto: d_dtos.DabCircuitDTO
        :return: Transformer requirements
        :rtype: TransformerRequirements
        """
        # get the single maximum operating point
        sorted_max_rms_angles, i_l_s_max_current_waveform, i_hf_2_max_current_waveform, vvp_max_index = HandleDabDto.get_max_peak_waveform_transformer(
            dab_dto, plot=False)

        # get all current waveforms for all operating points
        i_l_s_sorted = np.transpose(dab_dto.calc_currents.i_l_s_sorted, (1, 2, 3, 0))
        i_hf_2_sorted = np.transpose(dab_dto.calc_currents.i_hf_2_sorted, (1, 2, 3, 0))
        angles_rad_sorted = np.transpose(dab_dto.calc_currents.angles_rad_sorted, (1, 2, 3, 0))

        dimension_0 = np.shape(i_l_s_sorted)[0]
        dimension_1 = np.shape(i_l_s_sorted)[1]
        dimension_2 = np.shape(i_l_s_sorted)[2]
        dimension_3 = np.shape(i_l_s_sorted)[3]

        time_array = np.full((dimension_0, dimension_1, dimension_2, dimension_3 + 5), np.nan)
        current_1_array = np.full((dimension_0, dimension_1, dimension_2, dimension_3 + 5), np.nan)
        current_2_array = np.full((dimension_0, dimension_1, dimension_2, dimension_3 + 5), np.nan)

        for vec_vvp in np.ndindex(dab_dto.calc_modulation.phi.shape):
            time = full_angle_waveform_from_angles(angles_rad_sorted[vec_vvp]) / 2 / np.pi / dab_dto.input_config.fs

            # needs np.unique( , return_index=True)
            current_1 = full_current_waveform_from_currents(i_l_s_sorted[vec_vvp])  # [unique_indices]
            current_2 = full_current_waveform_from_currents(i_hf_2_sorted[vec_vvp])  # [unique_indices]

            time_array[vec_vvp] = time
            current_1_array[vec_vvp] = current_1
            current_2_array[vec_vvp] = -current_2

        time_array_resorted = np.transpose(time_array, (0, 1, 2, 3))
        current_1_array_resorted = np.transpose(current_1_array, (0, 1, 2, 3))
        current_2_array_resorted = np.transpose(current_2_array, (0, 1, 2, 3))

        transformer_requirements = TransformerRequirements(

            l_s12_target=dab_dto.input_config.Ls,
            l_h_target=dab_dto.calc_config.Lc2_,
            n_target=dab_dto.input_config.n,
            temperature=100,

            # exact a single current waveform to optimize the transformer
            time_vec=sorted_max_rms_angles / (2 * np.pi * dab_dto.input_config.fs),
            current_1_vec=i_l_s_max_current_waveform,
            current_2_vec=-i_hf_2_max_current_waveform,
            vvp_index=vvp_max_index,

            # all current waveforms for calculation the transformer loss for a single (optimized) transformer
            time_array=time_array_resorted,
            current_1_array=current_1_array_resorted,
            current_2_array=current_2_array_resorted,

            study_name="",
            circuit_id=dab_dto.circuit_id,
            transformer_number_in_circuit=0,
        )
        return transformer_requirements

    @staticmethod
    def generate_components_target_requirements(dab_dto: d_dtos.DabCircuitDTO) -> d_dtos.DabCircuitDTO:
        """
        DAB component requirements (capacitors, inductor, transformer).

        :param dab_dto: DAB circuit DTO
        :type dab_dto: d_dtos.DabCircuitDTO
        :return: updated DAB circuit DTO
        :rtype: d_dtos.DabCircuitDTO
        """
        capacitor_1_requirements = HandleDabDto.generate_capacitor_1_target_requirements(dab_dto)
        capacitor_2_requirements = HandleDabDto.generate_capacitor_2_target_requirements(dab_dto)
        inductor_requirements = HandleDabDto.generate_inductor_target_requirements(dab_dto)
        transformer_requirements = HandleDabDto.generate_transformer_target_requirements(dab_dto)

        dab_dto.component_requirements = d_dtos.ComponentRequirements(capacitor_requirements=[capacitor_1_requirements, capacitor_2_requirements],
                                                                      inductor_requirements=[inductor_requirements],
                                                                      transformer_requirements=[transformer_requirements])
        return dab_dto

    @staticmethod
    def generate_thermal_transistor_parameters(circuit_dto: d_dtos.DabCircuitDTO,
                                               transistor_b1_cooling: ComponentCooling,
                                               transistor_b2_cooling: ComponentCooling) -> d_dtos.DabCircuitDTO:
        """
        Generate the transistor thermal parameters.

        :param circuit_dto: DAB circuit DTO
        :type circuit_dto: d_dtos.DabCircuitDTO
        :param transistor_b1_cooling: Transistor cooling
        :type transistor_b1_cooling: ComponentCooling
        :param transistor_b2_cooling: Transistor cooling
        :type transistor_b2_cooling: ComponentCooling
        :return: DAB circuit DTO
        :rtype: d_dtos.DabCircuitDTO
        """
        if circuit_dto.calc_losses is None:
            raise ValueError("Missing transistor loss calculation of bridge 1.")
        if circuit_dto.calc_losses is None:
            raise ValueError("Missing transistor loss calculation of bridge 2.")
        if np.any(np.isnan(circuit_dto.calc_losses.p_m1_switching)):
            raise ValueError("Missing transistor switching loss calculation of bridge 1.")
        if np.any(np.isnan(circuit_dto.calc_losses.p_m2_switching)):
            raise ValueError("Missing transistor switching loss calculation of bridge 2.")

        b1_transistor_loss_matrix = circuit_dto.calc_losses.p_m1_conduction + circuit_dto.calc_losses.p_m1_switching
        b2_transistor_loss_matrix = circuit_dto.calc_losses.p_m2_conduction + circuit_dto.calc_losses.p_m2_switching

        # get all the losses in a matrix
        r_th_copper_coin_1, copper_coin_area_1 = ThermalCalcSupport.calculate_r_th_copper_coin(
            circuit_dto.input_config.transistor_dto_1.cooling_area)
        r_th_copper_coin_2, copper_coin_area_2 = ThermalCalcSupport.calculate_r_th_copper_coin(
            circuit_dto.input_config.transistor_dto_2.cooling_area)

        circuit_r_th_tim_1 = ThermalCalcSupport.calculate_r_th_tim(
            copper_coin_area_1, transistor_b1_cooling)
        circuit_r_th_tim_2 = ThermalCalcSupport.calculate_r_th_tim(
            copper_coin_area_2, transistor_b2_cooling)

        circuit_r_th_1_jhs = circuit_dto.input_config.transistor_dto_1.r_th_jc + r_th_copper_coin_1 + circuit_r_th_tim_1
        circuit_r_th_2_jhs = circuit_dto.input_config.transistor_dto_2.r_th_jc + r_th_copper_coin_2 + circuit_r_th_tim_2

        circuit_heat_sink_max_1_array = (circuit_dto.input_config.transistor_dto_1.t_j_max_op - circuit_r_th_1_jhs * b1_transistor_loss_matrix)
        circuit_heat_sink_max_2_array = (circuit_dto.input_config.transistor_dto_2.t_j_max_op - circuit_r_th_2_jhs * b2_transistor_loss_matrix)

        circuit_thermal_0 = CircuitThermal(
            t_j_max=circuit_dto.input_config.transistor_dto_1.t_j_max_op,
            r_th_jhs=circuit_r_th_1_jhs,
            area=4 * copper_coin_area_1,
            loss_array=b1_transistor_loss_matrix,
            temperature_heat_sink_max_array=circuit_heat_sink_max_1_array
        )

        circuit_thermal_1 = CircuitThermal(
            t_j_max=circuit_dto.input_config.transistor_dto_2.t_j_max_op,
            r_th_jhs=circuit_r_th_2_jhs,
            area=4 * copper_coin_area_2,
            loss_array=b2_transistor_loss_matrix,
            temperature_heat_sink_max_array=circuit_heat_sink_max_2_array
        )

        circuit_dto.circuit_thermal = [circuit_thermal_0, circuit_thermal_1]

        return circuit_dto
