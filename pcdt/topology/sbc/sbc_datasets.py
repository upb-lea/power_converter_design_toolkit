"""Describe the dataclasses how to store the input parameters and results."""

# python libraries
import os
import datetime
import logging
import pickle

# 3rd party libraries
import numpy as np
from numpy.typing import NDArray
import transistordatabase as tdb
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator as RGI


# own libraries
import pcdt.topology.sbc.sbc_datasets_dtos as s_dtos
import pcdt.components.component_dtos as c_dtos
from pcdt.components.heat_sink_optimization import ThermalCalcSupport
import pcdt.topology.sbc.sbc_currents as sbc_currents
from pcdt.topology.sbc.sbc_circuit_topology_dtos import CircuitSampling
from pcdt.constants import TRANSISTOR_TEMPERATURE_SAFETY_MARGIN

logger = logging.getLogger(__name__)

class HandleSbcDto:
    """Class to handle the Sbc data transfer object, e.g. save and load the files."""

    @staticmethod
    def init_config(name: str, mesh_v1: np.ndarray, mesh_duty_cycle: np.ndarray, mesh_i2: np.ndarray,
                    sampling: CircuitSampling, ls: float, fs: float,
                    transistor_dto_1: s_dtos.TransistorDTO, transistor_dto_2: s_dtos.TransistorDTO) -> s_dtos.SbcCircuitDTO:
        """
        Initialize the SBC structure.

        :param name: name of the simulation
        :type  name: str
        :param mesh_v1: mesh or hypercube sampling for v1
        :type  mesh_v1: np.ndarray
        :param mesh_duty_cycle: mesh or hypercube sampling for duty cycle
        :type  mesh_duty_cycle: np.ndarray
        :param mesh_i2: mesh or hypercube sampling for current
        :type  mesh_i2: np.ndarray
        :param sampling: Sampling parameters
        :type  sampling: d_dtos.Sampling
        :param ls: series inductance
        :type  ls: float
        :param fs: Switching frequency
        :type  fs: float
        :param transistor_dto_1: Transistor DTO for transistor bridge 1. Must match with transistordatabase available transistors.
        :type  transistor_dto_1: TransistorDTO
        :param transistor_dto_2: Transistor DTO for transistor bridge 1. Must match with transistordatabase available transistors.
        :type  transistor_dto_2: TransistorDTO
        :return:
        """
        input_configuration = s_dtos.CircuitConfig(mesh_v1=mesh_v1,
                                                   mesh_duty_cycle=mesh_duty_cycle,
                                                   mesh_i2=mesh_i2,
                                                   sampling=sampling,
                                                   Ls=np.array(ls),
                                                   fs=np.array(fs),
                                                   transistor_dto_1=transistor_dto_1,
                                                   transistor_dto_2=transistor_dto_2)

        # Design space:
        # fs, L M (HS and LS same type, later to replace by variable)
        #
        # Chose: fs, L M -> p_loss_transistor, ripple current, volume proxy L
        # Next iteration: Change...
        #
        # P_Loss=P_cond+P_switch=I_rms²*R_D_S_on=(I_out²+I_Ripple¹/12)*R_D_S_on
        # I_Ripple=(V_in-V_out)*D/(L*fs)  = V_in*D(1-D)/(L*fs)
        # P_switch= 0.5⋅V_in⋅I_pk⋅(t_raise+t_fall)⋅f (First assumption for the simulation, later calculation over switch energy)

        # Calculate the ripple current and rms current
        i_ripple, i_ms, i_rms = sbc_currents.calc_rms_currents(input_configuration)

        calc_currents = s_dtos.CalcCurrents(i_rms=i_rms, i_ms=i_ms, i_ripple=i_ripple)

        # Calculate the transistor conduction losses p_hs_cond = (I_out²+I_Ripple²/12)*R_D_S_on
        p_hs_cond = HandleTransistorDto.transistor_conduction_loss(i_ms * input_configuration.mesh_duty_cycle, transistor_dto_1)
        p_ls_cond = HandleTransistorDto.transistor_conduction_loss(i_ms * (1 - input_configuration.mesh_duty_cycle),
                                                                   transistor_dto_1)
        p_hs_switch, t_hs_switch_on, t_hs_switch_off = HandleTransistorDto.transistor_switch_loss(
            input_configuration.mesh_v1, i_rms, input_configuration.transistor_dto_1, input_configuration.fs)
        p_ls_switch, t_ls_switch_on, t_ls_switch_off = HandleTransistorDto.transistor_switch_loss(
            input_configuration.mesh_v1, i_rms, input_configuration.transistor_dto_2, input_configuration.fs)

        p_loss = s_dtos.CalcLosses(**{'p_hs_conduction': p_hs_cond.ravel(),
                                      'p_ls_conduction': p_ls_cond.ravel(),
                                      'p_hs_switch': p_hs_switch,
                                      't_hs_switch_on': t_hs_switch_on,
                                      't_hs_switch_off': t_ls_switch_off,
                                      'p_ls_switch': p_ls_switch,
                                      't_ls_switch_on': t_hs_switch_on,
                                      't_ls_switch_off': t_ls_switch_off,
                                      'p_sbc_total': p_hs_cond.ravel() + p_ls_cond.ravel() + p_hs_switch.ravel() + p_ls_switch.ravel()})

        sbc_dto = s_dtos.SbcCircuitDTO(
            timestamp=None,
            circuit_id=name,
            metadata=None,
            input_config=input_configuration,
            calc_currents=calc_currents,
            calc_losses=p_loss,
            component_requirements=None,
            capacitor_1_results=None,
            capacitor_2_results=None,
            inductor_results=None,
            circuit_thermal=None,
        )

        return sbc_dto

    @staticmethod
    def get_c_oss_from_tdb(transistor: tdb.Transistor, margin_factor: float = 1.2) -> tuple:
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
        q_oss = HandleSbcDto._integrate_c_oss(coss_interp)
        return c_oss, q_oss

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
    def save(sbc_dto: s_dtos.SbcCircuitDTO, name: str, directory: str, timestamp: bool = True) -> None:
        """
        Save the SbcCircuitDTO-class to a npz file.

        :param sbc_dto: Class to store
        :type sbc_dto: d_dtos.SbcCircuitDTO
        :param name: Filename
        :type name: str
        :param directory: Directory to store the results
        :type directory: str
        :param timestamp: [True] to add a timestamp to the file name.
        :type timestamp: bool
        """
        # Add some descriptive data to the file
        # Adding a timestamp, it may be useful
        sbc_dto.timestamp = np.asarray(datetime.datetime.now().isoformat())

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
                filename = "sbc_dataset"

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
            pickle.dump(sbc_dto, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_from_file(file: str) -> s_dtos.SbcCircuitDTO:
        """
        Load everything from the given .npz file.

        :param file: a .nps filename or file-like object, string, or pathlib.Path
        :return: two objects with type SBC_Specification and SBC_Results
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
            if not isinstance(loaded_circuit_dto, s_dtos.SbcCircuitDTO):
                raise TypeError(f"Loaded pickle file {loaded_circuit_dto} not of type CircuitSbcDTO.")
            return loaded_circuit_dto

    @staticmethod
    def get_max_peak_waveform_inductor(sbc_dto: s_dtos.SbcCircuitDTO, plot: bool = False) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
        """
        Get the inductor waveform with the maximum current peak out of the three-dimensional simulation array (v_1, v_2, P).

        :param sbc_dto: SBC data transfer object (DTO)
        :type sbc_dto: d_dtos.SbcCircuitDTO
        :param plot: True to plot the results, mostly for understanding and debugging
        :type plot: bool
        :return: sorted_max_angles, i_l_s_max_current_waveform, i_l1_max_current_waveform. All as a numpy array.
        :rtype: List[np.ndarray]
        """
        # Variable declaration
        i_current_vec: np.ndarray
        i_max_current_array: np.ndarray
        time_array: np.ndarray

        # Calculate the current vector for each operation point
        i_current_vec = sbc_dto.input_config.mesh_i2 + sbc_dto.calc_currents.i_ripple / 2
        # Get the maximum current and correspondent duty cycle and ripple current
        idx_max = i_current_vec.argmax()
        i_max_current = i_current_vec[idx_max]
        i_ripple_at_max = sbc_dto.calc_currents.i_ripple[idx_max] / 2
        duty_cycle_at_max = sbc_dto.input_config.mesh_duty_cycle[idx_max]
        # Create time vector for correspondent operation point
        time_array = np.array([0, duty_cycle_at_max.squeeze(), 1]) / sbc_dto.input_config.fs
        i_max_current_array = np.array([i_max_current - i_ripple_at_max, i_max_current, i_max_current - i_ripple_at_max])

        if plot:
            plt.plot(time_array, i_max_current_array, color='red', label='peak current full waveform')
            plt.grid()
            plt.xlabel('time in us')
            plt.ylabel('Current in A')
            plt.legend()
            plt.show()

        # Workaround: Satisfy vvp_index - shape
        max_index = (0, int(idx_max), 0)

        return time_array, i_max_current_array.squeeze(), max_index

    @staticmethod
    def get_waveform_inductor(sbc_dto: s_dtos.SbcCircuitDTO, plot: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the inductor waveform for all operation points.

        :param sbc_dto: SBC data transfer object (DTO)
        :type sbc_dto: d_dtos.SbcCircuitDTO
        :param plot: True to plot the results, mostly for understanding and debugging
        :type plot: bool
        :return: sorted_max_angles, i_l_s_max_current_waveform, i_l1_max_current_waveform. All as a numpy array.
        :rtype: List[np.ndarray]
        """
        # Variable declaration
        i_current_array: np.ndarray
        time_array: np.ndarray

        # Create time array with 3 columns and number of operation points rows
        n = sbc_dto.input_config.mesh_duty_cycle.shape[0]  # Number of lines
        time_array = np.zeros((n, 3))
        time_array[:, 0] = 0
        time_array[:, 1] = sbc_dto.input_config.mesh_duty_cycle.squeeze()
        time_array[:, 2] = 1
        # Calculate times
        time_array = time_array * (1 / sbc_dto.input_config.fs)

        # Get the current and correspondent duty cycle and ripple current
        # i_ripple_current = sbc_dto.calc_currents.i_ripple.reshape(8, 1)
        # i_current_array = np.array([-0.5, +0.5, -0.5]).reshape(1, 3)
        i_ripple_current = sbc_dto.calc_currents.i_ripple
        i_current_array = np.array([-0.5, +0.5, -0.5])
        i_current_array = np.outer(i_ripple_current, i_current_array)
        i_current_array = i_current_array + sbc_dto.input_config.mesh_i2

        return time_array, i_current_array

    @staticmethod
    def generate_components_target_requirements(sbc_dto: s_dtos.SbcCircuitDTO, act_study_name: str) -> s_dtos.SbcCircuitDTO:
        """
        SBC component requirements (capacitors, inductor, transformer).

        :param act_study_name: Name of the optuna study
        :type  act_study_name: str
        :param sbc_dto: SBC circuit DTO
        :type sbc_dto: s_dtos.SbcCircuitDTO
        :return: updated SBC circuit DTO
        :rtype: s_dtos.SbcCircuitDTO
        """
        capacitor_1_requirements = HandleSbcDto.generate_capacitor_1_target_requirements(sbc_dto)
        capacitor_2_requirements = HandleSbcDto.generate_capacitor_2_target_requirements(sbc_dto)
        inductor_requirements = HandleSbcDto.generate_inductor_target_requirements(sbc_dto, act_study_name)

        sbc_dto.component_requirements = s_dtos.ComponentRequirements(capacitor_requirements=[capacitor_1_requirements, capacitor_2_requirements],
                                                                      inductor_requirements=[inductor_requirements],
                                                                      transformer_requirements=[])

        return sbc_dto

    @staticmethod
    def generate_inductor_target_requirements(sbc_dto: s_dtos.SbcCircuitDTO, act_study_name: str) -> c_dtos.InductorRequirements:
        """Inductor requirements.

        :param act_study_name: Name of the optuna study
        :type  act_study_name: str
        :param sbc_dto: SBC circuit DTO
        :type sbc_dto: s_dtos.SbcCircuitDTO
        :return: Inductor requirements
        :rtype: InductorRequirements
        """
        # Get the single maximum operating point
        time_vec, i_current_vec, vvp_max_index = HandleSbcDto.get_max_peak_waveform_inductor(sbc_dto, plot=False)
        # Get the data of all operating points
        time_array, i_current_array = HandleSbcDto.get_waveform_inductor(sbc_dto, plot=False)

        inductor_requirements: c_dtos.InductorRequirements = c_dtos.InductorRequirements(
            time_vec=time_vec,
            current_vec=i_current_vec,
            vvp_index=vvp_max_index,
            time_array=time_array,
            current_array=i_current_array,
            study_name=act_study_name,
            target_inductance=sbc_dto.input_config.Ls,
            circuit_id=sbc_dto.circuit_id,
            inductor_number_in_circuit=0,
        )
        return inductor_requirements

    @staticmethod
    def generate_capacitor_1_target_requirements(sbc_dto: s_dtos.SbcCircuitDTO) -> c_dtos.CapacitorRequirements:
        """Capacitor 1 requirements.

        :param sbc_dto: SBC circuit DTO
        :type sbc_dto: c_dtos.SbcCircuitDTO
        :return: capacitor requirements
        :rtype: CapacitorRequirements
        """
        # Get all waveforms
        time_array, i_currents_array = HandleSbcDto.get_waveform_inductor(sbc_dto, plot=False)
        # Calculate the input current i1
        i1_current_array = sbc_dto.input_config.mesh_i2 * sbc_dto.input_config.mesh_duty_cycle
        # Set HS-transistor switch times to necessary format
        t_hs_switch_on = sbc_dto.calc_losses.t_hs_switch_on.reshape(-1, 1)
        t_hs_switch_off = sbc_dto.calc_losses.t_hs_switch_off.reshape(-1, 1)

        # Assemble time array for capacitor.
        # Start period at SWITCH_TIME_OFF/2 (from t0=SWITCH_TIME_OFF/2 to t_period= T_period+SWITCH_TIME_OFF/2)
        time_array_cap = np.hstack((time_array[:, 0][:, np.newaxis],
                                    time_array[:, 1][:, np.newaxis] - t_hs_switch_on / 2 - t_hs_switch_off / 2,
                                    time_array[:, 1][:, np.newaxis] + t_hs_switch_on / 2 - t_hs_switch_off / 2,
                                    time_array[:, 2][:, np.newaxis] - t_hs_switch_off,
                                    time_array[:, 2][:, np.newaxis]))

        # Assemble current vector [-i1(T_period-Ton),i2-i1((Ton+switch time/2) to Ton-Toff)
        i_currents_cap_array = np.hstack(
            (-i1_current_array, -i1_current_array, i_currents_array[:, 0][:, np.newaxis] - i1_current_array,
             i_currents_array[:, 1][:, np.newaxis] - i1_current_array, - i1_current_array))

        # Calculate delta current per operation point which is the slope at switch off the semiconductor (between time point 3 and 4)
        d_i_cap = i_currents_cap_array[:, 3] - i_currents_cap_array[:, 4]

        idx_max = d_i_cap.argmax()
        i_currents_single_vec = i_currents_cap_array[idx_max]
        time_single_vec = time_array_cap[idx_max]

        capacitor_1_requirements = c_dtos.CapacitorRequirements(
            current_vec=i_currents_single_vec.squeeze(),
            time_vec=time_single_vec.squeeze(),
            current_array=i_currents_cap_array.reshape(1, i_currents_cap_array.shape[0], 1, i_currents_cap_array.shape[1]),
            time_array=time_array_cap.reshape(1, time_array_cap.shape[0], 1, time_array_cap.shape[1]),
            v_dc_max=np.max(sbc_dto.input_config.mesh_v1),
            study_name="",
            circuit_id=sbc_dto.circuit_id,
            capacitor_number_in_circuit=0
        )
        return capacitor_1_requirements

    @staticmethod
    def generate_capacitor_2_target_requirements(sbc_dto: s_dtos.SbcCircuitDTO) -> c_dtos.CapacitorRequirements:
        """Capacitor 2 requirements.

        :param sbc_dto: SBC circuit DTO
        :type sbc_dto: c_dtos.SbcCircuitDTO
        :return: capacitor requirements
        :rtype: CapacitorRequirements
        """
        # Get all waveforms
        time_array_cap, i_currents_array = HandleSbcDto.get_waveform_inductor(sbc_dto, plot=False)
        # Calculate current waveform for the capacitor by removing average output current
        i_currents_cap_array = i_currents_array - sbc_dto.input_config.mesh_i2

        # Get the maximum delta current which corresponds to the maximum current
        i_currents_cap_max_vec = i_currents_cap_array.max(axis=1)
        idx_max = i_currents_cap_max_vec.argmax()
        i_currents_single_vec = i_currents_cap_array[idx_max]
        time_single_vec = time_array_cap[idx_max]

        # Calculate v2 from v1
        mesh_v2 = sbc_dto.input_config.mesh_v1 * sbc_dto.input_config.mesh_duty_cycle

        capacitor_2_requirements = c_dtos.CapacitorRequirements(
            current_vec=i_currents_single_vec,
            time_vec=time_single_vec,
            current_array=i_currents_cap_array.reshape(1, i_currents_cap_array.shape[0], 1, i_currents_cap_array.shape[1]),
            time_array=time_array_cap.reshape(1, time_array_cap.shape[0], 1, time_array_cap.shape[1]),
            v_dc_max=np.max(mesh_v2),
            study_name="",
            circuit_id=sbc_dto.circuit_id,
            capacitor_number_in_circuit=1
        )

        return capacitor_2_requirements

    @staticmethod
    def add_inductor_results(sbc_dto: s_dtos.SbcCircuitDTO, inductor_results: c_dtos.InductorResults) -> s_dtos.SbcCircuitDTO:
        """Add inductor results to the CircuitSbcDTO.

        :param sbc_dto: Dual-active bridge DTO
        :type sbc_dto: d_dtos.SbcCircuitDTO
        :param inductor_results: inductor losses
        :type inductor_results: c_dtos.InductorResults
        :return: Dual-active bridge DTO including the inductor losses
        :rtype: d_dtos.SbcCircuitDTO
        """
        sbc_dto.inductor_results = inductor_results

        return sbc_dto

class HandleTransistorDto:
    """Handle the transistor DTOs."""

    @staticmethod
    def tdb_to_transistor_dto(transistor_name: str, c_oss_margin_factor: float = 1.2) -> s_dtos.TransistorDTO:
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

        transistor: tdb.Transistor = db.load_transistor(transistor_name)

        if transistor is None:
            raise ValueError(f"Transistor type: {transistor_name} is of registered in transistor database!")

        # if transistor.type != "MOSFET" and transistor.type != "SiC-MOSFET":
        #     raise ValueError(f"Transistor 1: {transistor.name} is of non-allowed type {transistor.type}. "
        #                      f"Allowed types are MOSFET, SiC-MOSFET.")

        t_j_recommended = transistor.switch.t_j_max - TRANSISTOR_TEMPERATURE_SAFETY_MARGIN

        c_oss, q_oss = HandleSbcDto.get_c_oss_from_tdb(transistor, margin_factor=c_oss_margin_factor)

        # Merge all e_on switch loss data
        switch_e_on_data = HandleTransistorDto.calculate_2D_grid(transistor.switch.e_on)
        if switch_e_on_data.current_data.size == 0:
            raise ValueError(f"For transistor {transistor_name} e_on-switch loss data curve is not available!")
        # Merge all e_off switch loss data
        switch_e_off_data = HandleTransistorDto.calculate_2D_grid(transistor.switch.e_off)
        if switch_e_off_data.current_data.size == 0:
            raise ValueError(f"For transistor {transistor_name} e_off-switch loss data curve is not available!")

        # Calculate r_channel-resistance
        transistor.quickstart_wp()

        transistor_dto = s_dtos.TransistorDTO(
            name=transistor.name,
            t_j_max_op=t_j_recommended,
            c_oss=c_oss,
            q_oss=q_oss,
            switch_e_on_data=switch_e_on_data,
            switch_e_off_data=switch_e_off_data,
            r_th_jc=transistor.switch.thermal_foster.r_th_total,
            cooling_area=transistor.cooling_area,
            housing_area=transistor.housing_area,
            r_channel=transistor.wp.switch_r_channel
        )

        return transistor_dto

    @staticmethod
    def transistor_conduction_loss(transistor_ms_current: NDArray[np.float64],
                                   transistor_dto: s_dtos.TransistorDTO) -> np.ndarray:
        """
        Calculate the transistor conduction losses.

        :param transistor_ms_current: transistor mean square current in A²
        :type transistor_ms_current: np.ndarray
        :param transistor_dto: transistor DTO (Data transfer object)
        :type transistor_dto: pcdt.TransistorDTO
        :return: transistor conduction loss in W
        :rtype: np.ndarray[np.float64]
        """
        return transistor_dto.r_channel * transistor_ms_current

    @staticmethod
    def transistor_switch_loss(mesh_v1: np.ndarray, i_rms: np.ndarray, tr_data_dto: s_dtos.TransistorDTO,
                               fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the transistor conduction losses.

        :param mesh_v1: transistor drain-source voltage in V
        :type  mesh_v1:  np.ndarray[np.float64]
        :param i_rms: current mean root square in A
        :type  i_rms:  np.ndarray[np.float64]
        :param tr_data_dto: transistor DTO (Data transfer object) containing selected transistors
        :type  tr_data_dto:  dtos.TransistorDTO
        :param fs: Switching frequency
        :type  fs: float
        :return: transistor conduction loss in W
        :rtype:  np.ndarray
        """
        # Transform the mesh to mesh-points
        mesh_points = np.hstack((mesh_v1, i_rms))
        # Initialize interpolation object for e-on
        e_on_losses_obj = RGI(
            (tr_data_dto.switch_e_on_data.voltage_parameter, tr_data_dto.switch_e_on_data.current_data),
            tr_data_dto.switch_e_on_data.loss_data)

        # Calculate the loss results for switch on
        e_on_losses = e_on_losses_obj(mesh_points)
        # Calculate the switch time according t_on = 2 E_on / (V_DS * I_D)
        t_switch_on = 2 * e_on_losses / (mesh_v1.flatten() * i_rms.flatten())

        # Initialize interpolation object for e-off
        e_off_losses_obj = RGI(
            (tr_data_dto.switch_e_off_data.voltage_parameter, tr_data_dto.switch_e_off_data.current_data),
            tr_data_dto.switch_e_off_data.loss_data)

        # Calculate the loss results for switch on
        e_off_losses = e_off_losses_obj(mesh_points)
        # Calculate the switch time according t_on = 2 E_on / (V_DS * I_D)
        t_switch_off = 2 * e_off_losses / (mesh_v1.flatten() * i_rms.flatten())

        # Add both energies and calculate the power loss
        p_switch_loss: NDArray = (e_on_losses + e_off_losses) * fs

        # Calculate the sum of transistor switching losses
        return p_switch_loss, t_switch_on, t_switch_off

    @staticmethod
    def calculate_2D_grid(switch_energy_data_list: list[tdb.SwitchEnergyData]) -> s_dtos.LossDataGrid:
        """
        Calculate a 2D-grid with common distances by interpolation of actual transistor data.

        :param switch_energy_data_list: List of parameters and curves of the switch energy loss
        :type switch_energy_data_list: list[tdb.SwitchEnergyData]
        :return: cal2d-Array of the switch loss energy on a homogenous grid
        :rtype: d_dtos.LossDataGrid
        """
        # Variable declaration
        is_data_available: bool = False
        # x-grid list
        unsorted_grid: list = []
        # Common list of losses
        common_losses: list[np.ndarray] = []
        # Switch voltage list
        switch_voltage_list: list[float] = []
        # Curve array
        curve_2D_array: list[tuple[float, np.ndarray]] = []
        # Result data
        result_data: s_dtos.LossDataGrid

        # Overtake cures and collect all x-points
        for entry in switch_energy_data_list:
            # Check if type is correct and graph_i_e is valid
            if entry.dataset_type == "graph_i_e" and entry.graph_i_e is not None:
                # Check if graph_i_e is valid
                if entry.graph_i_e is not None:
                    # Extend the curve by 0,0
                    extend_entry = np.hstack([np.array([[0], [0]]), entry.graph_i_e])
                    # Overtake curve
                    curve_2D_array.append((entry.v_supply, extend_entry))
                    # Add x-axis values
                    unsorted_grid.extend(extend_entry[0])

        # Check, if minimum one curve is available
        if unsorted_grid:

            # Create common current grid based on all curves (merge all x-axis values)
            common_current_grid = np.sort(np.unique(unsorted_grid))

            # As low boundary create a curve with switch voltage = 0V
            currents_0V = np.array([common_current_grid, np.zeros_like(common_current_grid)])
            # curve_2D_array.insert(0, (0, currents_0V))
            curve_2D_array.append((0, currents_0V))
            # Sort the list according
            curve_2D_array = sorted(curve_2D_array, key=lambda x: x[0])

            # Generate the interpolation object for each curve
            for entry in curve_2D_array:
                # Overtake values (entry[0]=switch voltage, entry[1]=graph consists of entry[1][0]=current value, entry[1][1]=loss value
                act_interpole_obj = interp1d(entry[1][0], entry[1][1], kind='linear', bounds_error=False, fill_value='extrapolate')
                common_losses.append(act_interpole_obj(common_current_grid))
                switch_voltage_list.append(entry[0])

            # Assemble result and return variable
            losses_on_common_grid = np.vstack(common_losses)
            switch_voltage_array = np.array(switch_voltage_list)
            result_data = s_dtos.LossDataGrid(voltage_parameter=switch_voltage_array,
                                              loss_data=losses_on_common_grid,
                                              current_data=common_current_grid)
        else:
            result_data = s_dtos.LossDataGrid(voltage_parameter=np.array([]),
                                              loss_data=np.array([]),
                                              current_data=np.array([]))

        return result_data

    @staticmethod
    def generate_thermal_transistor_parameters(circuit_dto: s_dtos.SbcCircuitDTO,
                                               transistor_hs_cooling: c_dtos.ComponentCooling,
                                               transistor_ls_cooling: c_dtos.ComponentCooling) -> s_dtos.SbcCircuitDTO:
        """
        Generate the transistor thermal parameters.

        :param circuit_dto: DAB circuit DTO
        :type  circuit_dto: d_dtos.DabCircuitDTO
        :param transistor_hs_cooling: Transistor cooling
        :type  transistor_hs_cooling: ComponentCooling
        :param transistor_ls_cooling: Transistor cooling
        :type  transistor_ls_cooling: ComponentCooling
        :return:
        """
        if circuit_dto.calc_losses is None:
            raise ValueError("Missing transistor loss calculation of transistors.")

        hs_transistor_cond_loss_matrix = circuit_dto.calc_losses.p_hs_conduction + circuit_dto.calc_losses.p_hs_switch
        ls_transistor_cond_loss_matrix = circuit_dto.calc_losses.p_ls_conduction + circuit_dto.calc_losses.p_ls_switch

        # get all the losses in a matrix
        r_th_copper_coin_1, copper_coin_area_1 = ThermalCalcSupport.calculate_r_th_copper_coin(
            circuit_dto.input_config.transistor_dto_1.cooling_area)
        r_th_copper_coin_2, copper_coin_area_2 = ThermalCalcSupport.calculate_r_th_copper_coin(
            circuit_dto.input_config.transistor_dto_2.cooling_area)

        circuit_r_th_tim_1 = ThermalCalcSupport.calculate_r_th_tim(
            copper_coin_area_1, transistor_hs_cooling)
        circuit_r_th_tim_2 = ThermalCalcSupport.calculate_r_th_tim(
            copper_coin_area_2, transistor_ls_cooling)

        circuit_r_th_1_jhs = circuit_dto.input_config.transistor_dto_1.r_th_jc + r_th_copper_coin_1 + circuit_r_th_tim_1
        circuit_r_th_2_jhs = circuit_dto.input_config.transistor_dto_2.r_th_jc + r_th_copper_coin_2 + circuit_r_th_tim_2

        circuit_heat_sink_max_1_array = (circuit_dto.input_config.transistor_dto_1.t_j_max_op - circuit_r_th_1_jhs * hs_transistor_cond_loss_matrix)
        circuit_heat_sink_max_2_array = (circuit_dto.input_config.transistor_dto_2.t_j_max_op - circuit_r_th_2_jhs * ls_transistor_cond_loss_matrix)

        circuit_dto.circuit_thermal = c_dtos.CircuitThermal(
            t_j_max=[circuit_dto.input_config.transistor_dto_1.t_j_max_op, circuit_dto.input_config.transistor_dto_2.t_j_max_op],
            r_th_jhs=[circuit_r_th_1_jhs, circuit_r_th_2_jhs],
            area=[4 * copper_coin_area_1, 4 * copper_coin_area_2],
            loss_array=[hs_transistor_cond_loss_matrix, ls_transistor_cond_loss_matrix],
            temperature_heat_sink_max_array=[circuit_heat_sink_max_1_array, circuit_heat_sink_max_2_array]
        )

        return circuit_dto
