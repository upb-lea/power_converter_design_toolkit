"""Interval calculation to the phi, tau_1 and tau_2 in radiant for the ZVS switching pattern."""
# python libraries
import logging

# 3rd party libraries
import numpy as np

# own libraries

logger = logging.getLogger(__name__)

# The dict keys this modulation will return
MOD_KEYS = ['phi', 'tau1', 'tau2', 'mask_zvs', 'mask_Im2', 'mask_IIm2',
            'mask_IIIm1', 'zvs_coverage', 'zvs_coverage_notnan', 'mask_m2', 'mask_m1n', 'mask_m1p', 'q_ab_req1', 'q_ab_req2']

def calc_modulation_params(n: np.float64, ls: np.float64, lc1: np.float64, lc2: np.float64, fs: np.ndarray | int | float,
                           c_oss_1: np.ndarray, c_oss_2: np.ndarray,
                           v1: np.ndarray, v2: np.ndarray, power: np.ndarray) -> dict:
    """
    OptZVS (Optimal ZVS) Modulation calculation, which will return phi, tau1 and tau2.

    :param n: Transformer turns ratio n1/n2.
    :type n: float
    :param ls: DAB converter series inductance. (Must not be zero!)
    :type ls: float
    :param lc1: Side 1 commutation inductance. Use np.inf it not present.
    :type lc1: float
    :param lc2: Side 2 commutation inductance. Use np.inf it not present. (Must not be zero!)
    :type lc2: float
    :param fs: Switching frequency, can be a fixed value or a meshgrid with same shape as the other meshes.
    :type fs: float
    :param c_oss_1: Side 1 MOSFET Coss(Vds) curve from Vds=0V to >= V1_max. Just one row with Coss data and index = Vds.
    :type c_oss_1: np.array
    :param c_oss_2: Side 2 MOSFET Coss(Vds) curve from Vds=0V to >= V2_max. Just one row with Coss data and index = Vds.
    :type c_oss_2: np.array
    :param v1: Input voltage meshgrid (voltage on side 1).
    :type v1: np.array
    :param v2: Output voltage meshgrid (voltage on side 2).
    :type v2: np.array
    :param power: DAB input power meshgrid (P=V1*I1).
    :type power: np.array
    :return: dict with phi, tau1, tau2, masks (phi has First-Falling-Edge alignment!)
    """
    # Interval I is named with I and so on
    # Mode 1 is named with m1, Mode 2 = m2
    # leq: less-than-equal
    # g: greater

    # Create empty meshes
    phi = np.full_like(v1, np.nan, dtype=np.float64)
    tau1 = np.full_like(v1, np.nan, dtype=np.float64)
    tau2 = np.full_like(v1, np.nan, dtype=np.float64)
    zvs = np.full_like(v1, np.nan, dtype=np.float64)
    _Im2_mask = np.full_like(v1, False)
    _IIm2_mask = np.full_like(v1, False)
    _IIIm1_part_1_mask = np.full_like(v1, False)

    # Calculate all required values
    # Transform Lc2 to side 1
    Lc2_ = lc2 * n ** 2
    # Transform V2 to side 1
    V2_ = v2 * n
    # For negative P we have to recalculate phi at the end
    _negative_power_mask = np.less(power, 0)
    _positive_power_mask = np.less(0, power)
    I1 = np.abs(power) / v1
    # Convert fs into omega_s
    ws = 2 * np.pi * fs
    # Calculate required Q for each voltage
    # Q_AB_req_1 is the charge required for the whole switching event (Q_A_req_1 + Q_B_req_1)
    # Q_AB_req_2 is the charge required for the whole switching event (Q_A_req_2 + Q_B_req_2)
    Q_AB_req1 = _integrate_c_oss(c_oss_1 * 2, v1)
    Q_AB_req2 = _integrate_c_oss(c_oss_2 * 2, v2)

    # FIXME HACK for testing V1, V2 interchangeability
    # _V1 = V1
    # V1 = V2_
    # V2_ = V1

    # Calculate the Modulations
    # ***** Change in contrast to paper *****
    # Instead of fist checking the limits for each modulation and only calculate each mod. partly,
    # all the modulations are calculated first even for useless areas, and we decide later which part is useful.
    # This should be faster and easier.

    # Int. I (mode 2): calculate phi, tau1 and tau2
    phi_I, tau1_I, tau2_I = _calc_interval_1(n, ls, lc1, Lc2_, ws, Q_AB_req1, Q_AB_req2, v1, V2_, I1)

    # Int. II (mode 2): calculate phi, tau1 and tau2
    phi_II, tau1_II, tau2_II = _calc_interval_2(n, ls, lc1, Lc2_, ws, Q_AB_req1, Q_AB_req2, v1, V2_, I1)

    # Int. III (mode 1): calculate phi, tau1 and tau2
    phi_III, tau1_III, tau2_III, _IIIm1_part_2_mask = _calc_interval_3(n, ls, lc1, Lc2_, ws, Q_AB_req1, Q_AB_req2, v1, V2_, I1)

    # Interval I (mode 2):
    _phi_I_leq_zero_mask = np.less_equal(phi_I, 0)
    _tau1_I_leq_pi_mask = np.less_equal(tau1_I, np.pi)
    _phi_I_g_zero_mask = np.greater(phi_I, 0)
    _Im2_mask = np.bitwise_and(_phi_I_leq_zero_mask, _tau1_I_leq_pi_mask)

    # Interval II (mode 2):
    # helper masks for interval II
    _tau1_II_leq_pi_mask = np.less_equal(tau1_II, np.pi)
    _tau1_II_g_pi_mask = np.greater(tau1_II, np.pi)
    # interval II mask
    _IIm2_mask = np.bitwise_and(_tau1_II_leq_pi_mask, _phi_I_g_zero_mask)

    # Int. III (mode 1):
    # Interval III helper mask
    _tau2_III_leq_pi_mask = np.less_equal(tau2_III, np.pi)
    # Interval III part 1 mask
    # fix here: _phi_I_g_zero_mask was missing!!!
    _IIIm1_part_1_mask = np.bitwise_and(_tau2_III_leq_pi_mask, _tau1_II_g_pi_mask, _phi_I_g_zero_mask)
    # Interval III part 2 mask
    # from interval 3. Must be inside the function, as tau2 is manipulated inside this function and a calculation afterward is not possible

    # Interval III mode 1 Part 2 (set tau_2 = pi and recalculate phi)
    phi[_IIIm1_part_2_mask] = phi_III[_IIIm1_part_2_mask]
    tau1[_IIIm1_part_2_mask] = tau1_III[_IIIm1_part_2_mask]
    tau2[_IIIm1_part_2_mask] = tau2_III[_IIIm1_part_2_mask]
    zvs[_IIIm1_part_2_mask] = True

    # Interval III mode 1 Part 1
    phi[_IIIm1_part_1_mask] = phi_III[_IIIm1_part_1_mask]
    tau1[_IIIm1_part_1_mask] = tau1_III[_IIIm1_part_1_mask]
    tau2[_IIIm1_part_1_mask] = tau2_III[_IIIm1_part_1_mask]
    zvs[_IIIm1_part_1_mask] = True

    # Interval II mask
    phi[_IIm2_mask] = phi_II[_IIm2_mask]
    tau1[_IIm2_mask] = tau1_II[_IIm2_mask]
    tau2[_IIm2_mask] = tau2_II[_IIm2_mask]
    zvs[_IIm2_mask] = True

    # ZVS is analytically IMPOSSIBLE!
    zvs_impossible_mask = np.bitwise_and(_phi_I_leq_zero_mask, np.bitwise_not(_tau1_I_leq_pi_mask))
    zvs[zvs_impossible_mask] = False
    phi[zvs_impossible_mask] = np.nan
    tau1[zvs_impossible_mask] = np.nan
    tau2[zvs_impossible_mask] = np.nan

    # Interval I mask
    phi[_Im2_mask] = phi_I[_Im2_mask]
    tau1[_Im2_mask] = tau1_I[_Im2_mask]
    tau2[_Im2_mask] = tau2_I[_Im2_mask]
    zvs[_Im2_mask] = True

    # Recalculate phi for negative power
    phi_nP = -(tau1 + phi - tau2)
    phi[_negative_power_mask] = phi_nP[_negative_power_mask]

    # Initialize return dict
    da_mod_results: dict[str, np.ndarray | float | None] = dict()
    # Save the results in the dict
    # Convert phi because the math from the paper uses Middle-Pulse alignment, but we use First-Falling-Edge alignment!
    da_mod_results[MOD_KEYS[0]] = phi
    da_mod_results[MOD_KEYS[1]] = tau1
    da_mod_results[MOD_KEYS[2]] = tau2
    da_mod_results[MOD_KEYS[3]] = zvs
    da_mod_results[MOD_KEYS[4]] = _Im2_mask
    da_mod_results[MOD_KEYS[5]] = _IIm2_mask
    da_mod_results[MOD_KEYS[6]] = np.bitwise_or(_IIIm1_part_1_mask, _IIIm1_part_2_mask)

    # ZVS coverage based on calculation: Percentage ZVS based on all points (full operating range)
    da_mod_results[MOD_KEYS[7]] = np.count_nonzero(zvs) / np.size(zvs)

    # ZVS coverage based on calculation: Percentage ZVS based on all points where the converter can be operated (not full operating range)
    # ignore error in case of NaN. NaN invalid results will be ignored by the Pareto algorithm anyway
    with np.errstate(invalid='ignore'):
        da_mod_results[MOD_KEYS[8]] = np.count_nonzero(zvs[~np.isnan(tau1)]) / np.size(zvs[~np.isnan(tau1)])

    # lower power positive/negative mode 2
    da_mod_results[MOD_KEYS[9]] = np.bitwise_or(_Im2_mask, _IIm2_mask)
    # negative high power mode 1-
    da_mod_results[MOD_KEYS[10]] = np.bitwise_and(_negative_power_mask, np.bitwise_or(_IIIm1_part_1_mask, _IIIm1_part_2_mask))
    # positive high power mode 1+
    da_mod_results[MOD_KEYS[11]] = np.bitwise_and(_positive_power_mask, np.bitwise_or(_IIIm1_part_1_mask, _IIIm1_part_2_mask))

    # add required charge
    da_mod_results[MOD_KEYS[12]] = Q_AB_req1
    da_mod_results[MOD_KEYS[13]] = Q_AB_req2

    return da_mod_results

def _calc_interval_1(n: np.float64, l_s: np.float64, l_c_b1: np.float64, l_c_b2_: np.float64,
                     omega_s: np.ndarray | int | float, q_ab_req_b1: np.ndarray, q_ab_req_b2: np.ndarray,
                     v_b1: np.ndarray, v_b2_: np.ndarray, i_b1: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mode 2 Modulation (interval I) calculation, which will return phi_rad, tau_1_rad and tau_2_rad in rad.

    :param n: transformer ratio
    :type n: np.float64
    :param l_s: series inductance
    :type l_s: np.float64
    :param l_c_b1: commutation inductance L_1
    :type l_c_b1: np.float64
    :param l_c_b2_: primary side reflected commutation inductance L_2
    :type l_c_b2_: np.float64
    :param omega_s: omega_s = 2 * pi * f
    :type omega_s: float
    :param q_ab_req_b1: required coss charge for A/B-interval of bridge 1 semiconductors
    :type q_ab_req_b1: float
    :param q_ab_req_b2: required coss charge for A/B-interval of bridge 2 semiconductors
    :type q_ab_req_b2: float
    :param v_b1: DC voltage of bridge 1
    :type v_b1: float
    :param v_b2_: DC voltage of bridge 2
    :type v_b2_: float
    :param i_b1: current out of bridge 1 at the time of transistor turn-off
    :type i_b1: float
    """
    # Predefined Terms
    e1 = v_b2_ * q_ab_req_b2 * omega_s

    e2 = n * v_b1 * np.pi * i_b1

    # FIXME e3 gets negative for all values n*V2 < V1, why? Formula is checked against PhD.
    # TODO Maybe Ls is too small? Is that even possible? Error in Formula?
    e3 = n * (v_b2_ * (l_c_b2_ + l_s) - v_b1 * l_c_b2_)
    if np.any(np.less(e3, 0)):
        logger.debug('Something is wrong. Formula e3 is negative and it should not!')
        logger.debug('Please check your DAB Params, probably you must check n or iterate L, Lc1, Lc2.')

    e4 = 2 * n * np.sqrt(q_ab_req_b1 * l_s * np.power(omega_s, 2) * v_b1 * l_c_b1 * (l_c_b1 + l_s))

    e5 = l_s * l_c_b2_ * omega_s * (e2 + 2 * e1 + 2 * np.sqrt(e1 * (e1 + e2)))

    # Solution for interval I (mode 2)
    # ignore error in case of NaN. NaN invalid results will be ignored by the Pareto algorithm anyway
    with np.errstate(invalid='ignore'):
        tau_1_rad = (np.sqrt(2) * (l_c_b1 * np.sqrt(v_b2_ * e3 * e5) + e4 * e3 * 1 / n)) / (v_b1 * e3 * (l_c_b1 + l_s))

    # ignore error in case of NaN. NaN invalid results will be ignored by the Pareto algorithm anyway
    with np.errstate(invalid='ignore'):
        tau_2_rad = np.sqrt((2 * e5) / (v_b2_ * e3))

    phi_rad = (tau_2_rad - tau_1_rad) / 2 + (i_b1 * omega_s * l_s * np.pi) / (tau_2_rad * v_b2_)

    return phi_rad, tau_1_rad, tau_2_rad


def _calc_interval_2(n: np.float64, l_s: np.float64, l_c_b1: np.float64, l_c_b2_: np.float64,
                     omega_s: np.ndarray | int | float, q_ab_req_b1: np.ndarray, q_ab_req_b2: np.ndarray,
                     v_b1: np.ndarray, v_b2_: np.ndarray, i_b1: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mode 2 Modulation (interval II) calculation, which will return phi_rad, tau_1_rad and tau_2_rad in rad.

    :param n: transformer ratio
    :type n: np.float64
    :param l_s: series inductance
    :type l_s: np.float64
    :param l_c_b1: commutation inductance L_1
    :type l_c_b1: np.float64
    :param l_c_b2_: primary side reflected commutation inductance L_2
    :type l_c_b2_: np.float64
    :param omega_s: omega_s = 2 * pi * f
    :type omega_s: float
    :param q_ab_req_b1: required coss charge for A/B-interval of bridge 1 semiconductors
    :type q_ab_req_b1: float
    :param q_ab_req_b2: required coss charge for A/B-interval of bridge 2 semiconductors
    :type q_ab_req_b2: float
    :param v_b1: DC voltage of bridge 1
    :type v_b1: float
    :param v_b2_: DC voltage of bridge 2
    :type v_b2_: float
    :param i_b1: current out of bridge 1 at the time of transistor turn-off
    :type i_b1: float
    """
    # Predefined Terms
    e1 = v_b2_ * q_ab_req_b2 * omega_s

    e2 = n * v_b1 * np.pi * i_b1

    e3 = n * (v_b2_ * (l_c_b2_ + l_s) - v_b1 * l_c_b2_)

    e4 = 2 * n * np.sqrt(q_ab_req_b1 * l_s * np.power(omega_s, 2) * v_b1 * l_c_b1 * (l_c_b1 + l_s))

    e5 = l_s * l_c_b2_ * omega_s * (e2 + 2 * e1 + 2 * np.sqrt(e1 * (e1 + e2)))

    # Solution for interval II (mode 2)
    # ignore error in case of NaN. NaN invalid results will be ignored by the Pareto algorithm anyway
    with np.errstate(invalid='ignore'):
        tau_1_rad = (np.sqrt(2) * (e5 + omega_s * l_s * l_c_b2_ * e2 * (v_b2_ / v_b1 * (l_s / l_c_b2_ + 1) - 1))) / (np.sqrt(v_b2_ * e3 * e5))

    # ignore error in case of NaN. NaN invalid results will be ignored by the Pareto algorithm anyway
    with np.errstate(invalid='ignore'):
        tau_2_rad = np.sqrt((2 * e5) / (v_b2_ * e3))

    phi_rad = np.full_like(v_b1, 0)

    return phi_rad, tau_1_rad, tau_2_rad


def _calc_interval_3(n: np.float64, l_s: np.float64, l_c_b1: np.float64, l_c_b2_: np.float64,
                     omega_s: np.ndarray | int | float, q_ab_req_b1: np.ndarray, q_ab_req_b2: np.ndarray,
                     v_b1: np.ndarray, v_b2_: np.ndarray, i_b1: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Mode 1 Modulation (interval III) calculation, which will return phi_rad, tau_1_rad and tau_2_rad.

    :param n: transformer ratio
    :type n: np.float64
    :param l_s: series inductance
    :type l_s: np.float64
    :param l_c_b1: commutation inductance L_1
    :type l_c_b1: np.float64
    :param l_c_b2_: primary side reflected commutation inductance L_2
    :type l_c_b2_: np.float64
    :param omega_s: omega_s = 2 * pi * f
    :type omega_s: float
    :param q_ab_req_b1: required coss charge for A/B-interval of bridge 1 semiconductors
    :type q_ab_req_b1: float
    :param q_ab_req_b2: required coss charge for A/B-interval of bridge 2 semiconductors
    :type q_ab_req_b2: float
    :param v_b1: DC voltage of bridge 1
    :type v_b1: float
    :param v_b2_: DC voltage of bridge 2
    :type v_b2_: float
    :param i_b1: current out of bridge 1 at the time of transistor turn-off
    :type i_b1: float
    """
    # Predefined Terms
    e1 = v_b2_ * q_ab_req_b2 * omega_s

    e2 = n * v_b1 * np.pi * i_b1

    e3 = n * (v_b2_ * (l_c_b2_ + l_s) - v_b1 * l_c_b2_)

    e4 = 2 * n * np.sqrt(q_ab_req_b1 * l_s * np.power(omega_s, 2) * v_b1 * l_c_b1 * (l_c_b1 + l_s))

    e5 = l_s * l_c_b2_ * omega_s * (e2 + 2 * e1 + 2 * np.sqrt(e1 * (e1 + e2)))

    # Solution for interval III (mode 1)
    tau_1_rad = np.full_like(v_b1, np.pi)

    # ignore error in case of NaN. NaN invalid results will be ignored by the Pareto algorithm anyway
    with np.errstate(invalid='ignore'):
        tau_2_rad = np.sqrt((2 * e5) / (v_b2_ * e3))

    sqrt_part = (- np.power((tau_2_rad - np.pi), 2) + tau_1_rad * (2 * np.pi - tau_1_rad)) / 4 - (i_b1 * omega_s * l_s * np.pi) / v_b2_

    # in case of phi = NaN, ignore the warning
    # invalid results will be ignored by the Pareto algorithm
    with np.errstate(invalid='ignore'):
        phi_rad = (- tau_1_rad + tau_2_rad + np.pi) / 2 - np.sqrt(sqrt_part)

    # Check if tau_2_rad > pi: Set tau_2_rad = pi and recalculate phi_rad for these points
    tau2_III_g_pi_mask = np.greater(tau_2_rad, np.pi)
    tau2_III_part_2 = np.full_like(v_b1, np.pi)
    # ignore error in case of NaN. NaN invalid results will be ignored by the Pareto algorithm anyway
    with np.errstate(invalid='ignore'):
        phi_III_part_2 = (- tau_1_rad + tau2_III_part_2 + np.pi) / 2 - np.sqrt(
            (- np.power((tau2_III_part_2 - np.pi), 2) + tau_1_rad * (2 * np.pi - tau_1_rad)) / 4 - (i_b1 * omega_s * l_s * np.pi) / v_b2_)
    tau_2_rad[tau2_III_g_pi_mask] = tau2_III_part_2[tau2_III_g_pi_mask]
    phi_rad[tau2_III_g_pi_mask] = phi_III_part_2[tau2_III_g_pi_mask]

    return phi_rad, tau_1_rad, tau_2_rad, tau2_III_g_pi_mask


def _integrate_c_oss(coss: np.ndarray, voltage: np.ndarray) -> np.ndarray:
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
    # get a qoss vector that has the resolution 1V from 0 to V_max
    v_vec = np.arange(coss.shape[0])
    # get a qoss vector that fits the mesh_V scale
    # v_vec = np.linspace(V_min, V_max, int(V_step))
    qoss = coss_int(v_vec)

    # Calculate a qoss mesh that is like the V mesh
    # Each element in V gets its q(v) value
    def meshing_q(v):
        return np.interp(v, v_vec, qoss)

    q_meshgrid = np.vectorize(meshing_q)
    qoss_mesh: np.ndarray = q_meshgrid(voltage)

    return qoss_mesh
