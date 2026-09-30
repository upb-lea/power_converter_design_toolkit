"""Data generation for lab prototypes."""

# Python libraries
import os
import logging
import shutil
import subprocess

# 3rd party libraries
import pandas as pd
import femmt as fmt

# own libraries
import pcdt.toml_checker as tc
from pcdt import CapacitorConfiguration, InductorConfiguration, TransformerConfiguration, StudyData, CircuitOptimizationBase
from pcdt.constant_path import (DF_SUMMARY_FINAL_FILTERED_MEAN_LOSS, FILTERED_RESULTS_PATH, SUMMARY_COMBINATION_FOLDER,
                                CIRCUIT_INDUCTOR_FEM_LOSSES_FOLDER, CIRCUIT_TRANSFORMER_FEM_LOSSES_FOLDER, CAPACITOR_RESULTS,
                                DATA_GENERATION_WAVEFORM_FOLDER)
from pcdt.constants import FACTOR_M_TO_MM

logger = logging.getLogger(__name__)

class ManufactureDataGeneration:
    """Generate manufacturing data."""

    @staticmethod
    def _run_freecad(freecad_script_file, output_file, variables=None):
        """
        Run a FreeCAD Python script from the command line to export a STEP file.

        :param freecad_script_file: Path to the FreeCAD Python script (.py)
        :type freecad_script_file : str
        :param output_file: output STEP file path
        :type output_file: str
        :param variables: Optional parameters passed to the FreeCAD script as environment variables. Keys are converted to uppercase.
        :type variables: dict | None

        Example:
        {
            "core_inner_diameter_mm": 16.0,
            "l_air_gap_mm": 0.8
        }

        becomes:
        CORE_INNER_DIAMETER_MM=16.0
        L_AIR_GAP_MM=0.8
        """
        # Check whether the FreeCAD script exists.
        if not os.path.isfile(freecad_script_file):
            logger.error("Error: %s not found.", freecad_script_file)
            return False

        # Normalize paths.
        freecad_script_file = os.path.abspath(freecad_script_file)
        output_file = os.path.abspath(output_file)

        # Create the output folder when required.
        output_directory = os.path.dirname(output_file)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)

        # Copy the current environment to retain PATH, FreeCAD libraries, etc.
        environment = os.environ.copy()

        # This environment variable is read by the FreeCAD script.
        environment["OUTPUT_STEP_FILE"] = output_file

        # Pass optional script parameters through environment variables.
        if variables:
            for key, value in variables.items():
                environment[key.upper()] = str(value)

        # command option 1: FreeCADCmd (official command line interface)
        # command option 2: freecad.cmd (e.g. used in snap packages)
        # note: freecad -c ends in the freecad command line
        cmd_1 = [
            "FreeCADCmd",
            freecad_script_file
        ]
        cmd_2 = [
            "freecad.cmd",
            freecad_script_file
        ]
        if variables:
            logger.info(
                "FreeCAD parameters: %s",
                ", ".join(
                    f"{key.upper()}={value}"
                    for key, value in variables.items()
                )
            )

        try:
            logger.info("Running: %s", " ".join(cmd_1))
            result = subprocess.run(
                cmd_1,
                check=True,
                capture_output=True,
                text=True,
                env=environment
            )

            if result.stdout:
                logger.debug("FreeCAD output:\n%s", result.stdout)

            if result.stderr:
                # FreeCAD can write non-fatal messages to stderr.
                logger.warning("FreeCAD messages:\n%s", result.stderr)

            if not os.path.isfile(output_file):
                logger.error(
                    "FreeCAD completed without error, but no STEP file was found: %s",
                    output_file
                )
                return False

            logger.info("Success! STEP file saved to: %s", output_file)
            return True

        except FileNotFoundError:
            logger.info(f"{cmd_1[0]} does not work on this system, try {cmd_2[0]} instead.")
            logger.info("Running: %s", " ".join(cmd_2))
            try:
                result = subprocess.run(
                    cmd_2,
                    check=True,
                    capture_output=True,
                    text=True,
                    env=environment
                )

                if result.stdout:
                    logger.debug("FreeCAD output:\n%s", result.stdout)

                if result.stderr:
                    # FreeCAD can write non-fatal messages to stderr.
                    logger.warning("FreeCAD messages:\n%s", result.stderr)

                if not os.path.isfile(output_file):
                    logger.error(
                        "FreeCAD completed without error, but no STEP file was found: %s",
                        output_file
                    )
                    return False

                logger.info("Success! STEP file saved to: %s", output_file)
                return True

            except FileNotFoundError:
                logger.error(
                    "Error: 'FreeCADCmd' command not found. "
                    "Install FreeCAD or add FreeCADCmd to your PATH."
                )
                return False

            except subprocess.CalledProcessError as error:
                logger.error(
                    "FreeCAD exited with return code %s.",
                    error.returncode
                )

                if error.stdout:
                    logger.error("FreeCAD stdout:\n%s", error.stdout)

                if error.stderr:
                    logger.error("FreeCAD stderr:\n%s", error.stderr)

                return False

            except Exception:
                logger.exception("Unexpected error while running FreeCAD.")
                return False

        except subprocess.CalledProcessError as error:
            logger.error(
                "FreeCAD exited with return code %s.",
                error.returncode
            )

            if error.stdout:
                logger.error("FreeCAD stdout:\n%s", error.stdout)

            if error.stderr:
                logger.error("FreeCAD stderr:\n%s", error.stderr)

            return False

        except Exception:
            logger.exception("Unexpected error while running FreeCAD.")
            return False

    @staticmethod
    def _read_summary_parameters(combination_id: int, df: pd.DataFrame) -> tuple[int, list[int], list[int], list[int], int]:
        """
        Read component IDs from the summary file.

        :param combination_id: combination ID
        :type combination_id: int
        :param df: summary dataframe
        :type df: pd.DataFrame
        """
        capacitor_id_list = []
        inductor_id_list = []
        transformer_id_list = []

        circuit_id = df.loc[df['combination_id'] == combination_id, 'circuit_id'].values[0]

        # get the maximum number of capacitors
        count_capacitors_in_circuit = df.columns.str.fullmatch(r"capacitor_id_\d").sum()
        count_inductors_in_circuit = df.columns.str.fullmatch(r"inductor_id_\d").sum()
        count_transformers_in_circuit = df.columns.str.fullmatch(r"transformer_id_\d").sum()
        
        for count in list(range(count_capacitors_in_circuit)):
            capacitor_id_list.append(df.loc[df['combination_id'] == combination_id, f'capacitor_id_{count}'].values[0])
        for count in list(range(count_inductors_in_circuit)):
            inductor_id_list.append(df.loc[df['combination_id'] == combination_id, f'inductor_id_{count}'].values[0])
        for count in list(range(count_transformers_in_circuit)):
            transformer_id_list.append(df.loc[df['combination_id'] == combination_id, f'transformer_id_{count}'].values[0])
        heat_sink_id = df.loc[df['combination_id'] == combination_id, 'heat_sink_id'].values[0]

        return circuit_id, capacitor_id_list, inductor_id_list, transformer_id_list, heat_sink_id

    @staticmethod
    def _generate_circuit_data(circuit_id: int, df_circuit: pd.DataFrame, output_filepath: str) -> None:
        """
        Generate circuit manufacturing data.

        :param circuit_id: circuit ID
        :type circuit_id: int
        :param df_circuit: circuit dataframe
        :type df_circuit: pd.DataFrame
        :param output_filepath: output filepath
        :type output_filepath: str
        """
        frequency = df_circuit.loc[df_circuit['number'] == circuit_id, 'params_f_s_suggest'].values[0]
        l_1 = df_circuit.loc[df_circuit['number'] == circuit_id, 'params_l_1_suggest'].values[0]
        l_2_ = df_circuit.loc[df_circuit['number'] == circuit_id, 'params_l_2__suggest'].values[0]
        l_s = df_circuit.loc[df_circuit['number'] == circuit_id, 'params_l_s_suggest'].values[0]
        n = df_circuit.loc[df_circuit['number'] == circuit_id, 'params_n_suggest'].values[0]
        transistor_1 = df_circuit.loc[df_circuit['number'] == circuit_id, 'params_transistor_1_name_suggest'].values[0]
        transistor_2 = df_circuit.loc[df_circuit['number'] == circuit_id, 'params_transistor_2_name_suggest'].values[0]

        circuit_data = (f"{frequency=}\n"
                        f"{l_1=}\n"
                        f"{l_2_=}\n"
                        f"{l_s=}\n"
                        f"{n=}\n"
                        f"{transistor_1=}\n"
                        f"{transistor_2=}\n"
                        )
        with open(f"{output_filepath}/circuit_data.txt", "w", encoding="utf-8") as f:
            f.write(circuit_data)

    @staticmethod
    def _generate_capacitor_data(capacitor_id: int, df_capacitor: pd.DataFrame, output_filepath: str, capacitor_number: int) -> None:
        """
        Generate inductor manufacturing data.

        :param capacitor_id: inductor ID
        :type capacitor_id: int
        :param df_capacitor: inductor dataframe
        :type df_capacitor: pd.DataFrame
        :param output_filepath: output filepath
        :type output_filepath: str
        :param capacitor_number: capacitor number in circuit, e.g. 0 or 1
        :type capacitor_number: int
        """
        ordering_code = df_capacitor.loc[df_capacitor['ordering code'] == capacitor_id, 'ordering code'].values[0]
        in_series_needed = df_capacitor.loc[df_capacitor['ordering code'] == capacitor_id, 'in_series_needed'].values[0]
        in_parallel_needed = df_capacitor.loc[df_capacitor['ordering code'] == capacitor_id, 'in_parallel_needed'].values[0]

        capacitor_data = (f"{ordering_code=}\n"
                          f"{in_series_needed=}\n"
                          f"{in_parallel_needed=}\n"
                          )
        with open(f"{output_filepath}/capacitor_{capacitor_number}_data.txt", "w", encoding="utf-8") as f:
            f.write(capacitor_data)

    @staticmethod
    def _generate_inductor_data(inductor_id: int, df_inductor: pd.DataFrame, output_filepath: str, inductor_number: int,
                                inductor_insulations: tc.TomlInductorInsulation) -> None:
        """
        Generate inductor manufacturing data.

        :param inductor_id: inductor ID
        :type inductor_id: int
        :param df_inductor: inductor dataframe
        :type df_inductor: pd.DataFrame
        :param output_filepath: output filepath
        :type output_filepath: str
        :param inductor_number: number of the inductor in circuit
        :type inductor_number: int
        """
        params_core_name = df_inductor.loc[df_inductor['number'] == inductor_id, 'params_core_name'].values[0]
        params_litz_wire_name = df_inductor.loc[df_inductor['number'] == inductor_id, 'params_litz_wire_name'].values[0]
        params_material_name = df_inductor.loc[df_inductor['number'] == inductor_id, 'params_material_name'].values[0]
        params_turns = df_inductor.loc[df_inductor['number'] == inductor_id, 'params_turns'].values[0]
        params_window_h = df_inductor.loc[df_inductor['number'] == inductor_id, 'params_window_h'].values[0]
        user_attrs_core_inner_diameter = df_inductor.loc[df_inductor['number'] == inductor_id, 'user_attrs_core_inner_diameter'].values[0]
        user_attrs_dynamic_mu_r_abs = df_inductor.loc[df_inductor['number'] == inductor_id, 'user_attrs_dynamic_mu_r_abs'].values[0]
        user_attrs_flux_density_peak = df_inductor.loc[df_inductor['number'] == inductor_id, 'user_attrs_flux_density_peak'].values[0]
        user_attrs_l_air_gap = df_inductor.loc[df_inductor['number'] == inductor_id, 'user_attrs_l_air_gap'].values[0]
        user_attrs_window_w = df_inductor.loc[df_inductor['number'] == inductor_id, 'user_attrs_window_w'].values[0]

        inductor_data = (f"{params_core_name=}\n"
                         f"{params_litz_wire_name=}\n"
                         f"{params_material_name=}\n"
                         f"{params_turns=}\n"
                         f"{params_window_h=}\n"
                         f"{user_attrs_core_inner_diameter=}\n"
                         f"{user_attrs_dynamic_mu_r_abs=}\n"
                         f"{user_attrs_flux_density_peak=}\n"
                         f"{user_attrs_l_air_gap=}\n"
                         f"{user_attrs_window_w=}\n"
                         )
        with open(f"{output_filepath}/inductor_{inductor_number}_data.txt", "w", encoding="utf-8") as f:
            f.write(inductor_data)

        # PQ core step file generation
        core = fmt.core_database()[params_core_name]

        pq_core_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "freecad_models/pq_core_half.py")

        top_bottom_yoke_height = (core["core_h"] - core["window_h"]) / 2

        # Assemble file name
        target_file_path = os.path.join(output_filepath, f"inductor_{inductor_number}_core.step")
        # Create file, if it does not exist
        if not os.path.isfile(target_file_path):
            success = ManufactureDataGeneration._run_freecad(
                freecad_script_file=pq_core_filepath,
                output_file=target_file_path,
                variables={
                    "core_h_mm": (params_window_h + 2 * top_bottom_yoke_height) * FACTOR_M_TO_MM,
                    "core_inner_diameter_mm": user_attrs_core_inner_diameter * FACTOR_M_TO_MM,
                    "window_h_mm": params_window_h * FACTOR_M_TO_MM,
                    "window_w_mm": user_attrs_window_w * FACTOR_M_TO_MM,
                    "core_dimension_x_mm": core["core_dimension_x"] * FACTOR_M_TO_MM,
                    "core_dimension_y_mm": core["core_dimension_y"] * FACTOR_M_TO_MM,
                    "l_air_gap_mm": user_attrs_l_air_gap * FACTOR_M_TO_MM,
                    "save_fcstd_file": "0"
                }
            )
        else:
            success = True

        if not success:
            logger.warning(f"Inductor ID {inductor_id} custom core STEP export failed.")

        # Assemble file name
        target_file_path = os.path.join(output_filepath, f"inductor_{inductor_number}_core_original.step")
        # Create file, if it does not exist
        if not os.path.isfile(target_file_path):
            success = ManufactureDataGeneration._run_freecad(
                freecad_script_file=pq_core_filepath,
                output_file=target_file_path,
                variables={
                    "core_h_mm": core["core_h"] * FACTOR_M_TO_MM,
                    "core_inner_diameter_mm": core["core_inner_diameter"] * FACTOR_M_TO_MM,
                    "window_h_mm": core["window_h"] * FACTOR_M_TO_MM,
                    "window_w_mm": core["window_w"] * FACTOR_M_TO_MM,
                    "core_dimension_x_mm": core["core_dimension_x"] * FACTOR_M_TO_MM,
                    "core_dimension_y_mm": core["core_dimension_y"] * FACTOR_M_TO_MM,
                    "l_air_gap_mm": 0,
                    "save_fcstd_file": "0"
                }
            )
        else:
            success = True

        if not success:
            logger.warning(f"Inductor ID {inductor_id} pq core original dimensions STEP export failed.")

        bobbin_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "freecad_models/pq_bobbin.py")
        clearance_mm = 0.3

        # Assemble file name
        target_file_path = os.path.join(output_filepath, f"inductor_{inductor_number}_bobbin.step")
        # Create file, if it does not exist
        if not os.path.isfile(target_file_path):
            # generate inductor bobbin
            success = ManufactureDataGeneration._run_freecad(
                freecad_script_file=bobbin_filepath,
                output_file=target_file_path,

                variables={
                    "window_h_mm": params_window_h * FACTOR_M_TO_MM,
                    "window_w_mm": user_attrs_window_w * FACTOR_M_TO_MM,
                    "core_inner_diameter_mm": user_attrs_core_inner_diameter * FACTOR_M_TO_MM,

                    # Bobbin dimensions
                    "flange_thickness_inner_mm": inductor_insulations.core_left * FACTOR_M_TO_MM - clearance_mm,
                    "flange_thickness_top_mm": inductor_insulations.core_top * FACTOR_M_TO_MM - clearance_mm,
                    "flange_thickness_bot_mm": inductor_insulations.core_bot * FACTOR_M_TO_MM - clearance_mm,

                    "clearance": clearance_mm,
                    "inner_edge_radius": 0.6,
                    "outer_edge_radius": 0.6,
                    "enable_wire_slots": True,
                    "wire_slots_position": "both",
                    "wire_slot_width": 4.0
                }
            )
        else:
            success = True

        if not success:
            logger.warning(f"Inductor ID {inductor_id} bobbin STEP export failed.")

    @staticmethod
    def _generate_transformer_data(transformer_id: int, df_transformer: pd.DataFrame, output_filepath: str, transformer_number: int,
                                   transformer_insulations: tc.TomlTransformerInsulation) -> None:
        """
        Generate transformer manufacturing data.

        :param transformer_id: transformer ID
        :type transformer_id: int
        :param df_transformer: transformer dataframe
        :type df_transformer: pd.DataFrame
        :param output_filepath: output filepath
        :type output_filepath: str
        :param transformer_number: number of the transformer in circuit
        :type transformer_number: int
        """
        params_core_name = df_transformer.loc[df_transformer['number'] == transformer_id, 'params_core_name'].values[0]
        params_material_name = df_transformer.loc[df_transformer['number'] == transformer_id, 'params_material_name'].values[0]
        params_n_p_bot = df_transformer.loc[df_transformer['number'] == transformer_id, 'params_n_p_bot'].values[0]
        params_n_p_top = df_transformer.loc[df_transformer['number'] == transformer_id, 'params_n_p_top'].values[0]
        params_n_s_bot = df_transformer.loc[df_transformer['number'] == transformer_id, 'params_n_s_bot'].values[0]
        params_primary_litz_name = df_transformer.loc[df_transformer['number'] == transformer_id, 'params_primary_litz_name'].values[0]
        params_secondary_litz_name = df_transformer.loc[df_transformer['number'] == transformer_id, 'params_secondary_litz_name'].values[0]
        params_window_h_bot = df_transformer.loc[df_transformer['number'] == transformer_id, 'params_window_h_bot'].values[0]
        user_attrs_core_inner_diameter = df_transformer.loc[df_transformer['number'] == transformer_id, 'user_attrs_core_inner_diameter'].values[0]
        user_attrs_l_bot_air_gap = df_transformer.loc[df_transformer['number'] == transformer_id, 'user_attrs_l_bot_air_gap'].values[0]
        user_attrs_l_top_air_gap = df_transformer.loc[df_transformer['number'] == transformer_id, 'user_attrs_l_top_air_gap'].values[0]
        user_attrs_window_h_bot = df_transformer.loc[df_transformer['number'] == transformer_id, 'user_attrs_window_h_bot'].values[0]
        user_attrs_window_h_top = df_transformer.loc[df_transformer['number'] == transformer_id, 'user_attrs_window_h_top'].values[0]
        user_attrs_window_w = df_transformer.loc[df_transformer['number'] == transformer_id, 'user_attrs_window_w'].values[0]

        transformer_data = (f"{params_core_name=}\n"
                            f"{params_material_name=}\n"
                            f"{params_n_p_bot=}\n"
                            f"{params_n_p_top=}\n"
                            f"{params_n_s_bot=}\n"
                            f"{params_primary_litz_name=}\n"
                            f"{params_secondary_litz_name=}\n"
                            f"{params_window_h_bot=}\n"
                            f"{user_attrs_core_inner_diameter=}\n"
                            f"{user_attrs_l_bot_air_gap=}\n"
                            f"{user_attrs_l_top_air_gap=}\n"
                            f"{user_attrs_window_h_bot=}\n"
                            f"{user_attrs_window_h_top=}\n"
                            f"{user_attrs_window_w=}\n"

                            )
        with open(f"{output_filepath}/transformer_{transformer_number}_data.txt", "w", encoding="utf-8") as f:
            f.write(transformer_data)

        core = fmt.core_database()[params_core_name]

        top_bottom_yoke_height = (core["core_h"] - core["window_h"]) / 2

        pq_core_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "freecad_models/pq_core_half.py")

        # Assemble file name
        target_file_path = os.path.join(output_filepath, f"transformer_{transformer_number}_core_lower.step")
        # Create file, if it does not exist
        if not os.path.isfile(target_file_path):
            success = ManufactureDataGeneration._run_freecad(
                freecad_script_file=pq_core_filepath,
                output_file=target_file_path,
                variables={
                    "core_h_mm": (user_attrs_window_h_bot + 2 * top_bottom_yoke_height) * FACTOR_M_TO_MM,
                    "core_inner_diameter_mm": user_attrs_core_inner_diameter * FACTOR_M_TO_MM,
                    "window_h_mm": params_window_h_bot * FACTOR_M_TO_MM,
                    "window_w_mm": user_attrs_window_w * FACTOR_M_TO_MM,
                    "core_dimension_x_mm": core["core_dimension_x"] * FACTOR_M_TO_MM,
                    "core_dimension_y_mm": core["core_dimension_y"] * FACTOR_M_TO_MM,
                    "l_air_gap_mm": user_attrs_l_bot_air_gap * FACTOR_M_TO_MM,
                    "save_fcstd_file": "0"
                }
            )
        else:
            success = True

        if not success:
            logger.warning(f"Transformer ID {transformer_id} custom lower core STEP export failed.")

        # Assemble file name
        target_file_path = os.path.join(output_filepath, f"transformer_{transformer_number}_core_original.step")
        # Create file, if it does not exist
        if not os.path.isfile(target_file_path):
            success = ManufactureDataGeneration._run_freecad(
                freecad_script_file=pq_core_filepath,
                output_file=f"{output_filepath}/transformer_{transformer_number}_core_original.step",
                variables={
                    "core_h_mm": core["core_h"] * FACTOR_M_TO_MM,
                    "core_inner_diameter_mm": core["core_inner_diameter"] * FACTOR_M_TO_MM,
                    "window_h_mm": core["window_h"] * FACTOR_M_TO_MM,
                    "window_w_mm": core["window_w"] * FACTOR_M_TO_MM,
                    "core_dimension_x_mm": core["core_dimension_x"] * FACTOR_M_TO_MM,
                    "core_dimension_y_mm": core["core_dimension_y"] * FACTOR_M_TO_MM,
                    "l_air_gap_mm": 0,
                    "save_fcstd_file": "0"
                }
            )
        else:
            success = True

        if not success:
            logger.warning(f"Transformer ID {transformer_id} original core STEP export failed.")

        # Assemble file name
        target_file_path = os.path.join(output_filepath, f"transformer_{transformer_number}_core_upper.step")
        # Create file, if it does not exist
        if not os.path.isfile(target_file_path):
            success = ManufactureDataGeneration._run_freecad(
                freecad_script_file=pq_core_filepath,
                output_file=target_file_path,
                variables={
                    "core_h_mm": (2 * user_attrs_window_h_top + 2 * top_bottom_yoke_height) * FACTOR_M_TO_MM,
                    "core_inner_diameter_mm": user_attrs_core_inner_diameter * FACTOR_M_TO_MM,
                    "window_h_mm": user_attrs_window_h_top * 2 * FACTOR_M_TO_MM,  # upper core half needs twice the window_h
                    "window_w_mm": user_attrs_window_w * FACTOR_M_TO_MM,
                    "core_dimension_x_mm": core["core_dimension_x"] * FACTOR_M_TO_MM,
                    "core_dimension_y_mm": core["core_dimension_y"] * FACTOR_M_TO_MM,
                    "l_air_gap_mm": user_attrs_l_top_air_gap * FACTOR_M_TO_MM * 2,  # upper core half needs the full air gap, not the reduced one
                    "save_fcstd_file": "0"
                }
            )
        else:
            success = True

        if not success:
            logger.warning(f"Transformer ID {transformer_id} custom upper pq core STEP export failed.")

        # bobbin generation
        bobbin_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "freecad_models/pq_bobbin.py")
        clearance_mm = 0.3

        # Assemble file name
        target_file_path = os.path.join(output_filepath, f"transformer_{transformer_number}_bobbin_upper.step")
        # Create file, if it does not exist
        if not os.path.isfile(target_file_path):
            # generate transformer upper bobbin
            success = ManufactureDataGeneration._run_freecad(
                freecad_script_file=bobbin_filepath,
                output_file=target_file_path,

                variables={
                    "window_h_mm": user_attrs_window_h_top * FACTOR_M_TO_MM,
                    "window_w_mm": user_attrs_window_w * FACTOR_M_TO_MM,
                    "core_inner_diameter_mm": user_attrs_core_inner_diameter * FACTOR_M_TO_MM,

                    # Bobbin dimensions
                    "flange_thickness_inner_mm": transformer_insulations.iso_window_top_core_left * FACTOR_M_TO_MM - clearance_mm,
                    "flange_thickness_top_mm": transformer_insulations.iso_window_top_core_top * FACTOR_M_TO_MM - clearance_mm,
                    "flange_thickness_bot_mm": transformer_insulations.iso_window_top_core_bot * FACTOR_M_TO_MM - clearance_mm,

                    "clearance": clearance_mm,
                    "inner_edge_radius": 0.6,
                    "outer_edge_radius": 0.6,
                    "enable_wire_slots": True,
                    "wire_slots_position": "both",
                    "wire_slot_width": 4.0
                }
            )
        else:
            success = True

        if not success:
            logger.warning(f"Transformer ID {transformer_id} custom upper bobbin STEP export failed.")

        # Assemble file name
        target_file_path = os.path.join(output_filepath, f"transformer_{transformer_number}_bobbin_lower.step")
        # Create file, if it does not exist
        if not os.path.isfile(target_file_path):
            # generate transformer lower bobbin
            success = ManufactureDataGeneration._run_freecad(
                freecad_script_file=bobbin_filepath,
                output_file=target_file_path,

                variables={
                    "window_h_mm": params_window_h_bot * FACTOR_M_TO_MM,
                    "window_w_mm": user_attrs_window_w * FACTOR_M_TO_MM,
                    "core_inner_diameter_mm": user_attrs_core_inner_diameter * FACTOR_M_TO_MM,

                    # Bobbin dimensions
                    "flange_thickness_inner_mm": transformer_insulations.iso_window_bot_core_left * FACTOR_M_TO_MM - clearance_mm,
                    "flange_thickness_top_mm": transformer_insulations.iso_window_bot_core_top * FACTOR_M_TO_MM - clearance_mm,
                    "flange_thickness_bot_mm": transformer_insulations.iso_window_bot_core_bot * FACTOR_M_TO_MM - clearance_mm,

                    "clearance": clearance_mm,
                    "inner_edge_radius": 0.6,
                    "outer_edge_radius": 0.6,
                    "enable_wire_slots": True,
                    "wire_slots_position": "both",
                    "wire_slot_width": 4.0
                }
            )
        else:
            success = True

        if not success:
            logger.warning(f"Transformer ID {transformer_id} custom lower bobbin STEP export failed.")

    @staticmethod
    def _generate_heat_sink_data(heat_sink_id: int, df_heat_sink: pd.DataFrame, output_filepath: str) -> None:
        """
        Generate heat sink manufacturing data.

        :param heat_sink_id: heat sink ID
        :type heat_sink_id: int
        :param df_heat_sink: heat sink dataframe
        :type df_heat_sink: pd.DataFrame
        :param output_filepath: output filepath
        :type output_filepath: str
        """
        params_fan = df_heat_sink.loc[df_heat_sink['number'] == heat_sink_id, 'params_fan'].values[0]
        params_height_c = df_heat_sink.loc[df_heat_sink['number'] == heat_sink_id, 'params_height_c'].values[0]
        params_height_d = df_heat_sink.loc[df_heat_sink['number'] == heat_sink_id, 'params_height_d'].values[0]
        params_length_l = df_heat_sink.loc[df_heat_sink['number'] == heat_sink_id, 'params_length_l'].values[0]
        params_number_cooling_channels_n = df_heat_sink.loc[df_heat_sink['number'] == heat_sink_id, 'params_number_cooling_channels_n'].values[0]
        params_thickness_fin_t = df_heat_sink.loc[df_heat_sink['number'] == heat_sink_id, 'params_thickness_fin_t'].values[0]
        params_width_b = df_heat_sink.loc[df_heat_sink['number'] == heat_sink_id, 'params_width_b'].values[0]

        heat_sink_data = (
            f"{params_fan=}\n"
            f"{params_height_c=}\n"
            f"{params_height_d=}\n"
            f"{params_length_l=}\n"
            f"{params_number_cooling_channels_n=}\n"
            f"{params_thickness_fin_t=}\n"
            f"{params_width_b=}\n"
        )
        with open(f"{output_filepath}/heat_sink_data.txt", "w", encoding="utf-8") as f:
            f.write(heat_sink_data)

        heat_sink_model_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "freecad_models/heat_sink.py")

        # Assemble file name
        target_file_path = os.path.join(output_filepath, "heat_sink.step")
        # Create file, if it does not exist
        if not os.path.isfile(target_file_path):
            success = ManufactureDataGeneration._run_freecad(
                freecad_script_file=heat_sink_model_filepath,
                output_file=target_file_path,
                variables={
                    "height_c_mm": params_height_c * FACTOR_M_TO_MM,
                    "height_d_mm": params_height_d * FACTOR_M_TO_MM,
                    "length_l_mm": params_length_l * FACTOR_M_TO_MM,
                    "number_cooling_channels_n_mm": params_number_cooling_channels_n,
                    "thickness_fin_t_mm": params_thickness_fin_t * FACTOR_M_TO_MM,
                    "width_b_mm": params_width_b * FACTOR_M_TO_MM
                }
            )
        else:
            success = True

        if not success:
            logger.warning(f"Heat sink ID {heat_sink_id} STEP export failed.")

    @staticmethod
    def generate_manufacturing_data(debug: tc.Debug,
                                    circuit_configuration: CircuitOptimizationBase,
                                    inductor_configuration_list: list[InductorConfiguration],
                                    transformer_configuration_list: list[TransformerConfiguration],
                                    capacitor_configuration_list: list[CapacitorConfiguration],
                                    heat_sink_configuration: StudyData,
                                    summary_data: StudyData, data_generation_data: StudyData) -> None:
        """
        Generate data for all components to enable the manufacturing process.

        :param debug: Debug configuration
        :type debug: tc.Debug
        :param circuit_configuration: circuit configuration
        :type circuit_configuration:
        :param inductor_configuration_list: inductor configuration list
        :type inductor_configuration_list: list[InductorConfiguration]
        :param transformer_configuration_list: transformer configuration list
        :type transformer_configuration_list: list[TransformerConfiguration]
        :param capacitor_configuration_list: capacitor configuration list
        :type capacitor_configuration_list: list[CapacitorConfiguration]
        :param heat_sink_configuration: heat sink configuration
        :type heat_sink_configuration: StudyData
        :param summary_data: summary data
        :type summary_data: StudyData
        :param data_generation_data: data generation data
        :type data_generation_data: StudyData
        """
        # read summary parameters
        summary_filepath = os.path.join(summary_data.optimization_directory, DF_SUMMARY_FINAL_FILTERED_MEAN_LOSS)

        df_summary = pd.read_csv(summary_filepath)

        if debug.general.is_debug:
            # reduce dataset to the given number from the debug configuration
            df_summary = df_summary.iloc[:debug.data_generation.number_combinations_max]

        for combination_id in df_summary["combination_id"]:
            logger.info(f"Generate manufacturing data for {combination_id=}")

            circuit_id, capacitor_id_list, inductor_id_list, transformer_id_list, heat_sink_id = ManufactureDataGeneration._read_summary_parameters(
                combination_id, df_summary)

            # read circuit file
            circuit_filepath = os.path.join(circuit_configuration.circuit_study_data.optimization_directory,
                                            f"{circuit_configuration.circuit_study_data.study_name}.csv")

            output_filepath = os.path.join(data_generation_data.optimization_directory, str(combination_id))
            if not os.path.exists(output_filepath):
                os.makedirs(output_filepath)
            waveform_filepath = os.path.join(output_filepath, DATA_GENERATION_WAVEFORM_FOLDER)
            if not os.path.exists(waveform_filepath):
                os.makedirs(waveform_filepath)

            df_circuit = pd.read_csv(circuit_filepath)
            ManufactureDataGeneration._generate_circuit_data(circuit_id, df_circuit, output_filepath)

            # generate operating point table for microcontroller programming
            circuit_id_filepath = os.path.join(circuit_configuration.circuit_study_data.optimization_directory, FILTERED_RESULTS_PATH, f"{circuit_id}.pkl")
            circuit_configuration.generate_operating_point_table(circuit_id_filepath, output_filepath)
            circuit_configuration.plot_compare_waveforms(circuit_id_filepath, waveform_filepath)

            # generate plots of operating points
            result_dto_path = os.path.join(summary_data.optimization_directory, SUMMARY_COMBINATION_FOLDER)

            # Assemble pkl-filename
            combination_id_filepath = os.path.join(result_dto_path, f"{combination_id}.pkl")

            # Get circuit results
            circuit_configuration.plot_single_design_operating_points(combination_id_filepath, output_filepath, combination_id)

            # read capacitor file
            for count, capacitor_id in enumerate(capacitor_id_list):
                capacitor_filepath = os.path.join(capacitor_configuration_list[count].study_data.optimization_directory,
                                                  str(circuit_id), capacitor_configuration_list[count].study_data.study_name, CAPACITOR_RESULTS)
                df_capacitor = pd.read_csv(capacitor_filepath)
                ManufactureDataGeneration._generate_capacitor_data(capacitor_id, df_capacitor, output_filepath, count)

            # read inductor file
            for count, inductor_id in enumerate(inductor_id_list):
                inductor_filepath = os.path.join(inductor_configuration_list[count].study_data.optimization_directory, str(circuit_id),
                                                 inductor_configuration_list[count].study_data.study_name,
                                                 f"{inductor_configuration_list[count].study_data.study_name}.csv")
                df_inductor = pd.read_csv(inductor_filepath)
                inductor_insulations = inductor_configuration_list[count].inductor_toml_data.insulations  # type: ignore
                ManufactureDataGeneration._generate_inductor_data(inductor_id, df_inductor, output_filepath, count, inductor_insulations)

                inductor_figure_filepath = os.path.join(inductor_configuration_list[count].study_data.optimization_directory, str(circuit_id),
                                                        inductor_configuration_list[count].study_data.study_name, CIRCUIT_INDUCTOR_FEM_LOSSES_FOLDER,
                                                        f"{inductor_id}.png")

                if os.path.exists(inductor_figure_filepath):
                    shutil.copy(inductor_figure_filepath, os.path.join(output_filepath, f"inductor_{inductor_id}.png"))

            # read transformer file
            for count, transformer_id in enumerate(transformer_id_list):
                transformer_filepath = os.path.join(transformer_configuration_list[count].study_data.optimization_directory, str(circuit_id),
                                                    transformer_configuration_list[count].study_data.study_name,
                                                    f"{transformer_configuration_list[count].study_data.study_name}.csv")
                df_transformer = pd.read_csv(transformer_filepath)

                transformer_insulations = transformer_configuration_list[count].transformer_toml_data.insulation  # type: ignore
                if not isinstance(transformer_insulations, tc.TomlTransformerInsulation):
                    raise TypeError(f"Transformer insulations missing (Type {type(transformer_insulations)}, "
                                    f"but not type pcdt.toml_checker.TomlTransformerInsulation.")
                ManufactureDataGeneration._generate_transformer_data(transformer_id, df_transformer, output_filepath, count, transformer_insulations)

                transformer_figure_filepath = os.path.join(transformer_configuration_list[count].study_data.optimization_directory, str(circuit_id),
                                                           transformer_configuration_list[count].study_data.study_name, CIRCUIT_TRANSFORMER_FEM_LOSSES_FOLDER,
                                                           f"{transformer_id}.png")

                if os.path.exists(transformer_figure_filepath):
                    shutil.copy(transformer_figure_filepath, os.path.join(output_filepath, f"transformer_{transformer_id}.png"))

            # read heat sink file
            heat_sink_filepath = os.path.join(heat_sink_configuration.optimization_directory, f"{heat_sink_configuration.study_name}.csv")
            df_heat_sink = pd.read_csv(heat_sink_filepath)
            ManufactureDataGeneration._generate_heat_sink_data(heat_sink_id, df_heat_sink, output_filepath)
