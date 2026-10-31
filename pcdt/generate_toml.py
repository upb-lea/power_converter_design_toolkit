"""Generate default toml files and folder structure."""
# python libraries
import os
import shutil
import logging
import select
import sys

# 3rd party libraries
from jinja2 import Environment
import json

# own libraries
from pcdt.circuit_enums import TopologyEnum
from pcdt.constant_path import LOGGING_CONF_TPL
from pcdt.topology import dab, sbc
from pcdt.circuit_enums import TemplateType
from pcdt import toml_checker as tc
from pcdt.topology.circuit_optimization_base import CircuitOptimizationBase
from pcdt.constant_path import (PCDT_ROOT, PROG_FLOW_FILE_NAME, CONFIG_FILE_FOLDER, PROG_FLOW_CONF_TPL, CAPACITOR_CONF_TPL,
                                INDUCTOR_CONF_TPL, TRANSFORMER_CONF_TPL, HEAT_SINK_DISC_FOLDER, HEAT_SINK_CONF_TPL,
                                SUMMARY_CONF_TPL, TEMPLATE_FOLDER, TOMPL_TEMPLATE_FOLDER_NAME, GENERAL_CONF_FILE_NAME,
                                CIRCUIT_CONF_FILE_NAME, CAPACITOR_XCONF_FILE_NAME, INDUCTOR_XCONF_FILE_NAME,
                                TRANSFORMER_XCONF_FILE_NAME, HEAT_SINK_CONF_FILE_NAME, SUMMARY_CONF_FILE_NAME, PLACE_HOLDER,
                                FILEPATH_CONFIG_JSON, CIRCUIT_FOLDER, CAPACITOR_FOLDER, INDUCTOR_FOLDER, TRANSFORMER_FOLDER,
                                HEAT_SINK_FOLDER, PRE_SUMMARY_FOLDER, SUMMARY_FOLDER, DATA_GENERATION_FOLDER)

logger = logging.getLogger(__name__)

class GenerateDefaultConfig:
    """Class to generate configuration files if missing."""

    @staticmethod
    def generate_default_files(file_path: str) -> None:
        """
        Check a default set of all configuration files.

        :param file_path: file name and path of program flow file
        :type  file_path: str
        """
        # Variable declaration and initialization
        # Configuration value dictionary
        conf_dict: dict = {}
        topology_selection: TopologyEnum
        default_project_name: str = "_default_project"
        default_config_name: str = "default_config"
        folder_dict: dict = {}
        topology: CircuitOptimizationBase

        # Select the topology for the default configuration
        topology_selection = GenerateDefaultConfig._select_default_topology()

        # Add default configuration name
        conf_dict["default_project_name"] = str(topology_selection.value) + default_project_name
        conf_dict["default_config_name"] = default_config_name
        if topology_selection is TopologyEnum.dab:
            conf_dict["topology"] = "dab"
            conf_dict["circuit_numbers_of_trials"] = "500"
            conf_dict["capacitor_calculation_modes"] = "[\"new\",\"new\"]"
            conf_dict["inductor_numbers_of_trials"] = "500"
            conf_dict["inductor_calculation_modes"] = "[\"new\"]"
            conf_dict["transformer_numbers_of_trials"] = "500"
            conf_dict["transformer_calculation_modes"] = "[\"new\"]"
            conf_dict["heat_sink_numbers_of_trials"] = "500"
            # Allocate and initialize topology configuration for dual active bridge
            topology = dab.DabCircuitOptimization()
        elif topology_selection is TopologyEnum.sbc:
            conf_dict["topology"] = "sbc"
            conf_dict["circuit_numbers_of_trials"] = "500"
            conf_dict["capacitor_calculation_modes"] = "[\"new\",\"new\"]"
            conf_dict["inductor_numbers_of_trials"] = "500"
            conf_dict["inductor_calculation_modes"] = "[\"new\"]"
            conf_dict["transformer_numbers_of_trials"] = "500"
            conf_dict["transformer_calculation_modes"] = "[]"
            conf_dict["heat_sink_numbers_of_trials"] = "500"
            # Allocate and initialize topology configuration for dual active bridge
            topology = sbc.SbcCircuitOptimization()
        else:
            raise ValueError("Serious programming error in topology selection in file"
                             f"{__name__}.‑\nPlease write an issue!")

        template_path = str(os.path.join(PCDT_ROOT, TEMPLATE_FOLDER, TOMPL_TEMPLATE_FOLDER_NAME))
        project_directory = os.path.join(os.path.dirname(file_path), conf_dict["default_project_name"])

        # Generate program flow configuration file
        progflow_file_path = os.path.join(os.path.dirname(file_path), PROG_FLOW_FILE_NAME)
        template_file_path = os.path.join(template_path, PROG_FLOW_CONF_TPL)

        # Check if template file is available
        if os.path.isfile(template_file_path):
            with open(template_file_path, "r", encoding="utf-8") as f:
                template_source = f.read()
        else:
            logger.error(f"Data corruption: Missing template file {template_file_path}")
            return

        GenerateDefaultConfig._generate_default_toml(progflow_file_path, template_source, conf_dict)

        # Generate configuration files
        config_path = str(os.path.join(project_directory, CONFIG_FILE_FOLDER, default_config_name))
        GenerateDefaultConfig.generate_component_toml_files(topology, config_path)

        # Set up the dictionary
        folder_dict["circuit_path"] = os.path.join(project_directory, CIRCUIT_FOLDER)
        folder_dict["capacitor_path"] = os.path.join(project_directory, CAPACITOR_FOLDER)
        folder_dict["inductor_path"] = os.path.join(project_directory, INDUCTOR_FOLDER)
        folder_dict["transformer_path"] = os.path.join(project_directory, TRANSFORMER_FOLDER)
        folder_dict["heat_sink_path"] = os.path.join(project_directory, HEAT_SINK_FOLDER)
        folder_dict["heat_sink_disc_path"] = os.path.join(project_directory, HEAT_SINK_DISC_FOLDER)
        folder_dict["pre_summary_path"] = os.path.join(project_directory, PRE_SUMMARY_FOLDER)
        folder_dict["summary_path"] = os.path.join(project_directory, SUMMARY_FOLDER)
        folder_dict["data_generation_path"] = os.path.join(project_directory, DATA_GENERATION_FOLDER)

        # Generate project folder structure
        GenerateDefaultConfig._set_up_folder_structure(folder_dict)

    @staticmethod
    def generate_component_toml_files(topology: CircuitOptimizationBase, config_directory: str) -> None:
        """
        Check for missing toml default files and generate them in case of missing.

        :param topology: Topology instance will be overloaded by the derived class
        :type  topology: CircuitOptimizationBase
        :param config_directory: Storage location of the configuration file
        :type  config_directory: str
        """
        # Variable declaration and initialization
        # Path for templates
        template_path = str(os.path.join(PCDT_ROOT, TEMPLATE_FOLDER, TOMPL_TEMPLATE_FOLDER_NAME))

        # Read the number of required components
        number_of_capacitors = topology.get_number_of_required_capacitors()
        number_of_inductors = topology.get_number_of_required_inductors()
        number_of_transformers = topology.get_number_of_required_transformers()

        # Generate the file
        config_file_path = str(os.path.join(config_directory, GENERAL_CONF_FILE_NAME))
        GenerateDefaultConfig.generate_general_toml(config_file_path, topology)

        # Generate the file
        config_file_path = str(os.path.join(config_directory, CIRCUIT_CONF_FILE_NAME))
        GenerateDefaultConfig.generate_circuit_toml(config_file_path, topology)

        # Check for missing capacitor configuration files
        # Prepare template
        template_file_path = os.path.join(template_path, CAPACITOR_CONF_TPL)
        # Read template
        is_valid, template_source = GenerateDefaultConfig._read_template(template_file_path)
        if is_valid:
            for index in range(number_of_capacitors):
                config_file_path = str(os.path.join(config_directory, CAPACITOR_XCONF_FILE_NAME.replace(PLACE_HOLDER, str(index))))
                GenerateDefaultConfig.generate_default_capacitor_toml(config_file_path, index, topology.get_topology_type())

        # Generate the file
        # Prepare template
        template_file_path = os.path.join(template_path, INDUCTOR_CONF_TPL)
        # Read template
        is_valid, template_source = GenerateDefaultConfig._read_template(template_file_path)
        if is_valid:
            for index in range(number_of_inductors):
                config_file_path = str(os.path.join(config_directory, INDUCTOR_XCONF_FILE_NAME.replace(PLACE_HOLDER, str(index))))
                GenerateDefaultConfig.generate_default_inductor_toml(config_file_path, index, topology.get_topology_type())

        # Generate the file
        # Prepare template
        template_file_path = os.path.join(template_path, TRANSFORMER_CONF_TPL)
        # Read template
        is_valid, template_source = GenerateDefaultConfig._read_template(template_file_path)
        if is_valid:
            for index in range(number_of_transformers):
                config_file_path = str(os.path.join(config_directory, TRANSFORMER_XCONF_FILE_NAME.replace(PLACE_HOLDER, str(index))))
                GenerateDefaultConfig.generate_default_transformer_toml(config_file_path, index, topology.get_topology_type())

        # Generate the file
        config_file_path = str(os.path.join(config_directory, HEAT_SINK_CONF_FILE_NAME))
        if not os.path.isfile(config_file_path):
            # Generate the file
            GenerateDefaultConfig.generate_default_heat_sink_toml(config_file_path)

        # Generate the file
        config_file_path = str(os.path.join(config_directory, SUMMARY_CONF_FILE_NAME))
        if not os.path.isfile(config_file_path):
            # Generate the file
            GenerateDefaultConfig.generate_default_summary_toml(config_file_path)

    @staticmethod
    def generate_general_toml(general_file_path: str, topology: CircuitOptimizationBase) -> None:
        """
        Generate general toml default file if missing.

        :param general_file_path: Path and filename of general configuration file
        :type  general_file_path: str
        :param topology: Topology instance will be overloaded by the derived class
        :type  topology: CircuitOptimizationBase
        """
        # Read template
        is_valid, template_source = topology.get_template(TemplateType.general)
        if is_valid:
            # Prepare configuration dictionary
            conf_dict: dict = {}
            # Generate the file
            GenerateDefaultConfig._generate_default_toml(general_file_path, template_source, conf_dict)

    @staticmethod
    def generate_circuit_toml(circuit_file_path: str, topology: CircuitOptimizationBase) -> None:
        """
        Generate circuit toml default file if missing.

        :param circuit_file_path: Path and filename of circuit configuration file
        :type  circuit_file_path: str
        :param topology: Topology instance will be overloaded by the derived class
        :type  topology: CircuitOptimizationBase
        """
        # Read template
        is_valid, template_source = topology.get_template(TemplateType.circuit)
        if is_valid:
            # Prepare configuration dictionary
            conf_dict: dict = {}
            # Generate the file
            GenerateDefaultConfig._generate_default_toml(circuit_file_path, template_source, conf_dict)

    @staticmethod
    def generate_default_capacitor_toml(capacitor_file_path: str, index: int, topology_type: TopologyEnum) -> None:
        """
        Generate capacitor toml default file if missing.

        :param capacitor_file_path: Path and filename of capacitor configuration file
        :type  capacitor_file_path: str
        :param index: Component index of capacitor component (Used by configuration dictionary)
        :type  index: int
        :param topology_type: Type of the topology
        :type  topology_type: TopologyEnum
        """
        # Variable declaration and initialization
        # Path for templates
        template_path = str(os.path.join(PCDT_ROOT, TEMPLATE_FOLDER, TOMPL_TEMPLATE_FOLDER_NAME))

        # Prepare template
        template_file_path = os.path.join(template_path, CAPACITOR_CONF_TPL)
        # Read template
        is_valid, template_source = GenerateDefaultConfig._read_template(template_file_path)
        if is_valid:
            # Prepare configuration dictionary
            conf_dict: dict = {}
            # Generate the file
            GenerateDefaultConfig._generate_default_toml(capacitor_file_path, template_source, conf_dict)

    @staticmethod
    def generate_default_inductor_toml(inductor_file_path: str, index: int, topology_type: TopologyEnum) -> None:
        """
        Generate inductor toml default file if missing.

        :param inductor_file_path: Path and filename of inductor configuration file
        :type  inductor_file_path: str
        :param index: Component index of capacitor component (Used by configuration dictionary)
        :type  index: int
        :param topology_type: Type of the topology
        :type  topology_type: TopologyEnum
        """
        # Variable declaration and initialization
        conf_dict: dict = {}
        section_dict: dict = {}
        # Path for templates
        template_path = str(os.path.join(PCDT_ROOT, TEMPLATE_FOLDER, TOMPL_TEMPLATE_FOLDER_NAME))

        # Prepare template
        template_file_path = os.path.join(template_path, INDUCTOR_CONF_TPL)
        # Read template
        is_valid, template_source = GenerateDefaultConfig._read_template(template_file_path)
        if is_valid:
            if topology_type is TopologyEnum.dab:
                section_dict["permeability_datasource"] = "LEA_MTB"
                section_dict["permittivity_datasource"] = "LEA_MTB"
                conf_dict["material_data_sources"] = section_dict
            elif topology_type is TopologyEnum.sbc:
                section_dict["permeability_datasource"] = "MagNet"
                section_dict["permittivity_datasource"] = "LEA_MTB"
                conf_dict["material_data_sources"] = section_dict
            else:
                raise ValueError("Serious programming error in topology selection in file"
                                 f"{__name__}.‑\nPlease write an issue!")
            # Generate the file
            GenerateDefaultConfig._generate_default_toml(inductor_file_path, template_source, conf_dict)

    @staticmethod
    def generate_default_transformer_toml(transformer_file_path: str, index: int, topology_type: TopologyEnum) -> None:
        """
        Generate transformer toml default file if missing.

        :param transformer_file_path: Path and filename of transformer configuration file
        :type  transformer_file_path: str
        :param index: Component index of capacitor component (Used by configuration dictionary)
        :type  index: int
        :param topology_type: Type of the topology
        :type  topology_type: TopologyEnum
        """
        # Variable declaration and initialization
        # Path for templates
        template_path = str(os.path.join(PCDT_ROOT, TEMPLATE_FOLDER, TOMPL_TEMPLATE_FOLDER_NAME))

        # Prepare template
        template_file_path = os.path.join(template_path, TRANSFORMER_CONF_TPL)
        # Read template
        is_valid, template_source = GenerateDefaultConfig._read_template(template_file_path)
        if is_valid:
            # Prepare configuration dictionary
            conf_dict: dict = {}
            # Generate the file
            GenerateDefaultConfig._generate_default_toml(transformer_file_path, template_source, conf_dict)

    @staticmethod
    def generate_default_heat_sink_toml(heat_sink_file_path: str) -> None:
        """
        Generate heat sink toml default file if missing.

        :param heat_sink_file_path: Path and filename of heat sink configuration file
        :type  heat_sink_file_path: str
        """
        # Variable declaration and initialization
        # Path for templates
        template_path = str(os.path.join(PCDT_ROOT, TEMPLATE_FOLDER, TOMPL_TEMPLATE_FOLDER_NAME))

        # Prepare template
        template_file_path = os.path.join(template_path, HEAT_SINK_CONF_TPL)
        # Read template
        is_valid, template_source = GenerateDefaultConfig._read_template(template_file_path)
        if is_valid:
            # Prepare configuration dictionary
            conf_dict: dict = {}
            # Generate the file
            GenerateDefaultConfig._generate_default_toml(heat_sink_file_path, template_source, conf_dict)

    @staticmethod
    def generate_default_summary_toml(summary_file_path: str) -> None:
        """
        Generate summary toml default file if missing.

        :param summary_file_path: Path and filename of summary configuration file
        :type  summary_file_path: str
        """
        # Variable declaration and initialization
        # Path for templates
        template_path = str(os.path.join(PCDT_ROOT, TEMPLATE_FOLDER, TOMPL_TEMPLATE_FOLDER_NAME))

        # Prepare template
        template_file_path = os.path.join(template_path, SUMMARY_CONF_TPL)
        # Read template
        is_valid, template_source = GenerateDefaultConfig._read_template(template_file_path)
        if is_valid:
            # Prepare configuration dictionary
            conf_dict: dict = {}
            # Generate the file
            GenerateDefaultConfig._generate_default_toml(summary_file_path, template_source, conf_dict)

    @staticmethod
    def generate_logging_config(logging_file_path: str) -> None:
        """
        Generate the default logging configuration file.

        :param logging_file_path: path and file name of logging configuration file
        :type  logging_file_path: str
        """
        # Variable declaration and initialization
        conf_dict: dict = {}

        # file path of logging template
        template_file_path = str(os.path.join(PCDT_ROOT, TEMPLATE_FOLDER, TOMPL_TEMPLATE_FOLDER_NAME, LOGGING_CONF_TPL))
        # Read template
        is_valid, template_source = GenerateDefaultConfig._read_template(template_file_path)
        if is_valid:
            # Generate the file
            GenerateDefaultConfig._generate_default_toml(logging_file_path, template_source, conf_dict)

    @staticmethod
    def _backup_toml_file(file_path: str) -> None:
        """
        Generate the default toml file.

        :param target_file_path: filename and path of the file to generate
        :type  target_file_path: str
        :param act_template_source: Template source data
        :type  act_template_source: str
        :param config_data: Configuration dictionary with data to fill in template
        :type  config_data: dict
        """
        # Check if backup is necessary
        if os.path.isfile(file_path):
            file_path_backup = file_path + ".backup"
            shutil.move(file_path, file_path_backup)
            logger.info(f"A backup is created: {file_path_backup}.\n")

    @staticmethod
    def _generate_default_toml(target_file_path: str, act_template_source: str, config_data: dict) -> None:
        """
        Generate the default toml file.

        :param target_file_path: filename and path of the file to generate
        :type  target_file_path: str
        :param act_template_source: Template source data
        :type  act_template_source: str
        :param config_data: Configuration dictionary with data to fill in template
        :type  config_data: dict
        """
        # Backup old file, if it exists
        GenerateDefaultConfig._backup_toml_file(target_file_path)
        # Prepare rendering
        env = Environment(autoescape=False, keep_trailing_newline=True)
        template = env.from_string(act_template_source)
        configuration_content = template.render(cdata=config_data)

        target_path = os.path.dirname(target_file_path)
        # Check if folder exists, if not create it
        os.makedirs(target_path, exist_ok=True)

        # Create or overwrite file
        with open(target_file_path, 'w') as output:
            output.write(configuration_content)

    @staticmethod
    def _read_template(act_template_file_path: str) -> tuple[bool, str]:
        """
        Read template file, if it is available.

        :param timeout: Timeout to select the topology in seconds
        :type timeout: int
        :return: validity indicator and template source data
        :rtype: tuple[bool, str]
        """
        # Variable declaration
        is_valid: bool = True
        template_source: str = ""

        # Check if template file is available
        if os.path.isfile(act_template_file_path):
            with open(act_template_file_path, "r", encoding="utf-8") as f:
                template_source = f.read()
        else:
            is_valid = False
            logger.error(f"Data corruption: Missing template file {act_template_file_path}")

        return is_valid, template_source

    @staticmethod
    def _select_default_topology(timeout: float = 5) -> TopologyEnum:
        """
        Select the topology for default program flow configuration file.

        If no configuration files exists the user can select the topology, for which the
        default configuration will be generated. The selection opportunity has got a time limit.

        :param timeout: Timeout to select the topology in seconds
        :type timeout: int
        """
        # Variable declaration
        configurations: dict = {
            "1": "Dual active bridge",
            "2": "Synchronous buck converter",
        }
        # Default topology selection and timeout limit calculation
        selection = TopologyEnum.dab

        # Request to user
        print("Dual active bridge -> press 1")
        print("Synchronous buck converter -> press 2")
        print(f"Confirm selection with 'Enter' ({timeout:g}s).")

        while True:
            # Request selection
            print("Selection [1–2]: ", end="", flush=True)
            result_list, _, _ = select.select([sys.stdin], [], [], timeout)

            # Check for timeout
            if not result_list:
                print("\nSelection time is finished. Topology 'Dual active bridge' will be used.")
                break
            # Read the input
            line = sys.stdin.readline()
            # Check for close terminal
            if not line:
                break

            # Extract the input
            choice = line.strip()
            if choice == "1":
                selection = TopologyEnum.dab
                break
            elif choice == "2":
                selection = TopologyEnum.sbc
                break
            else:
                # Notify user and reset timeout
                print("Invalid input! Please select 1 or 2.\n(Timeout is set again to {timeout:g}s!")

        return selection

    @staticmethod
    def check_generate_folder_structure(act_toml_prog_flow: tc.FlowControl) -> None:
        """
        Check the folder structure and generate all folders, which are missing.

        :param act_toml_prog_flow: Flow control toml file data
        :type act_toml_prog_flow: tc.FlowControl
        """
        # Variable declaration
        folder_dict: dict = {}

        # Set up the dictionary
        project_directory = act_toml_prog_flow.general.project_directory
        folder_dict["circuit_path"] = os.path.join(project_directory, act_toml_prog_flow.circuit.subdirectory)
        folder_dict["capacitor_path"] = os.path.join(project_directory, act_toml_prog_flow.capacitor.subdirectory)
        folder_dict["inductor_path"] = os.path.join(project_directory, act_toml_prog_flow.inductor.subdirectory)
        folder_dict["transformer_path"] = os.path.join(project_directory, act_toml_prog_flow.transformer.subdirectory)
        folder_dict["heat_sink_path"] = os.path.join(project_directory, act_toml_prog_flow.heat_sink.subdirectory)
        folder_dict["heat_sink_disc_path"] = os.path.join(project_directory, HEAT_SINK_DISC_FOLDER)
        folder_dict["pre_summary_path"] = os.path.join(project_directory, act_toml_prog_flow.pre_summary.subdirectory)
        folder_dict["summary_path"] = os.path.join(project_directory, act_toml_prog_flow.summary.subdirectory)
        folder_dict["data_generation_path"] = os.path.join(project_directory, act_toml_prog_flow.data_generation.subdirectory)

        # Setup folder structure
        GenerateDefaultConfig._set_up_folder_structure(folder_dict)

        json_filepath = os.path.join(project_directory, FILEPATH_CONFIG_JSON)

        with open(json_filepath, 'w', encoding='utf8') as json_file:
            json.dump(folder_dict, json_file, ensure_ascii=False, indent=4)

    @staticmethod
    def _set_up_folder_structure(act_folder_dict: dict) -> None:
        """
        Set up the folder structure for the subprojects.

        :param toml_prog_flow: Flow control toml file
        :type toml_prog_flow: tc.FlowControl
        """
        # Generate folder structure according dictionary
        for _, value in act_folder_dict.items():
            os.makedirs(value, exist_ok=True)
