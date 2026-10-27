"""Candidate browser data generator for result visualization."""

# Python libraries
import os
import logging
import csv
import shutil
import copy

# 3rd party libraries
from jinja2 import Environment, FileSystemLoader

# own libraries
from pcdt import CapacitorConfiguration, InductorConfiguration, TransformerConfiguration, StudyData, CircuitOptimizationBase
from pcdt.constant_path import (TEMPLATE_FOLDER, HTML_TEMPLATE_FOLDER_NAME, STYLESHEET_FOLDER_NAME, CANDIDATE_BROWSER_HTML,
                                PYTHON_TEMPLATE_FOLDER_NAME, WEBSERVER_TEMPLATE, VISUALIZATION_TEMPLATE, CB_START_CSS,
                                CB_DETAIL_CCS, PCDT_ROOT, WEB_SERVER_PY, TMP_DETAILED_HTML, CB_ICON, CB_IDLE_PNG)

logger = logging.getLogger(__name__)

class CandidateBrowserGen:
    """Generate visualization data."""

    def __init__(self, study_data: StudyData) -> None:
        """Initialize the member variable with study data.

        :param act_candidate_list: Study data of candidate browser
        :type  act_candidate_list: StudyData
        """
        self.candidate_browser_study_data: StudyData = study_data
        self.html_template_folder: str = os.path.join(PCDT_ROOT, TEMPLATE_FOLDER, HTML_TEMPLATE_FOLDER_NAME)
        self.webserver_template_folder: str = os.path.join(PCDT_ROOT, TEMPLATE_FOLDER, PYTHON_TEMPLATE_FOLDER_NAME)

    @staticmethod
    def _read_brief_candidate_data(path_file: str) -> list[dict]:
        """
        Read design candidate from csv-file.

        :param path_file: path and file name
        :type  path_file: str
        :return: list of candidate with brief parameter
        :rtype: list[dict]
        """
        # Variable declaration
        candidate_list: list[dict] = []

        # Check if file exists
        if os.path.isfile(path_file):
            # Read file content
            with open(path_file, 'r') as f:
                reader = csv.DictReader(f)
                # Get brief data from file
                for row in reader:
                    candidate_list.append({
                        'circuit_id': row['circuit_id'],
                        'combination_id': row['combination_id'],
                        'total_mean_loss': row['total_mean_loss'],
                        'weighted_efficiency': row['weighted_efficiency'],
                        'total_volume': row['total_volume'],
                        '3D_file': "place_holder"
                    })
            logger.info("Start result visualization data generation")
        else:
            # Notify the user about not existing file
            logger.warning(f"File '{path_file}' does not exists!")

        # Return the candidate list
        return candidate_list

    def _generate_html(self, act_candidate_list: list[dict], act_destination_file_path: str) -> None:
        """
        Generate html file.

        :param act_candidate_list: List with design candidate information
        :type  act_candidate_list: list[dict]
        :param act_destination_path: storage location of the html-file
        :type  act_destination_path: str
        """
        # Variable declaration and initialization
        # Style sheet list
        stylesheet_list: list[str] = [CB_START_CSS, CB_DETAIL_CCS, CB_ICON, CB_IDLE_PNG]
        # Destination path for style sheets
        style_sheet_path: str
        # Style sheet source path
        style_sheet_source_path: str = os.path.join(self.html_template_folder, STYLESHEET_FOLDER_NAME)

        # Template are load by data (Special approach of Jinja2 to access the template)
        env = Environment(loader=FileSystemLoader(self.html_template_folder))
        template = env.get_template(VISUALIZATION_TEMPLATE)
        html = template.render(candidate_list=act_candidate_list)

        # Get folder name
        destination_path = os.path.dirname(act_destination_file_path)
        style_sheet_path = os.path.join(destination_path, STYLESHEET_FOLDER_NAME)

        # Check if folder exists, if not create it
        os.makedirs(style_sheet_path, exist_ok=True)

        # Create or overwrite file
        with open(act_destination_file_path, 'w') as f:
            f.write(html)
        # Copy style sheets to define the style
        for stylesheet in stylesheet_list:
            shutil.copy(f"{style_sheet_source_path}/{stylesheet}", style_sheet_path)

    def _generate_webbrowser(self, act_conf_names: dict, act_destination_file_path: str) -> None:
        """
        Generate html file.

        :param act_candidate_list: List with design candidate information
        :type  act_candidate_list: list[dict]
        :param act_destination_path: storage location of the html-file
        :type  act_destination_path: str
        """
        # Variable declaration and initialization
        server_path_information: dict = {}
        # Destination path for template
        template_path: str
        # Template list
        template_list: list[str] = [TMP_DETAILED_HTML]

        # Set path values
        server_path_information["work_path"] = act_conf_names["work_path"]
        server_path_information["summary_data_csv"] = act_conf_names["summary"]
        server_path_information["circuit_data_csv"] = os.path.join(act_conf_names["circuit_folder"], act_conf_names["circuit"] + ".csv")
        server_path_information["capacitor_base_path"] = os.path.join(act_conf_names["capacitor"])
        server_path_information["inductor_base_path"] = os.path.join(act_conf_names["inductor"])
        server_path_information["transformer_base_path"] = os.path.join(act_conf_names["transformer"])
        for index, component in enumerate(act_conf_names["capacitor_component"]):
            server_path_information["capacitor_csv" + str(index)] = os.path.join(component + str(index), "results.csv")
        for index, component in enumerate(act_conf_names["inductor_component"]):
            server_path_information["inductor_csv" + str(index)] = os.path.join(component,
                                                                                component + ".csv")
        for index, component in enumerate(act_conf_names["transformer_component"]):
            server_path_information["transformer_csv" + str(index)] = os.path.join(component,
                                                                                   component + ".csv")
        server_path_information["heat_sink_csv"] = os.path.join(act_conf_names["heat_sink"], act_conf_names["heat_sink"] + ".csv")
        server_path_information["visualization_base_path"] = act_conf_names["data_generation"]

        # Template are load by data (Special approach of Jinja2 to access the template)
        env = Environment(loader=FileSystemLoader(self.webserver_template_folder))
        web_template = env.get_template(WEBSERVER_TEMPLATE)
        webserver_app = web_template.render(path_info=server_path_information)

        # Get folder name
        destination_path = os.path.dirname(act_destination_file_path)
        template_path = os.path.join(destination_path, HTML_TEMPLATE_FOLDER_NAME)

        # Check if folder exists, if not create it
        os.makedirs(template_path, exist_ok=True)

        # Create or overwrite file
        with open(act_destination_file_path, 'w') as f:
            f.write(webserver_app)

        # Copy template to target location
        for template in template_list:
            shutil.copy(f"{self.html_template_folder}/{template}", template_path)

    def generate_html_visualization(self,
                                    circuit_configuration: CircuitOptimizationBase,
                                    inductor_configuration_list: list[InductorConfiguration],
                                    transformer_configuration_list: list[TransformerConfiguration],
                                    capacitor_configuration_list: list[CapacitorConfiguration],
                                    heat_sink_configuration: StudyData,
                                    summary_data: StudyData) -> None:
        """
        Generate data for all components to enable the manufacturing process.

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
        """
        # Variable declaration
        # List of design candidates information
        candidate_info_list: list[dict]
        # Configuration path dictionary
        conf_names: dict[str, str | list] = {}
        # List for sub path
        conf_sub_list: list[str] = []

        # File path for csv data
        csv_filepath: str = os.path.join(summary_data.optimization_directory, "df_final.csv")

        # Read csv-file
        candidate_info_list = CandidateBrowserGen._read_brief_candidate_data(csv_filepath)
        # Check if file has content
        if candidate_info_list:
            # Assemble target name
            html_filepath: str = os.path.join(self.candidate_browser_study_data.optimization_directory, CANDIDATE_BROWSER_HTML)
            # Generate html-file
            self._generate_html(candidate_info_list, html_filepath)

            # Generate web browser
            conf_names["work_path"] = circuit_configuration.project_directory
            conf_names["circuit_folder"] = circuit_configuration.circuit_study_data.optimization_directory
            conf_names["circuit"] = circuit_configuration.circuit_study_data.study_name
            # Check if capacitors are part of the topology
            if capacitor_configuration_list:
                # Set base path for capacitor
                conf_names["capacitor"] = capacitor_configuration_list[0].study_data.optimization_directory
                # Add capacitors
                for cap_config in capacitor_configuration_list:
                    conf_sub_list.append(cap_config.study_data.study_name)
                # Add list to dictionary and clear list
                conf_names["capacitor_component"] = copy.deepcopy(conf_sub_list)
                conf_sub_list = []
            else:
                conf_names["capacitor"] = ""
            # Check if inductors are part of the topology
            if inductor_configuration_list:
                # Set base path for inductor
                conf_names["inductor"] = inductor_configuration_list[0].study_data.optimization_directory
                # Add inductors
                for ind_config in inductor_configuration_list:
                    conf_sub_list.append(ind_config.study_data.study_name)
                # Add list to dictionary and clear list
                conf_names["inductor_component"] = copy.deepcopy(conf_sub_list)
                conf_sub_list = []
            else:
                conf_names["inductor"] = ""
            # Check if inductors are part of the topology
            if transformer_configuration_list:
                # Set base path for transformer
                conf_names["transformer"] = transformer_configuration_list[0].study_data.optimization_directory
                # Add transformers
                for tr_config in transformer_configuration_list:
                    conf_sub_list.append(tr_config.study_data.study_name)
                # Add list to dictionary
                conf_names["transformer_component"] = conf_sub_list
            else:
                conf_names["transformer"] = ""
            # Add heat sink
            conf_names["heat_sink"] = heat_sink_configuration.optimization_directory
            # Add summary and data generation
            conf_names["summary"] = csv_filepath
            conf_names["data_generation"] = self.candidate_browser_study_data.optimization_directory

            # Assemble target name
            webserver_filepath: str = os.path.join(self.candidate_browser_study_data.optimization_directory, WEB_SERVER_PY)
            # Generate html-file
            self._generate_webbrowser(conf_names, webserver_filepath)
