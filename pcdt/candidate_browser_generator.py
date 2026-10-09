"""Candidate browser data generator for result visualization."""

# Python libraries
import os
import logging
import csv
import shutil

# 3rd party libraries
from jinja2 import Environment, FileSystemLoader

# own libraries
import pcdt.toml_checker as tc
from pcdt import CapacitorConfiguration, InductorConfiguration, TransformerConfiguration, StudyData, CircuitOptimizationBase
from pcdt.constant_path import (TEMPLATE_FOLDER, HTML_TEMPLATE_FOLDER_NAME, STYLESHEET_FOLDER_NAME, CANDIDATE_BROWSER_HTML,
                                PYTHON_TEMPLATE_FOLDER_NAME, WEBSERVER_TEMPLATE, VISUALIZATION_TEMPLATE, CB_START_CSS,
                                CB_DETAIL_CCS, PCDT_ROOT, WEB_SERVER_PY, TMP_DETAILED_HTML)

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
        stylesheet_list: list[str] = [CB_START_CSS, CB_DETAIL_CCS]
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
        server_path_information["summary_data_csv"] = os.path.join("07_summary", act_conf_names["circuit"], "df_final.csv")
        server_path_information["circuit_data_csv"] = os.path.join("01_circuit", act_conf_names["circuit"], act_conf_names["circuit"] + ".csv")
        server_path_information["capacitor_base_path"] = os.path.join("02_capacitor", act_conf_names["circuit"])
        server_path_information["inductor_base_path"] = os.path.join("03_inductor", act_conf_names["circuit"])
        server_path_information["transformer_base_path"] = os.path.join("04_transformer", act_conf_names["circuit"])
        server_path_information["capacitor0_csv"] = os.path.join(act_conf_names["capacitor0"], "results.csv")
        server_path_information["capacitor1_csv"] = os.path.join(act_conf_names["capacitor1"], "results.csv")
        server_path_information["inductor_csv"] = os.path.join(act_conf_names["inductor0"], act_conf_names["inductor0"] + ".csv")
        server_path_information["transformer_csv"] = os.path.join(act_conf_names["transformer0"], act_conf_names["transformer0"] + ".csv")
        server_path_information["heat_sink_csv"] = os.path.join("05_heat_sink", act_conf_names["circuit"], act_conf_names["heat_sink"] + ".csv")
        server_path_information["visualization_base_path"] = os.path.join("08_data_generation", act_conf_names["circuit"], "visualization_data")

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


    def generate_html_visualization(self, debug: tc.Debug,
                                    circuit_configuration: CircuitOptimizationBase,
                                    inductor_configuration_list: list[InductorConfiguration],
                                    transformer_configuration_list: list[TransformerConfiguration],
                                    capacitor_configuration_list: list[CapacitorConfiguration],
                                    heat_sink_configuration: StudyData,
                                    summary_data: StudyData) -> None:
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
        """
        # Variable declaration
        # List of design candidates information
        candidate_info_list: list[dict]
        # Configuration path dictionary
        conf_names: dict = {}

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
            conf_names["circuit"] = circuit_configuration.circuit_study_data.study_name
            # Add capacitors
            for index, cap_config in enumerate(capacitor_configuration_list):
                key = "capacitor" + str(index)
                conf_names[key] = cap_config.study_data.study_name
            # Add inductors
            for index, ind_config in enumerate(inductor_configuration_list):
                key = "inductor" + str(index)
                conf_names[key] = ind_config.study_data.study_name
            # Add transformers
            for index, tr_config in enumerate(transformer_configuration_list):
                key = "transformer" + str(index)
                conf_names[key] = tr_config.study_data.study_name
            # Add heat sink
            conf_names["heat_sink"] = heat_sink_configuration.study_name

            # Assemble target name
            webserver_filepath: str = os.path.join(self.candidate_browser_study_data.optimization_directory, WEB_SERVER_PY)
            # Generate html-file
            self._generate_webbrowser(conf_names, webserver_filepath)
