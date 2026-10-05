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
                                VISUALIZATION_TEMPLATE, CB_START_CSS, CB_DETAIL_CCS, PCDT_ROOT)

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
