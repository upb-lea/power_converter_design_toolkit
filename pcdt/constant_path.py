"""Constant path and file name storage."""

# python libraries
import os

# 3rd party libraries
# own libraries
import pcdt

# root directory
PCDT_ROOT = os.path.dirname(os.path.realpath(pcdt.__file__))

FILEPATH_CONFIG_JSON = "filepath_config.json"

# GECKO simulation
GECKO_PATH = os.path.join(PCDT_ROOT, "topology", "dab", "GeckoCIRCUITS")
GECKO_COMPONENT_MODELS_DIRECTORY = "gecko_component_models"

# folder names
CIRCUIT_CAPACITOR_LOSS_FOLDER = "01_capacitor_results"
CIRCUIT_INDUCTOR_RELUCTANCE_LOSSES_FOLDER = "08_reluctance_inductor_results"
CIRCUIT_INDUCTOR_FEM_LOSSES_FOLDER = "09_fem_inductor_results"
CIRCUIT_TRANSFORMER_RELUCTANCE_LOSSES_FOLDER = "08_reluctance_transformer_results"
CIRCUIT_TRANSFORMER_FEM_LOSSES_FOLDER = "09_fem_transformer_results"
SUMMARY_COMBINATION_FOLDER = "combination_results"
SUMMARY_COMBINATION_PlOTS_FOLDER = "combination_plots"
FILTERED_RESULTS_PATH = "filtered_results"
CIRCUIT_WAVEFORMS_FOLDER = "filtered_results_waveforms"
PARETO_PLOT_PDF_FOLDER = "pareto_plot_pdf"
PARETO_PLOT_PNG_FOLDER = "pareto_plot_png"
PARETO_PLOT_PKL_FOLDER = "pareto_plot_pkl"
HEAT_SINK_DISC_FOLDER = "heat_sink_disc"
DATA_GENERATION_WAVEFORM_FOLDER = "waveforms"
DATA_GENERATION_MANUFACTURE = "manufacture_data"
DATA_GENERATION_VISUALIZATION = "visualization_data"
TEMPLATE_FOLDER = "templates"
HTML_TEMPLATE_FOLDER_NAME = "htmltemplates"
WEBSERVER_TEMPLATE = "web_server.pytmpl"
PYTHON_TEMPLATE_FOLDER_NAME = "pythontemplates"
STYLESHEET_FOLDER_NAME = "StyleSheets"
VISUALIZATION_TEMPLATE = "visualization_template.html"
CANDIDATE_BROWSER_HTML = "candidate_browser.html"
WEB_SERVER_PY = "web_server.py"
TMP_DETAILED_HTML = "browser_detailed_page_template.html"
CB_START_CSS = "browser_start_page.css"
CB_DETAIL_CCS = "browser_detailed_page.css"
# processing status
RELUCTANCE_COMPLETE_FILE = "reluctance_processing_complete.json"
FEM_COMPLETE_FILE = "fem_processing_complete.json"
PROCESSING_COMPLETE_FILE = "processing_complete.json"

# capacitor optimization result file names
CAPACITOR_RESULTS = "results.csv"
CAPACITOR_RESULTS_FILTERED = "results_filtered.csv"

# result data frames filenames
DF_SUMMARY_WITHOUT_HEAT_SINK_WITHOUT_OFFSET = "df_wo_hs_wo_offset.csv"
DF_SUMMARY_WITH_HEAT_SINK_WITHOUT_OFFSET = "df_w_hs_wo_offset.csv"
DF_SUMMARY_FINAL = "df_final.csv"
DF_SUMMARY_FINAL_FILTERED_MEAN_LOSS = "df_final_filtered_mean_loss.csv"
DF_SUMMARY_FINAL_FILTERED_WEIGHTED_EFFICIENCY = "df_final_filtered_weighted_efficiency.csv"
PLOT_CIRCUIT = "circuit"
PLOT_HEAT_SINK = "heat_sink"
PLOT_SUMMARY_WEIGHTED_EFFICIENCY = "summary_weighted_efficiency"
PLOT_SUMMARY_MEAN_LOSS = "summary_mean_loss"

# FEMMT fixed folder names, do not change!
FEMMT_FEM_RESULTS_FOLDER = "02_fem_simulation_results"
