"""Generate default toml files."""
# python libraries
import os

def generate_missing_toml_files(working_directory: str) -> None:
    """
    Check for missing toml default files. Generate them, if missing.

    :param working_directory: working directory
    :type working_directory: str
    """
    # check for missing component configuration files
    if not os.path.isfile(os.path.join(working_directory, "DabInductorConf.toml")):
        generate_default_inductor_toml(working_directory)
    if not os.path.isfile(os.path.join(working_directory, "DabTransformerConf.toml")):
        generate_default_transformer_toml(working_directory)
    if not os.path.isfile(os.path.join(working_directory, "DabHeatSinkConf.toml")):
        generate_default_heat_sink_toml(working_directory)
    if not os.path.isfile(os.path.join(working_directory, "DabSummaryConf.toml")):
        generate_default_summary_toml(working_directory)
    if not os.path.isfile(os.path.join(working_directory, "Misc.toml")):
        generate_default_misc_toml(working_directory)

def generate_default_flow_control_toml(working_directory: str) -> None:
    """
    Generate the default progFlow.toml file.

    :param working_directory: working directory
    :type working_directory: str
    """
    toml_data = '''
    [default data] # After update this configuration file according your project delete this line to validate it
    # Path configuration
    [general]
        project_directory = "2026-01-09_example"
        topology = "dab"
    
    [breakpoints]
        # possible values: no/pause/stop
        circuit_pareto = "no"  # After Electrical pareto front calculation
        circuit_filtered = "no"  # After Electrical filtered result calculation
        capacitor = "no"   # After capacitor selection result calculation
        inductor = "no"    # After inductor pareto front calculations of for all correspondent electrical points
        transformer = "no" # After transformer pareto front calculations of for all correspondent electrical points
        heat_sink = "no"    # After heat sink pareto front calculation
        pre_summary = "no"    # After pre-summary calculation
        summary = "no"    # After summary calculation
    
    [conditional_breakpoints] # conditional breakpoints in case of bad definition array (only for experts and currently not implemented)
        circuit = 1000        # Number of trials with ZVS less than 70%
        inductor = 1000       # Number of trials which exceed a limit value?
        transformer = 1000      # Number of trials which exceed a limit value?
        heat_sink = 1000         # Number of trials which exceed a limit value?
    
    [circuit]
        number_of_trials = 100
        calculation_mode = "new" # (new,continue,skip)
        subdirectory = "01_circuit"
    
    [capacitor]
        # number of the entries corresponds to the number of required capacitors of the  topology
        calculation_modes = ["new", "new"] # (new,skip)
        subdirectory = "02_capacitor"
    
    [inductor]
        numbers_of_trials = [200]
        calculation_modes = ["new"] # (new,continue,skip)
        subdirectory = "03_inductor"
    
    [transformer]
        numbers_of_trials = [400]
        calculation_modes = ["new"] # (new,continue,skip)
        subdirectory = "04_transformer"
    
    [heat_sink]
        number_of_trials = 500
        calculation_mode = "new" # (new,continue,skip,disc)
        subdirectory = "05_heat_sink"
    
    [pre_summary]
        calculation_mode = "new" # (new,skip)
        subdirectory = "06_pre_summary"
    
    [summary]
        calculation_mode = "new" # (new,skip)
        subdirectory = "07_summary"
        
    [data_generation]
        calculation_mode_visualization = "new" # (new,skip_purge)
        calculation_mode_manufacture = "new"   # (new,continue,skip_purge)        
        subdirectory = "08_data_generation"
    
    [configuration_data_files]
        # General configuration file followed by circuit configuration file
        topology_files =  ["DabGeneralConf.toml","DabCircuitConf.toml"]
        # Number of capacitor configuration files corresponds to the required number of capacitors of the topology
        capacitor_configuration_files = ["DabCapacitor1Conf.toml", "DabCapacitor2Conf.toml"]
        # Number of capacitor configuration files corresponds to the required number of capacitors of the topology
        inductor_configuration_files = ["DabInductorConf.toml"]
        # Number of capacitor configuration files corresponds to the required number of capacitors of the topology
        transformer_configuration_files = ["DabTransformerConf.toml"]
        heat_sink_configuration_file = "DabHeatSinkConf.toml"
        summary_configuration_file = "DabSummaryConf.toml"
        #    # General configuration file followed by circuit configuration file
        #    topology_files =  ["SbcGeneralConf.toml","SbcCircuitConf.toml"]
        #    # Number of capacitor configuration files corresponds to the required number of capacitors of the topology
        #    capacitor_configuration_files = [""]
        #    # Number of capacitor configuration files corresponds to the required number of capacitors of the topology
        #    inductor_configuration_files = ["SbcInductorConf.toml"]
        #    # Number of capacitor configuration files corresponds to the required number of capacitors of the topology
        #    transformer_configuration_files = [""]
        #    heat_sink_configuration_file = "SbcHeatSinkConf.toml"
    '''
    with open(f"{working_directory}/progFlow.toml", 'w') as output:
        output.write(toml_data)

def generate_default_misc_toml(file_path: str) -> None:
    """
    Generate the default misc toml file.

    :param file_path: filename including absolute path
    :type file_path: str
    """
    toml_data = '''
    [default_data] # After update this configuration file according your project delete this line to validate it
    min_efficiency_percent=80
    control_board_volume=10e-6
    control_board_loss=1
    '''
    with open(file_path, 'w') as output:
        output.write(toml_data)

def generate_default_capacitor_toml(file_path: str) -> None:
    """
    Generate the default capacitor selection toml file.

    :param file_path: filename including absolute path
    :type file_path: str
    """
    toml_data = '''
    [default_data] # After update this configuration file according your project delete this line to validate it
    maximum_peak_to_peak_voltage_ripple = 1
    temperature_ambient = 90
    voltage_safety_margin_percentage = 10
    maximum_number_series_capacitors = 2
    lifetime_h = 30_000
    '''
    with open(file_path, 'w') as output:
        output.write(toml_data)

def generate_default_inductor_toml(file_path: str) -> None:
    """
    Generate the default inductor  configuration toml file.

    :param file_path: working directory
    :type file_path: str
    """
    toml_data = '''
    [default_data] # After update this configuration file according your project delete this line to validate it
    [design_space]
        core_name_list=["PQ 50/50", "PQ 50/40", "PQ 40/40", "PQ 40/30", "PQ 35/35", "PQ 32/30", "PQ 32/20", "PQ 26/25", "PQ 26/20", "PQ 20/20", "PQ 20/16"]
        material_name_list=["N49"]
        litz_wire_name_list=["1.5x105x0.1", "1.4x200x0.071", "1.1x60x0.1"]
        core_inner_diameter_min_max_list=[]
        window_h_min_max_list=[]
        window_w_min_max_list=[]
    
    [boundary_conditions]
        temperature=100
    
    [material_data_sources]
        permeability_datasource="LEA_MTB"
        permittivity_datasource="LEA_MTB"
    
    [insulations]
        primary_to_primary=0.2e-3
        core_bot=1e-3
        core_top=1e-3
        core_right=1e-3
        core_left=1e-3

    [thermal_data]
        # [tim_thickness, tim_conductivity]
        thermal_cooling = [1e-3,12.0]
        
    [filter_distance]
        factor_dc_losses_min_max_list=[0.01, 100]
    '''
    with open(file_path, 'w') as output:
        output.write(toml_data)

def generate_default_transformer_toml(file_path: str) -> None:
    """
    Generate the default transformer configuration toml file.

    :param file_path: filename including absolute path
    :type file_path: str
    """
    toml_data = '''
    [default_data] # After update this configuration file according your project delete this line to validate it
    [design_space]
        material_name_list=['N49']
        core_name_list=["PQ 40/40", "PQ 40/30", "PQ 35/35", "PQ 32/30", "PQ 32/20", "PQ 26/25", "PQ 26/20", "PQ 20/20", "PQ 20/16"]
        core_inner_diameter_min_max_list=[15e-3, 30e-3]
        window_w_min_max_list=[10e-3, 40e-3]
        window_h_bot_min_max_list=[10e-3, 50e-3]
        primary_litz_wire_list=['1.1x60x0.1']
        secondary_litz_wire_list=['1.35x200x0.071', '1.1x60x0.1']
        # sweep parameters: geometry and materials
        n_p_top_min_max_list=[1, 30]
        n_p_bot_min_max_list=[10, 80]
    
    [boundary_conditions]
        # maximum limitation for transformer total height and core volume
        max_transformer_total_height=60e-3
        max_core_volume=0.007853982  # 50e-3 ** 2 * 3.141593
        temperature=100
    
    [insulation]
        # insulation for top core window
        iso_window_top_core_top=1.3e-3
        iso_window_top_core_bot=1.3e-3
        iso_window_top_core_left=1.3e-3
        iso_window_top_core_right=1.3e-3
        # insulation for bottom core window
        iso_window_bot_core_top=1.3e-3
        iso_window_bot_core_bot=1.3e-3
        iso_window_bot_core_left=1.3e-3
        iso_window_bot_core_right=1.3e-3
        # winding-to-winding insulation
        iso_primary_to_primary=0.2e-3
        iso_secondary_to_secondary=0.2e-3
        iso_primary_to_secondary=0.2e-3
    
    [material_data_sources]
        permeability_datasource="LEA_MTB"
        permittivity_datasource="LEA_MTB"
        
    [thermal_data]
        # [tim_thickness, tim_conductivity]
        thermal_cooling = [1e-3,12.0]
            
    [settings]
        fft_filter_value_factor=0.01
        mesh_accuracy=0.8
    
    [filter_distance]
        factor_dc_losses_min_max_list=[0.01, 100]
    '''
    with open(file_path, 'w') as output:
        output.write(toml_data)

def generate_default_heat_sink_toml(file_path: str) -> None:
    """
    Generate the default heat sink configuration toml file.

    :param file_path: filename including absolute path
    :type file_path: str
    """
    toml_data = '''
    [default_data] # After update this configuration file according your project delete this line to validate it
    [design_space]
        height_c_min_max_list=[0.02, 0.08]
        width_b_min_max_list=[0.02, 0.08]
        length_l_min_max_list=[0.08, 0.20]
        height_d_min_max_list=[0.001, 0.003]
        number_fins_n_min_max_list=[5, 20]
        thickness_fin_t_min_max_list=[1e-3, 5e-3]
    
    [boundary_conditions]
        t_ambient=40
        t_hs_max=95
        area_min=0.001
    
    [settings]
        number_directions=3
        factor_pcb_area_copper_coin = 1.42
        factor_bottom_area_copper_coin = 0.39
        # W/(m*K)
        thermal_conductivity_copper = 136

    '''
    with open(file_path, 'w') as output:
        output.write(toml_data)

def generate_default_summary_toml(file_path: str) -> None:
    """
    Generate the default summary configuration toml file.

    :param file_path: filename including absolute path
    :type file_path: str
    """
    toml_data = '''
    [default_data] # After update this configuration file according your project delete this line to validate it
    [pre_summary]
        filter_distance_losses = [0.01, 100]
        filter_distance_efficiency = [0.03, 0.9]

    [summary]
        filter_distance_losses = [0.01, 100]
        filter_distance_efficiency = [0.03, 0.9]
    '''
    with open(file_path, 'w') as output:
        output.write(toml_data)


def generate_logging_config(working_directory: str) -> None:
    """
    Generate the default logging configuration file.

    :param working_directory: working directory
    :type working_directory: str
    """
    logging_data_config = '''[loggers]
keys=root,pcdt,transistordatabase,femmt,hct,uvicorn

[handlers]
keys=console

[formatters]
keys=simple

[logger_root]
level=INFO
handlers=console
qualname=

[logger_pcdt]
level=INFO
handlers=console
qualname=pcdt
propagate=0

[logger_transistordatabase]
level=INFO
handlers=console
qualname=transistordatabase
propagate=0

[logger_femmt]
level=INFO
handlers=console
qualname=femmt
propagate=0

[logger_hct]
level=INFO
handlers=console
qualname=hct
propagate=0

[logger_uvicorn]
level=INFO
handlers=console
qualname=uvicorn
propagate=0

[handler_console]
class=StreamHandler
level=NOTSET
formatter=simple
args=(sys.stdout,)

[formatter_simple]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
datefmt=%Y-%m-%d %H:%M:%S
    '''
    with open(f"{working_directory}/logging.conf", 'w') as output:
        output.write(logging_data_config)
