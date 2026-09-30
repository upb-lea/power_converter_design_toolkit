"""Server control class to visualize current metrics."""

# python libraries
import multiprocessing
from multiprocessing import Queue
import threading
import time
import os
from os.path import abspath
from typing import Any
from enum import Enum

# 3rd party libraries
import uvicorn
from fastapi import FastAPI, Request, Form, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.templating import _TemplateResponse
from starlette.middleware.sessions import SessionMiddleware

# own libraries
import pcdt.server_ctl_dtos as server_ctl_dtos
from pcdt.constant_path import TEMPLATE_FOLDER, HTML_TEMPLATE_FOLDER_NAME, STYLESHEET_FOLDER_NAME, PCDT_ROOT

# Structure classes
# Structure class of request command
class RequestCmd(Enum):
    """Enum of possible commands."""

    page_main = 0     # Request statistical data of main page
    page_detail = 1   # Request detailed statistical data
    pareto_front = 2  # Request pareto front
    continue_opt = 3  # Request to continue (if breakpoint reached)

# Structure class of request command
class ParetoFrontSource(Enum):
    """Enum of Pareto front types."""

    pareto_circuit = 0      # Request Pareto front of the circuit
    pareto_inductor = 1     # Request Pareto front of the inductor
    pareto_transformer = 2  # Request Pareto front of the transformer
    pareto_heat_sink = 3    # Request Pareto front of the heat sink
    pareto_summary = 4      # Request Pareto front of the summary?

# Structure class of request
class ServerRequestData:
    """Request command structure."""

    # Request command
    request_cmd: RequestCmd
    # Pareto front source
    pareto_source: ParetoFrontSource
    # Index of the circuit configuration
    c_configuration_index: int
    # Index of the configuration item (inductor, transformer, heat sink or summary)
    item_configuration_index: int
    # Index of circuit filtered point
    c_filtered_point_index: int

class PcdtServer:
    """Server to visualize the actual progress and calculated Pareto-fronts."""

    # Method declaration
    server_thread: threading.Thread

    # Allocate FastAPI-Server
    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key="your_secret_key")

    # Set template directory
    _template_directory: str = os.path.join(PCDT_ROOT, TEMPLATE_FOLDER, HTML_TEMPLATE_FOLDER_NAME)
    templates = Jinja2Templates(directory=_template_directory)

    # Currently dummy password, later to secure in specific way
    users = {"Andi": "hallo"}

    # Progress status icon definition to display
    icon = [f"{STYLESHEET_FOLDER_NAME}/OptIdle.png", f"{STYLESHEET_FOLDER_NAME}/OptInProgress.png",
            f"{STYLESHEET_FOLDER_NAME}/OptDone.png", f"{STYLESHEET_FOLDER_NAME}/OptSkipped.png"]

    # Allocate multi processing variables
    server_request_queue: Queue
    server_response_queue: Queue

    # ASA Create switch for release
    # Release:
    # _ssl_cert = os.getenv("SSL_CERT_PATH", "ssl/cert.pem")  # Default for development
    # _ssl_key = os.getenv("SSL_KEY_PATH", "ssl/key.pem")
    # Development
    _ssl_cert = abspath("ssl/cert.pem")  # Default for development
    _ssl_key = abspath("ssl/key.pem")
    _optuna_path = abspath(f"{_template_directory}/OptunaOrg.html")

    status_message = "Wait for button press!"  # Initial text
    # Server object
    server_object: uvicorn.Server
    # Shared memory variable
    req_stop = multiprocessing.Value('i', 0)
    stop_flag = multiprocessing.Value('i', 0)
    # Server process
    _server_process = None
    # program exit flag
    _prog_exit_flag = False
    # Server supervision
    _server_supervision = None
    # Selected configuration index
    _c_config_index = 0
    # Selected filtered point index
    _c_filtered_point_index = 0

    # Breakpoint flag
    _breakpoint_flag = False

    # Debug
    break_status: int = 0

    # Mount of style sheet path
    app.mount("/StyleSheets", StaticFiles(directory=os.path.join(_template_directory, "StyleSheets")), name="Stylesheets")

    @staticmethod
    def get_current_user(request: Request) -> Any:
        """Provide the user of the current session.

        User of the current session in case of valid login

        :param request: Request information of the client request
        :type  request: Request
        :return: User name, if client is logged in, otherwise None
        :rtype: Any
        """
        return request.session.get("user")

    @staticmethod
    def start_pcdt_server(act_server_request_queue: Queue, act_server_response_queue: Queue, program_exit_flag: bool) -> None:
        """Start the server to control and supervise simulation.

        :param act_server_request_queue: Queue object to request data from main process
        :type  act_server_request_queue: Queue
        :param act_server_response_queue: Queue object to responds to server process
        :type  act_server_response_queue: Queue
        :param program_exit_flag: Flag, which indicates to terminate the program on request
        :type  program_exit_flag: boolean
        """
        PcdtServer._prog_exit_flag = program_exit_flag

        # Start the server process
        PcdtServer._server_process = multiprocessing.Process(target=PcdtServer._run_server,
                                                             args=(act_server_request_queue, act_server_response_queue,))
        PcdtServer._server_process.start()

        # Check if server process supervision is to start due to program exit requested by server
        if PcdtServer._prog_exit_flag:
            # Create thread for the server supervision and start it
            PcdtServer._server_supervision = threading.Thread(target=PcdtServer._supervise_server_stop, daemon=True)
            PcdtServer._server_supervision.start()

    @staticmethod
    def stop_pcdt_server() -> None:
        """Stop the server for the control and supervision of the simulation."""
        # Set program exit flag to false because program will be exit by themselves
        PcdtServer._prog_exit_flag = False

        # Request server to stop
        PcdtServer.req_stop.value = 1
        # Stop the server  process (if started)
        if PcdtServer._server_process is not None:
            PcdtServer._server_process.join(5)
        # Stop server supervision if started
        if PcdtServer._server_supervision is not None:
            PcdtServer._server_supervision.join(5)

    @staticmethod
    def _supervise_server_stop() -> None:
        """Stop the supervision of the server in main process."""
        # Supervise if the server is stopped by user request
        while True:
            # Reduce CPU-supervise load by toggle each second
            time.sleep(1)
            # Check if server is stopped
            if PcdtServer.stop_flag.value == 1:
                # Requested to stop if the program needs to stop too
                if PcdtServer._prog_exit_flag:
                    # Check if the program needs to stop too
                    print("Program stop is requested")
                    # Hard kill of process
                    os._exit(0)
                break

    @staticmethod
    def _run_server(act_server_request_queue: Queue, act_server_response_queue: Queue) -> None:
        """Start the FastAPI-Server.

        :param act_server_request_queue: Queue object to request data from main process
        :type  act_server_request_queue: Queue
        :param act_server_response_queue: Queue object to responds to server process
        :type  act_server_response_queue: Queue
        """
        # Overtake the shared memory variable
        PcdtServer.server_request_queue = act_server_request_queue
        PcdtServer.server_response_queue = act_server_response_queue
        # Start the server (blocking call)
        config = uvicorn.Config(PcdtServer.app, host="127.0.0.1", port=8008,
                                ssl_keyfile=PcdtServer._ssl_key, ssl_certfile=PcdtServer._ssl_cert, log_config=None)

        PcdtServer.server_object = uvicorn.Server(config)
        # Create thread for the server and start it
        PcdtServer.server_thread = threading.Thread(target=PcdtServer.pcdt_server_thread, daemon=True)
        PcdtServer.server_thread.start()
        # Supervise if the server is stopped by main
        while True:
            # Reduce CPU-supervise load by toggle each second
            time.sleep(1)
            # Check if server is requested to stop
            if PcdtServer.req_stop.value == 1:
                break

        # Stop the server
        PcdtServer.server_object.should_exit = True
        # Wait for thread stop
        PcdtServer.server_thread.join()
        # Set server stop flag to 0
        PcdtServer.stop_flag.value = 1

    @staticmethod
    def pcdt_server_thread() -> None:
        """Start the FastAPI-server."""
        # Start the server in a blocking call
        PcdtServer.server_object.run()

    @staticmethod
    def get_table_data(component_data_list: list[server_ctl_dtos.ConfigurationDataEntryDto]) -> list[dict]:
        """Fill the table data to display progress based on the configuration progress data.

        :param component_data_list: List of configuration data for progress reporting
        :type  component_data_list: list[server_ctl_dtos.ConfigurationDataEntryDto]
        :return: List of dict with the formatted entries
        :rtype:  list[dict]
        """
        # Variable declaration
        # Return value
        table_data: list[dict] = []
        # Index variable
        index = 0

        # Loop over the configurations
        for entry in component_data_list:
            table_data.append({"conf_name": entry.configuration_name, "nb_trails": entry.number_of_trials,
                               "nb_filt_pts": entry.progress_data.number_of_filtered_points,
                               "process_time": PcdtServer.get_format_time(entry.progress_data.run_time),
                               "image_link": PcdtServer.icon[entry.progress_data.progress_status.value],
                               "status": entry.progress_data.progress_status.name,
                               "index": index})
            # Increment index
            index = index + 1
        return table_data

    @staticmethod
    def get_magnetic_table_data(magnetic_data_list: list[server_ctl_dtos.MagneticDataEntryDto]) -> list[dict]:
        """Fill the table data for display magnetic progress data of one configuration with filtered point name.

        :param magnetic_data_list: Configuration data of magnetic configuration for progress reporting (inductor or transformer)
        :type  magnetic_data_list: list[server_ctl_dtos.MagneticDataEntryDto]
        :return: List of formatted entries for progress reporting
        :rtype:  list[dict]
        """
        # Variable declaration
        # Return value
        magnetic_table_data_list: list[dict] = []

        # Loop over the configurations
        for entry in magnetic_data_list:
            # Enter the table data
            magnetic_table_data_list.append({"conf_name": entry.magnetic_configuration_name,
                                             "number_performed_calculations": entry.number_performed_calculations,
                                             "number_calculations": entry.number_calculations,
                                             "progress_time": PcdtServer.get_format_time(entry.progress_data.run_time),
                                             "image_link": PcdtServer.icon[entry.progress_data.progress_status.value],
                                             "status": entry.progress_data.progress_status.name, "index": 0})

        return magnetic_table_data_list

    @staticmethod
    def get_circuit_table_data(circuit_data: server_ctl_dtos.CircuitConfigurationDataDto) -> dict:
        """Fill the table data for display circuit progress data of one configuration with filtered point name.

        :param circuit_data: Configuration data of circuit configuration for progress reporting
        :type  circuit_data: server_ctl_dtos.CircuitConfigurationDataDto
        :return: Formatted entries of circuit configuration data for progress reporting
        :rtype:  dict
        """
        # Variable declaration
        # Enter the table data
        circuit_table_data: dict = {"conf_name": circuit_data.configuration_name,
                                    "nb_trails": circuit_data.number_of_trials,
                                    "c_filtered_points_name_list": circuit_data.filtered_points_name_list,
                                    "process_time": PcdtServer.get_format_time(circuit_data.progress_data.run_time),
                                    "image_link": PcdtServer.icon[circuit_data.progress_data.progress_status.value],
                                    "status": circuit_data.progress_data.progress_status.name, "index": 0}

        return circuit_table_data

    @staticmethod
    def get_heat_sink_table_data(act_heat_sink_data_list: list[server_ctl_dtos.ConfigurationDataEntryDto]) -> list[dict]:
        """Fill the table data to display progress based on the configuration progress data.

        :param act_heat_sink_data_list: List of configuration data for progress reporting
        :type  act_heat_sink_data_list: list[server_ctl_dtos.ConfigurationDataEntryDto]
        :return: List of dict with the formatted entries
        :rtype:  list[dict]
        """
        # Variable declaration
        # Return value
        heat_sink_table_data_list: list[dict[str, Any]] = []

        # Index variable
        entry_index = 0

        # Loop over the configurations
        for entry in act_heat_sink_data_list:
            # Enter the table data
            heat_sink_table_data_list.append({"conf_name": entry.configuration_name, "nb_trails": entry.number_of_trials,
                                              "process_time": PcdtServer.get_format_time(entry.progress_data.run_time),
                                              "image_link": PcdtServer.icon[entry.progress_data.progress_status.value],
                                              "status": entry.progress_data.progress_status.name, "index": entry_index})
            # Increment index
            entry_index = entry_index + 1

        return heat_sink_table_data_list

    @staticmethod
    def get_summary_table_data(act_summary_data_list: list[server_ctl_dtos.SummaryDataEntryDto]) -> list[dict]:
        """Fill the table data to display summary progress based on the configuration progress data.

        :param act_summary_data_list: List of configuration data for progress reporting
        :type  act_summary_data_list: list[server_ctl_dtos.SummaryDataEntryDto]
        :return: List of dict with the formatted entries
        :rtype:  list[dict]
        """
        # Variable declaration
        # Return value
        summary_table_data_list: list[dict[str, Any]] = []

        # Index variable
        entry_index = 0

        # Loop over the configurations
        for entry in act_summary_data_list:
            # Enter the table data
            summary_table_data_list.append({"conf_name": entry.configuration_name, "process_time": PcdtServer.get_format_time(entry.progress_data.run_time),
                                            "nb_of_combinations": entry.number_of_combinations,
                                            "image_link": PcdtServer.icon[entry.progress_data.progress_status.value],
                                            "status": entry.progress_data.progress_status.name, "index": entry_index})
            # Increment index
            entry_index = entry_index + 1

        return summary_table_data_list

    @staticmethod
    def load_optuna_html_file(filepath: str) -> str:
        """Load the optuna file to string.

        !!!Later to replace
        :param filepath: File name inclusive path
        :type  filepath: str
        :return: file data as string
        :rtype:  str
        """
        with open(filepath, "r", encoding="utf-8") as file:
            return file.read()

    @staticmethod
    def _calculate_key_parameter(button_index: int = -1, table_index: int = -1, filtered_point_index: int = -1,
                                 page_index: int = -1) -> tuple[ParetoFrontSource, int]:
        """Create the information string based on the input parameters.

        With the input parameter the information string is generated.
        :param button_index: Index of the button, which is pressed
        :type  button_index: int
        :param table_index: Index of the table, which contains the button
        :type  table_index: in
        :param filtered_point_index: Index of actual filter point index
        :type  filtered_point_index: int
        :param page_index: Actual html-page
        :type  page_index: int
        :return: Pareto front source information and configuration index
        :rtype: ParetoFrontSource, int
        """
        # Variable declaration
        info_string = ""
        # Assign default value to pareto_front_source
        pareto_front_source = ParetoFrontSource.pareto_circuit

        # Check pageID
        # page_index corresponds to main_page1.html
        if page_index == 0:
            if table_index == 1:
                pareto_front_source = ParetoFrontSource.pareto_circuit
            elif table_index == 4:
                pareto_front_source = ParetoFrontSource.pareto_heat_sink
            elif table_index == 5:
                pareto_front_source = ParetoFrontSource.pareto_summary
        # page_index corresponds to main_page2.html
        elif page_index == 1:
            if table_index == 1:
                pareto_front_source = ParetoFrontSource.pareto_circuit
            elif table_index == 2:
                pareto_front_source = ParetoFrontSource.pareto_inductor
            elif table_index == 3:
                pareto_front_source = ParetoFrontSource.pareto_transformer
            elif table_index == 4:
                pareto_front_source = ParetoFrontSource.pareto_heat_sink
            elif table_index == 5:
                pareto_front_source = ParetoFrontSource.pareto_summary

        # Button index corresponds to configuration index
        item_configuration_index = button_index

        return pareto_front_source, item_configuration_index

    @staticmethod
    def get_format_time(time_value: float) -> str:
        """Create the information string based on the input parameters.

        The displayed time value is displayed in human understandable way.
        Depending on thresholds the time is displayed in different way.
        With the input parameter the information string is generated.
        :param time_value: time value in second
        :type  time_value: float
        :return: formatted time value to match the display area
        :rtype:  str
        """
        # 0-6000s
        if time_value < 6000:
            format_time = "{:.0f}s".format(time_value)
        # 10min-100min
        elif time_value < 60000:
            minutes = int(time_value // 60)
            seconds = int(time_value % 60)
            format_time = f"{minutes}min{seconds}s"
        # 1h40min- 99h59min
        elif time_value < 360000:
            minutes = int(time_value // 60)
            hours = int(minutes // 60)
            minutes = int(minutes % 60)
            format_time = f"{hours}h{minutes}min"
        # >=100h
        else:
            hours = int(time_value // 3600)
            format_time = f"{hours}h"

        return format_time

    # -- Responses on client requests --------------------------------------------------------------

    @staticmethod
    @app.get("/", response_class=HTMLResponse, response_model=None)
    @app.get("/html_homepage1", response_class=HTMLResponse, response_model=None)
    async def main_page1(request: Request, action: str = "", user: str = Depends(get_current_user),
                         button_index: int | None = None, table_index: int | None = None) -> _TemplateResponse | HTMLResponse:
        """Provide the html-information based on client request to html_homepage1.

        :param request: Request information of the client request
        :type  request: Request
        :param action: Information about the requested action (Keyword driven)
        :type  action: str
        :param user: User, in case of valid user
        :type  user: str
        :param button_index: Index of selected button, if a button was pressed
        :type  button_index: int
        :param table_index: Index of selected table in case of a button press
        :type  table_index: int
        :return: html-page
        :rtype:  _TemplateResponse | HTMLResponse
        """
        if action == "logout":
            request.session.clear()
            user = ""
            PcdtServer.status_message = "User is logged off"
        elif action == "pareto_circuit":
            # Display Pareto front
            return PcdtServer.get_pareto_front(request, button_index, table_index, "/html_homepage1")
        elif action == "details" and table_index == 1:
            # Check if button_index is valid
            if button_index is not None:
                PcdtServer._c_config_index = button_index
            else:
                PcdtServer._c_config_index = 0
            return await PcdtServer.main_page2(request, "", 0, user)
        elif action == "control_sheet":
            # Check if user is authorized
            if user is not None:
                if PcdtServer.break_status == 1:
                    PcdtServer.break_status = 0
                else:
                    PcdtServer.break_status = 1
                # Display the control page
                break_status = PcdtServer.break_status
                return PcdtServer.templates.TemplateResponse(request=request, name="control_page.html",
                                                             context={"url_back": "/html_homepage1", "break_status": break_status})

        # Init request for main process
        request_data = ServerRequestData()
        request_data.request_cmd = RequestCmd.page_main
        # Later to set the circuit configuration index in case of multiple circuit configurations
        request_data.c_configuration_index = 0

        # Request data from main process
        PcdtServer.server_request_queue.put(request_data)
        # Wait for response
        data: server_ctl_dtos.QueueMainData = PcdtServer.server_response_queue.get()

        # Add content circuit config
        # Create list (in future it is a list of configurations)
        circuit_conf_list: list[server_ctl_dtos.ConfigurationDataEntryDto] = data.circuit_list
        table_data_circuit = PcdtServer.get_table_data(circuit_conf_list)

        table_main_data_inductor = PcdtServer.get_magnetic_table_data(data.inductor_main_list)
        table_main_data_transformer = PcdtServer.get_magnetic_table_data(data.transformer_main_list)

        # Add content heat sink config
        table_data_heat_sink = PcdtServer.get_heat_sink_table_data(data.heat_sink_list)
        # table_data_heat_sink = PcdtServer.get_table_data(data.heat_sink_list)

        # Add content summary
        table_data_summary = PcdtServer.get_summary_table_data(data.summary_list)

        # Add breakpoint notification text and evaluate breakpoint status
        breakpoint_message = data.break_point_notification
        if len(breakpoint_message) > 0:
            PcdtServer._breakpoint_flag = True
        else:
            PcdtServer._breakpoint_flag = False

        return PcdtServer.templates.TemplateResponse(request=request, name="main_page1.html",
                                                     context={"c_table_data": table_data_circuit,
                                                              "i_table_main_data": table_main_data_inductor,
                                                              "t_table_main_data": table_main_data_transformer,
                                                              "h_table_data": table_data_heat_sink,
                                                              "s_table_data": table_data_summary,
                                                              "total_process_time": PcdtServer.get_format_time(data.total_process_time),
                                                              "text_message": PcdtServer.status_message,
                                                              "break_pt_text": breakpoint_message,
                                                              "user": user})

    @staticmethod
    @app.get("/html_homepage2", response_class=HTMLResponse, response_model=None)
    async def main_page2(request: Request, action: str = "", c_selected_filtered_point_index: int = -1,
                         user: str | None = Depends(get_current_user),
                         button_index: int | None = None, table_index: int | None = None) -> _TemplateResponse | HTMLResponse:
        """Provide the html-information based on client request to html_homepage2.

        :param request: Request information of the client request
        :type  request: Request
        :param action: Information about the requested action (Keyword driven)
        :type  action: str
        :param c_selected_filtered_point_index: Index of the selected circuit filtered point
        :type  c_selected_filtered_point_index: int
        :param user: User, in case of valid user
        :type  user: str
        :param button_index: Index of selected button, if a button was pressed
        :type  button_index: int
        :param table_index: Index of selected table in case of a button press
        :type  table_index: int
        :return: html-page
        :rtype:  _TemplateResponse | HTMLResponse
        """
        # Check if an action is requested
        # User request: Logout
        if action == "logout":
            request.session.clear()
            user = None
            PcdtServer.status_message = "User is logged out"
        # User request: Button press to display Pareto-front
        elif action == "pareto_circuit":
            # Display the Pareto-front
            return PcdtServer.get_pareto_front(request, button_index, table_index, "/html_homepage2")
        # User request: Button press to change to control sheet
        elif action == "control_sheet":
            # Check if user is authorized
            if user is not None:
                # Set the break status
                break_status = PcdtServer._breakpoint_flag
                return PcdtServer.templates.TemplateResponse(request=request, name="control_page.html",
                                                             context={"url_back": "/html_homepage2", "break_status": break_status})

        # Init request for main process
        request_data = ServerRequestData()
        request_data.request_cmd = RequestCmd.page_detail
        # Later to set the circuit configuration index in case of multiple circuit configurations
        request_data.c_configuration_index = 0
        # Check selected filtered point index
        if not c_selected_filtered_point_index == -1:
            # Save the selected filtered point index
            PcdtServer._c_filtered_point_index = c_selected_filtered_point_index
        else:
            c_selected_filtered_point_index = PcdtServer._c_filtered_point_index

        request_data.c_filtered_point_index = c_selected_filtered_point_index

        # Request data from main process
        PcdtServer.server_request_queue.put(request_data)
        # Wait for response
        data: server_ctl_dtos.QueueDetailData = PcdtServer.server_response_queue.get()

        # Add content circuit config
        table_data_circuit = PcdtServer.get_circuit_table_data(data.circuit_data)

        # Add content inductor config
        table_data_inductor = PcdtServer.get_table_data(data.inductor_list)

        # Add content transformer config
        table_data_transformer = PcdtServer.get_table_data(data.transformer_list)

        # Add content heat_sink config
        table_data_heat_sink = PcdtServer.get_heat_sink_table_data(data.heat_sink_list)
        # table_data_heat_sink = PcdtServer.get_table_data(data.heat_sink_list)

        # Add content summary
        table_data_summary = PcdtServer.get_summary_table_data([data.summary_data])

        # Add breakpoint notification text and evaluate breakpoint status
        breakpoint_message = data.break_point_notification
        if len(breakpoint_message) > 0:
            PcdtServer._breakpoint_flag = True
        else:
            PcdtServer._breakpoint_flag = False

        return PcdtServer.templates.TemplateResponse(request=request, name="main_page2.html",
                                                     context={"c_table_data": table_data_circuit,
                                                              "i_table_data": table_data_inductor,
                                                              "t_table_data": table_data_transformer,
                                                              "h_table_data": table_data_heat_sink,
                                                              "s_table_data": table_data_summary,
                                                              "configuration_process_time": PcdtServer.get_format_time(data.conf_process_time),
                                                              "c_selected_filtered_point_index": c_selected_filtered_point_index,
                                                              "text_message": PcdtServer.status_message,
                                                              "break_pt_text": breakpoint_message,
                                                              "user": user})

    @staticmethod
    @app.get("/control_page", response_class=HTMLResponse, response_model=None)
    async def control_page(request: Request, action: str = "", url_back: str = "/html_homepage1") -> _TemplateResponse | HTMLResponse | RedirectResponse:
        """Provide the html-information based on client request to control_page.

        :param request: Request information of the client request
        :type  request: Request
        :param action: Information about the requested action (Keyword driven)
        :type  action: str
        :param url_back: Uniform resource locator for jump back
        :type  url_back: str
        :return: html-page
        :rtype:  _TemplateResponse
        """
        if action == "continue":
            PcdtServer.status_message = "Continue is active"

            request_data: ServerRequestData = ServerRequestData()

            # Request command
            request_data.request_cmd = RequestCmd.continue_opt
            # Later to set the circuit configuration index in case of multiple circuit configurations
            request_data.c_configuration_index = 0
            # Index of circuit filtered point (will be ignored)
            request_data.c_filtered_point_index = 0
            # Request continue
            PcdtServer.server_request_queue.put(request_data)
            # Wait for response
            data: bool = PcdtServer.server_response_queue.get()
            # Go back to main page
            return RedirectResponse(url="/html_homepage1", status_code=303)
        elif action == "pause":
            PcdtServer.status_message = "Pause is active"
        elif action == "stop":
            PcdtServer.status_message = "Stops the server and the optimization (if prog_exit_flag==true)"
            PcdtServer.req_stop.value = 1

        return PcdtServer.templates.TemplateResponse(request=request, name="control_page.html",
                                                     context={"url_back": url_back, "break_status": PcdtServer._breakpoint_flag})

    @staticmethod
    @app.get("/login", response_class=HTMLResponse, response_model=None)
    async def login_page(request: Request) -> _TemplateResponse:
        """Provide the html-information based on client request to login_page.

        :param request: Request information of the client request
        :type  request: Request
        :return: html-page
        :rtype:  _TemplateResponse
        """
        return PcdtServer.templates.TemplateResponse(request=request, name="login.html")

    @staticmethod
    @app.post("/login", response_model=None)
    async def login(request: Request, username: str = Form(...), password: str = Form(...)) -> _TemplateResponse | RedirectResponse:
        """Evaluate the login data and create a session.

        If the form send by '@app.get("/login"' send back this method is called.
        It verify the username and password and provide the suitable evaluation result.
        :param request: Request information of the client request
        :type  request: Request
        :param username: Name of the user
        :type  username: str
        :param password: password
        :type  password: str
        :return: html-page
        :rtype:  _TemplateResponse | RedirectResponse
        """
        if PcdtServer.users.get(username) == password:
            request.session["user"] = username
            # Send back success and request client to request base url '/' with GET-Method (303)
            return RedirectResponse(url="/", status_code=303)

        return PcdtServer.templates.TemplateResponse(request=request, name="login.html", context={"error": "Invalid credentials"})

    @staticmethod
    @app.get("/admin", response_class=HTMLResponse, response_model=None)
    async def admin_page(request: Request, username: str = Depends(get_current_user)) -> _TemplateResponse | RedirectResponse:
        """Evaluate the permission of user and return the suitable page.

        If the form send by '@app.get("/login"' send back this method is called.
        It verify the username and password and provide the suitable evaluation result.
        :param request: Request information of the client request
        :type  request: Request
        :param username: Name of the user
        :type username: str
        :return: html-page
        :rtype:  _TemplateResponse | RedirectResponse
        """
        if not username:
            return RedirectResponse(url="/login")

        return PcdtServer.templates.TemplateResponse(request=request, name="admin.html", context={"user": username})

    @staticmethod
    @app.get("/pareto_front", response_class=HTMLResponse, response_model=None)
    def get_pareto_front(request: Request, button_index: int | None, table_index: int | None, url_back: str = "") -> HTMLResponse:
        """Provide the Pareto-front.

        Later to replace by the selected Pareto-front
        :param request: Request information of the client request
        :type  request: Request
        :param button_index: Index of selected button, if a button was pressed
        :type  button_index: int
        :param table_index: Index of selected table in case of a button press
        :type  table_index: int
        :param url_back: Uniform resource locator for jump back
        :type  url_back: str
        :return: html-page
        :rtype:  HTMLResponse
        """
        # Check for valid parameters
        if button_index is not None and table_index is not None:
            pareto_front_source, item_configuration_index = PcdtServer._calculate_key_parameter(button_index, table_index,
                                                                                                PcdtServer._c_filtered_point_index, 1)
        else:
            pareto_front_source, item_configuration_index = PcdtServer._calculate_key_parameter(-1, -1, 0, 0)

        # Init request for main process
        request_data = ServerRequestData()
        request_data.request_cmd = RequestCmd.pareto_front
        # Later to set the circuit configuration index in case of multiple circuit configurations
        request_data.c_configuration_index = 0
        # Set remaining request parameter
        request_data.item_configuration_index = item_configuration_index
        request_data.pareto_source = pareto_front_source
        # Index of circuit filtered point
        request_data.c_filtered_point_index = PcdtServer._c_filtered_point_index
        # Request continue
        PcdtServer.server_request_queue.put(request_data)
        # Wait for response
        html_data: server_ctl_dtos.QueueParetoFrontData = PcdtServer.server_response_queue.get()

        # Check if result is not valid
        if not html_data.validity:
            # Invalid result page
            return PcdtServer.templates.TemplateResponse(request=request, name="pareto_front.html",
                                                         context={"request": request,
                                                                  "info_string": html_data.evaluation_info,
                                                                  "pareto_front": "No data available",
                                                                  "url_back": url_back})
        else:
            # Add information and back-button
            return PcdtServer.templates.TemplateResponse(request=request, name="pareto_front.html",
                                                         context={"info_string": html_data.evaluation_info,
                                                                  "pareto_front": html_data.pareto_front_optuna,
                                                                  "url_back": url_back})
