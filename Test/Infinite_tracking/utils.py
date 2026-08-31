from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo



def _dataframe_two_levels_to_dict(frame):
    """Convert a (component, volume) Excel index into nested dictionaries."""
    flat = pd.DataFrame.to_dict(frame, orient="index")
    nested = {key[0]: {} for key in flat}
    for key, values in flat.items():
        nested[key[0]][key[1]] = values
    return nested


def _import_network_data(network_data_path):
    arcs = pd.read_excel(
        network_data_path, sheet_name="Arcs", index_col=0).to_dict(
            orient="index")
    pipes = pd.read_excel(
        network_data_path, sheet_name="Pipes", index_col=0).to_dict(
            orient="index")
    nodes = pd.read_excel(
        network_data_path, sheet_name="Nodes", index_col=0).to_dict(
            orient="index")
    stations = pd.read_excel(
        network_data_path, sheet_name="Stations", index_col=0).to_dict(
            orient="index")
    try:
        valves = pd.read_excel(
            network_data_path, sheet_name="Valves", index_col=0).to_dict(
                orient="index")
    except ValueError:
        valves = {}
        print("!!! Valves sheet missing in networkData")
    return {
        "Arcs": arcs,
        "Pipes": pipes,
        "Nodes": nodes,
        "Valves": valves,
        "Stations": stations,
    }


def _import_time_varying_data(input_data_path):
    sources = pd.read_excel(
        input_data_path, sheet_name="SourcesSP", index_col=[0, 1], header=0)
    pressure_source = {}
    flow_source = {}
    for element, variable in sources.index:
        values = {
            t: sources.loc[element, variable][t] for t in sources.columns}
        if variable == "p":
            pressure_source[element] = values
        elif variable == "w":
            flow_source[element] = values

    consumption = _dataframe_two_levels_to_dict(pd.read_excel(
        input_data_path, sheet_name="wcons", index_col=[0, 1], header=0))
    parameters = pd.read_excel(
        input_data_path, sheet_name="GasParams", header=0).to_dict()
    return {
        "pSource": pressure_source,
        "wSource": flow_source,
        "wCons": consumption,
        "GasParams": parameters,
    }


def import_data_from_excel(network_data_path, input_data_path):
    """Load all network and time-varying data using only local code."""
    network_data = _import_network_data(network_data_path)
    input_data = _import_time_varying_data(input_data_path)
    consumption = input_data["wCons"]
    if network_data["Pipes"]:
        reference_pipe = next(iter(network_data["Pipes"]))
        reference_volume = next(iter(consumption[reference_pipe]))
        times = consumption[reference_pipe][reference_volume].keys()
        for pipe, pipe_data in network_data["Pipes"].items():
            consumption.setdefault(pipe, {})
            for volume in range(1, int(pipe_data["Nvol"])):
                consumption[pipe].setdefault(
                    volume, {time: 0 for time in times})
    return network_data, input_data


def write_data_to_excel(
        sim_data, m_plant, sheets_keys_dict, file_name,
        terminal_pressure_slack_dict=None, terminal_flow_slack_dict=None,
        controller_1_lyapunov=None, controller_2_lyapunov=None,
        controller_3_lyapunov=None, controller_multistage_lyapunov=None):
    """Write simulated trajectories to an explicitly selected local path."""
    output_path = Path(file_name).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        for sheet_name, keys in sheets_keys_dict.items():
            frame = pd.DataFrame()
            for key in keys:
                frame[str(key)] = sim_data.get_data_from_key(key)
            frame.to_excel(writer, sheet_name=sheet_name)

        terminal = pd.DataFrame()
        if terminal_pressure_slack_dict is not None:
            terminal["terminal_pressure_slacks"] = list(
                terminal_pressure_slack_dict.values())
        if terminal_flow_slack_dict is not None:
            terminal["terminal_flow_slacks"] = list(
                terminal_flow_slack_dict.values())
        terminal.to_excel(writer, sheet_name="terminal_slacks")

        lyapunov = pd.DataFrame()
        for column, values in (
                ("controller_1_lyapunov", controller_1_lyapunov),
                ("controller_2_lyapunov", controller_2_lyapunov),
                ("controller_3_lyapunov", controller_3_lyapunov),
                ("controller_multistage_lyapunov",
                 controller_multistage_lyapunov)):
            if values is not None:
                lyapunov[column] = list(values.values())
        lyapunov.to_excel(writer, sheet_name="controller_lyapunov")


def load_css(m_controller, ocss_file_path):
    """Load the cyclic-steady-state reference from the local workbook."""
    if m_controller.component("interm_p_ocss") is not None:
        return m_controller

    sheets = pd.read_excel(ocss_file_path, sheet_name=None)
    for frame in sheets.values():
        if "Unnamed: 0" in frame.columns:
            frame.set_index("Unnamed: 0", inplace=True)

    cycle_length = len(sheets["interm_p"].index) - 1

    def css_time(t):
        return t % cycle_length

    m_controller.interm_p_ocss = pyo.Var(
        m_controller.Pipes_VolExtrR_interm, m_controller.Times)
    for pipe, vol in m_controller.Pipes_VolExtrR_interm:
        column = f"interm_p['{pipe}', {vol}, :]"
        for t in m_controller.Times:
            m_controller.interm_p_ocss[pipe, vol, t].fix(
                sheets["interm_p"].loc[css_time(t), column])

    m_controller.compressor_beta_ocss = pyo.Var(
        m_controller.Stations, m_controller.Times)
    m_controller.compressor_P_ocss = pyo.Var(
        m_controller.Stations, m_controller.Times)
    for station in m_controller.Stations:
        beta_column = f"compressor_beta['{station}', :]"
        power_column = f"compressor_P['{station}', :]"
        for t in m_controller.Times:
            row = css_time(t)
            m_controller.compressor_beta_ocss[station, t].fix(
                sheets["compressor beta"].loc[row, beta_column])
            m_controller.compressor_P_ocss[station, t].fix(
                sheets["compressor power"].loc[row, power_column])

    m_controller.wSource_ocss = pyo.Var(
        m_controller.NodesSources, m_controller.Times)
    m_controller.pSource_ocss = pyo.Var(
        m_controller.NodesSources, m_controller.Times)
    for source in m_controller.NodesSources:
        w_column = f"wSource['{source}', :]"
        p_column = f"pSource['{source}', :]"
        for t in m_controller.Times:
            row = css_time(t)
            m_controller.wSource_ocss[source, t].fix(
                sheets["wSource"].loc[row, w_column])
            m_controller.pSource_ocss[source, t].fix(
                sheets["pSource"].loc[row, p_column])

    return m_controller


def css_terminal_last_point(m, horizon=6, ocss_file_path=None):
    """Add Sakshi's soft terminal pressure/power equalities locally."""
    load_css(m, ocss_file_path)
    m.terminal_p_slack = pyo.Var(
        m.Pipes_VolExtrR_interm, initialize=0, domain=pyo.Reals)
    m.terminal_power_slack = pyo.Var(
        m.Stations, initialize=0, domain=pyo.Reals)
    m.terminal_pressure = pyo.Constraint(
        m.Pipes_VolExtrR_interm,
        rule=lambda model, pipe, vol: (
            model.interm_p[pipe, vol, horizon]
            == model.interm_p_ocss[pipe, vol, 0]
            + model.terminal_p_slack[pipe, vol]))
    m.terminal_supply_flow_css = pyo.Constraint(
        m.Stations,
        rule=lambda model, station: (
            model.compressor_P[station, horizon]
            == model.compressor_P_ocss[station, 0]
            + model.terminal_power_slack[station]))
    return m


def dynamic_demand_calculation(
        m, num_time_periods=1, time_length=None, extended_profile=False,
        epsilon=0.0):
    """Return the 20% sinusoidal demand used by css.xlsx and R_mod.

    This case's CSS and transformed infinite tail use a 1/5 amplitude.  The
    local implementation keeps the finite horizon on the same disturbance.
    """
    if time_length is None:
        time_length = len(m.Times)
    shape = np.sin(np.linspace(0.0, 2.0*num_time_periods*np.pi,
                               time_length))
    result = {}
    for idx in m.wCons:
        if idx[0].startswith(('sink', 'exit')):
            nominal = pyo.value(m.wCons[idx])
            profile = nominal + (nominal + epsilon)*shape/5.0
            if extended_profile:
                # Preserve the upstream convention: first cycle includes both
                # endpoints; subsequent cycles omit their repeated t=0 point.
                profile = np.concatenate(
                    [profile] + [profile[1:]]*4)
            result[idx[0]] = profile
    return result

def init_network_default(m, p_default = 55e5):

    scale = {
        "p": 100000, # pressure
        }

    # pressure initialization
    m.node_p[:, :] = p_default / scale['p']
    m.interm_p[:, :, :] = p_default / scale['p']

    # ! add other default initialization

    return m
