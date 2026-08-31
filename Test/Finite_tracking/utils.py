import numpy as np
import pandas as pd
import pyomo.environ as pyo
import sys
from idaes.core.util.model_diagnostics import DiagnosticsToolbox
from idaes.core.util.model_statistics import degrees_of_freedom


#==============================================================================
""" write_log_file """
#==============================================================================

class Tee:
    def __init__(self, log_path):
        self._terminal = sys.stdout
        self._log = open(log_path, "w", buffering=1)
        sys.stdout = self

    def write(self, message):
        self._terminal.write(message)
        self._log.write(message)

    def flush(self):
        self._terminal.flush()
        self._log.flush()

    def close(self):
        sys.stdout = self._terminal
        self._log.close()


#==============================================================================
""" debug_model """
#==============================================================================

def debug_gas_model(model):
    for station in model.Stations:
        for time in model.Times:
            model.compressor_P[station, time].fix()

    print("degrees_of_freedom = ", degrees_of_freedom(model))
    diagnostics = DiagnosticsToolbox(model)
    diagnostics.report_structural_issues()

    solver = pyo.SolverFactory("ipopt")
    solver.solve(model, tee=True)

    diagnostics = DiagnosticsToolbox(model)
    diagnostics.report_numerical_issues()
    diagnostics.display_variables_at_or_outside_bounds()
    diagnostics.display_underconstrained_set()
    diagnostics.display_overconstrained_set()
    diagnostics.display_constraints_with_large_residuals()


#==============================================================================
""" import_data_from_excel """
#==============================================================================

def _dataframe_2levels_to_dict(df):
    source = df.to_dict(orient="index")
    result = {key[0]: {} for key in source}
    for key, values in source.items():
        result[key[0]][key[1]] = values
    return result


def _import_network_data(data_path):
    sheets = {
        "Arcs": "Arcs",
        "Pipes": "Pipes",
        "Nodes": "Nodes",
        "Stations": "Stations",
    }
    data = {
        key: pd.read_excel(data_path, sheet_name=sheet, index_col=0)
        .to_dict(orient="index")
        for key, sheet in sheets.items()
    }

    try:
        data["Valves"] = pd.read_excel(
            data_path, sheet_name="Valves", index_col=0
        ).to_dict(orient="index")
    except ValueError:
        data["Valves"] = {}
        print("!!! Valves sheet missing in networkData")

    return data


def _rearrange_setpoint_data(df):
    pressure = {}
    flow = {}
    for component, variable in df.index:
        values = {
            time: df.loc[(component, variable), time]
            for time in df.columns
        }
        if variable == "p":
            pressure[component] = values
        elif variable == "w":
            flow[component] = values
    return pressure, flow


def _set_pipe_consumption_defaults(consumption, pipes, value=0):
    if not pipes:
        return consumption

    reference_pipe = next(iter(consumption))
    reference_volume = next(iter(consumption[reference_pipe]))
    times = consumption[reference_pipe][reference_volume].keys()

    for pipe, pipe_data in pipes.items():
        consumption.setdefault(pipe, {})
        for volume in range(1, int(pipe_data["Nvol"])):
            consumption[pipe].setdefault(
                volume, {time: value for time in times}
            )
    return consumption


def _import_time_varying_data(data_path):
    sources = pd.read_excel(
        data_path, sheet_name="SourcesSP", index_col=[0, 1]
    )
    pressure, flow = _rearrange_setpoint_data(sources)

    consumption = _dataframe_2levels_to_dict(
        pd.read_excel(data_path, sheet_name="wcons", index_col=[0, 1])
    )
    gas_parameters = pd.read_excel(
        data_path, sheet_name="GasParams"
    ).to_dict()

    return {
        "pSource": pressure,
        "wSource": flow,
        "wCons": consumption,
        "GasParams": gas_parameters,
    }


def import_data_from_excel(network_data_path, input_data_path):
    network_data = _import_network_data(network_data_path)
    input_data = _import_time_varying_data(input_data_path)
    input_data["wCons"] = _set_pipe_consumption_defaults(
        input_data["wCons"], network_data["Pipes"]
    )
    return network_data, input_data


#==============================================================================
""" dynamic_demand_calculation """
#==============================================================================

def dynamic_demand_calculation(m, num_time_periods=1, time_length=None, epsilon=0):
    if time_length is None:
        time_length = len(m.Times)

    shape = np.sin(
        np.linspace(0, 2 * num_time_periods * np.pi, time_length)
    )

    dynamic_demand = {}
    for s in m.wCons:
        if s[0].startswith("sink") or s[0].startswith("exit"):
            ss_demand = pyo.value(m.wCons[s])
            dynamic_demand[s[0]] = ss_demand + (ss_demand + epsilon) * shape / 5
    return dynamic_demand


#==============================================================================
""" write_data_to_excel """
#==============================================================================

def write_data_to_excel(
        sim_data, model, sheets_keys_dict, file_name,
        terminal_pressure_slack_dict=None,
        terminal_flow_slack_dict=None,
        controller_1_lyapunov=None,
        controller_2_lyapunov=None,
        controller_3_lyapunov=None,
        controller_multistage_lyapunov=None,
        iteration_times=None,
        total_runtime=None):
    with pd.ExcelWriter(file_name) as writer:
        for sheet_name, keys in sheets_keys_dict.items():
            data = {
                str(key): sim_data.get_data_from_key(key)
                for key in keys
            }
            pd.DataFrame(data).to_excel(writer, sheet_name=sheet_name)

        slacks = pd.DataFrame()
        if terminal_pressure_slack_dict is not None:
            slacks["terminal_pressure_slacks"] = (
                terminal_pressure_slack_dict.values()
            )
        if terminal_flow_slack_dict is not None:
            slacks["terminal_flow_slacks"] = terminal_flow_slack_dict.values()
        slacks.to_excel(writer, sheet_name="terminal_slacks")

        lyapunov = pd.DataFrame()
        for column, values in (
            ("controller_1_lyapunov", controller_1_lyapunov),
            ("controller_2_lyapunov", controller_2_lyapunov),
            ("controller_3_lyapunov", controller_3_lyapunov),
            ("controller_multistage_lyapunov", controller_multistage_lyapunov),
        ):
            if values is not None:
                lyapunov[column] = values.values()
        lyapunov.to_excel(writer, sheet_name="controller_lyapunov")

        if iteration_times is not None or total_runtime is not None:
            runtime = pd.DataFrame()
            if iteration_times is not None:
                runtime = pd.DataFrame({
                    "iteration": list(iteration_times),
                    "iteration_time_s": list(iteration_times.values()),
                })
            if total_runtime is not None:
                runtime.loc[len(runtime)] = ["TOTAL", total_runtime]
            runtime.to_excel(writer, sheet_name="runtime", index=False)
