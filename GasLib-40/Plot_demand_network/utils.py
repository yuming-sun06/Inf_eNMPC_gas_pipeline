from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.model_diagnostics import (DiagnosticsToolbox)
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
import pyomo.environ as pyo
import pandas as pd
import numpy as np
import os

#==============================================================================
""" debug_model """
#==============================================================================
'''
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.model_diagnostics import (DiagnosticsToolbox)
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
import pyomo.environ as pyo
'''
def debug_gas_model(model):
    
    #Fix degrees of freedom to make the problem square
    for s in model.Stations:
        for t in model.Times:
            model.compressor_P[s, t].fix()
    print("degrees_of_freedom = ", degrees_of_freedom(model))
    #assert degrees_of_freedom(model) == 0
    dt = DiagnosticsToolbox(model)
    dt.report_structural_issues()

    ipopt = pyo.SolverFactory('ipopt')
    ipopt.solve(model, tee=True)
    
    dt = DiagnosticsToolbox(model)
    dt.report_numerical_issues()
    dt.display_variables_at_or_outside_bounds()
    dt.display_underconstrained_set()
    dt.display_overconstrained_set()
    dt.display_constraints_with_large_residuals()
   
def analyze_violations(model):
    # Analyze constraint violations
    violated_constraints = []
    for constr in model.component_data_objects(pyo.Constraint, active=True):
        constr_value = pyo.value(constr)
        if constr_value > 10:  # Constraint is violated
            violated_constraints.append((constr.name, constr_value))
    
    # Sort the violated constraints by the degree of violation
    violated_constraints.sort(key=lambda x: x[1], reverse=True)
    
    # Print the most violated constraints
    print("Most violated constraints:")
    for name, value in violated_constraints[:]:  # Display top 5 violated constraints
        print(f"{name}: {value}")
        
    # Analyze variable bound violations
    violated_variable_bounds = []
    for var in model.component_data_objects(pyo.Var):
        if var.has_lb():
            if pyo.value(var) < var.lb:
                violation_amount = var.lb - pyo.value(var)
                violated_variable_bounds.append((var.name, "Lower bound", violation_amount))
        if var.has_ub():
            if pyo.value(var) > var.ub:
                violation_amount = pyo.value(var) - var.ub
                violated_variable_bounds.append((var.name, "Upper bound", violation_amount))
    
    # Sort the violated variable bounds by the degree of violation
    violated_variable_bounds.sort(key=lambda x: x[2], reverse=True)
    
    # Print the most violated variable bounds
    print("Most violated variable bounds:")
    for name, bound_type, violation_amount in violated_variable_bounds[:5]:  # Display top 5 violated variable bounds
        print(f"{name}: {bound_type} violated by {violation_amount}")

#==============================================================================
""" import data """
#==============================================================================
'''
import pandas as pd
'''
def DataFrame_2levels_to_dict(df):
    """
    Rearrange pipe dataframe with indices: (pipe, vol) and columns: time
    into a dictionary of type {pipe: vol: {t0: .., t1: }}
    """

    dct_interm = pd.DataFrame.to_dict(df, orient = "index")
    dct = {keys[0]: {} for keys in dct_interm.keys()}
    for keys in dct_interm.keys():
        dct[keys[0]][keys[1]] = dct_interm[keys] 
    return dct

#==============================================================================
""" NETWORK DATA """

def import_network_data_from_excel(data_path):
    # ARCS
    df_arcs = pd.read_excel(data_path, sheet_name = "Arcs", index_col = 0)
    Arcs = pd.DataFrame.to_dict(df_arcs, orient = "index") 
    # PIPES
    df_pipes = pd.read_excel(data_path, sheet_name = "Pipes", index_col = 0)
    Pipes = pd.DataFrame.to_dict(df_pipes, orient = "index")      
    # NODES
    df_nodes = pd.read_excel(data_path, sheet_name = "Nodes", index_col = 0)
    Nodes = pd.DataFrame.to_dict(df_nodes, orient = "index")
    
    # STATIONS
    df_stations = pd.read_excel( data_path, sheet_name = "Stations", index_col = 0)
    Stations = pd.DataFrame.to_dict(df_stations, orient = "index")    

    # VALVES
    Valves = {}
    try:
        df_v = pd.read_excel(
            data_path, sheet_name = "Valves", index_col = 0)
        Valves = pd.DataFrame.to_dict(df_v, orient = "index")   
    except:
        print('!!! Valves sheet missing in networkData')
           
    # DATA
    Data = {"Arcs": Arcs, "Pipes": Pipes, "Nodes": Nodes, 
            "Valves" : Valves, 
            "Stations": Stations}   
    return Data

#==============================================================================
"""  TIME VARYING DATA """

## ADJUST DATA
def rearrange_setpoint_data(df):
    """ Rearrange setpoints dataframe with indices: (source, setpoint_type) and columns: time
    into 2 dictionaries (p setpoint and w setpoint) of type {source: {t0: .., t1: }}"""
    pCV = {}
    wCV = {}
    for elem, var in df.index:
        if var == "p":
            pCV[elem] = {t: df.loc[elem,"p"][t] for t in df.columns} 
        elif var == "w":
            wCV[elem] = {t: df.loc[elem,"w"][t] for t in df.columns} 
    return pCV, wCV

def set_pipe_cons_to_default(wcons, Pipes, value = 0):
    """ 
    Set wcons = 0 in all Pipes volumes that are not specified in wcons dictionary 
    """
    if len(list(Pipes.keys())) >= 1:
        ref_pipe = list(Pipes.keys())[0]
        times = wcons[ref_pipe][list(wcons[ref_pipe].keys())[0]].keys()
              
        for pipe in Pipes.keys(): # add wcons = 0 in other pipes volumes
            if pipe not in wcons.keys():
                wcons[pipe] = {}
            for vol in range(1, int(Pipes[pipe]["Nvol"])):
                if vol not in wcons[pipe].keys():
                    wcons[pipe][vol] = {}
                    for counter, t in enumerate(times):
                        wcons[pipe][vol][t] = value
    return wcons

## IMPORT DATA
def import_time_varying_data_from_excel(data_path):
    # SOURCES
    df_sources = pd.read_excel(data_path, sheet_name = "SourcesSP",index_col = [0,1], header = 0)
    pSource, wSource = rearrange_setpoint_data(df_sources)                 
    # WCONS (pipe/nodes)
    df_wcons = pd.read_excel(data_path, sheet_name = "wcons",index_col = [0,1], header = 0)
    wcons = DataFrame_2levels_to_dict(df_wcons)   
    #Input gas parameters
    df_params = pd.read_excel(data_path, sheet_name = "GasParams", header = 0)
    params = pd.DataFrame.to_dict(df_params)   
    Data = {
        "pSource": pSource,"wSource": wSource, "wCons":wcons, "GasParams":params}
    return Data


#==============================================================================
""" ALL """

def import_data_from_excel(network_data_path, input_data_path):
    # topology
    networkData = import_network_data_from_excel(network_data_path)
    # profiles
    inputData = import_time_varying_data_from_excel(input_data_path)
    # add missing consumption in pipes finite volumes
    inputData['wCons'] = set_pipe_cons_to_default(inputData['wCons'], networkData['Pipes']) 
    # try:
    #     inputData['wCons'] = set_pipe_cons_to_default(inputData['wCons'], networkData['Pipes'])
    # except:
    #     print('!!! ERROR importing inputData. No matching with networkData')
    return networkData, inputData

#==============================================================================
""" make_demand_dynamic """
#==============================================================================
'''
import numpy as np
import pyomo.environ as pyo
'''
def dynamic_demand_profile(ss_demand, num_time_periods = 1, epsilon = 0):
    amplitude = 0.175
    time_length = len(ss_demand)
    ss_demand = np.asarray(ss_demand)

    phase = np.mod(np.arange(time_length), 6)
    profile_time = np.array([0, 1, 2, 3, 4, 5, 6])
    profile_shape = np.array([0, 1, 1, 0, -1, -1, 0])

    cyclic_shape = np.interp(phase, profile_time, profile_shape)
    dynamic_demand = ss_demand + (ss_demand + epsilon)*cyclic_shape*amplitude

    return dynamic_demand

def dynamic_demand_profile_extended(dynamic_demand):
    dynamic_demand1 = dynamic_demand
    dynamic_demand2 = dynamic_demand[1:]
    
    dynamic_demand_extended = np.concatenate((dynamic_demand1, dynamic_demand2))
    return dynamic_demand_extended
    
    
def dynamic_demand_calculation(m, num_time_periods = 1, time_length = None, extended_profile = False, epsilon = 0):
    if time_length is None:
        time_length = len(m.Times)
        
    dynamic_demand = {}
    for s in m.wCons:
        if s[0].startswith('sink'):
            ss_demand = [pyo.value(m.wCons[s])]*time_length
            dynamic_demand[s[0]] = dynamic_demand_profile(ss_demand,  num_time_periods, epsilon = epsilon)
            if extended_profile:
                dynamic_demand[s[0]] = dynamic_demand_profile_extended(dynamic_demand[s[0]])
    return dynamic_demand

def uncertain_demand_calculation(m, dynamic_demand, uncertainty={(0,13): -0.1, (13, 25): -0.1}):
    uncertain_demand = {}
    for s in m.wCons:
        if s[0].startswith('sink'):
            uncertain_demand_list = []
            for keys in uncertainty.keys():
                dyn_demand = dynamic_demand[s[0]][keys[0]:keys[1]]
                ss_demand = dynamic_demand[s[0]][0]
                uncertain_demand_profile =  dyn_demand + (dyn_demand - ss_demand)*uncertainty[keys]
                uncertain_demand_list.append(uncertain_demand_profile)
            uncertain_demand[s[0]] = np.concatenate(uncertain_demand_list)
    return uncertain_demand
                
#==============================================================================
""" write_data_to_excel """
#==============================================================================
'''
import pandas as pd
import os
'''
path = r"D:\MPC\pipeline\pipe_rewrite"
def write_data_to_excel(sim_data, m_plant, sheets_keys_dict, file_name, terminal_pressure_slack_dict = None, terminal_flow_slack_dict = None, 
                        controller_1_lyapunov = None,
                        controller_2_lyapunov = None,
                        controller_3_lyapunov = None,
                        controller_multistage_lyapunov = None,
                        iteration_times = None,
                        total_runtime = None):
    writer = pd.ExcelWriter(os.path.join(path, file_name))
    for sheet_name, keys in sheets_keys_dict.items():
        data_dict = {}
        for key in keys:
            data_dict[str(key)] = sim_data.get_data_from_key(key)
        df = pd.DataFrame(data_dict)
        df.to_excel(writer, sheet_name=sheet_name)
            
    df = pd.DataFrame()
    if terminal_pressure_slack_dict is not None:
        df['terminal_pressure_slacks'] = terminal_pressure_slack_dict.values()
    
    if terminal_flow_slack_dict is not None:
        df['terminal_flow_slacks'] = terminal_flow_slack_dict.values()
    df.to_excel(writer, sheet_name="terminal_slacks")
    
    df = pd.DataFrame()
    if controller_1_lyapunov is not None:
        df['controller_1_lyapunov'] = controller_1_lyapunov.values()
    if controller_2_lyapunov is not None:
        df['controller_2_lyapunov'] = controller_2_lyapunov.values()
    if controller_3_lyapunov is not None:
        df['controller_3_lyapunov'] = controller_3_lyapunov.values()
    if controller_multistage_lyapunov is not None:
        df['controller_multistage_lyapunov'] = controller_multistage_lyapunov.values()
    
    df.to_excel(writer, sheet_name="controller_lyapunov")

    if iteration_times is not None or total_runtime is not None:
        df = pd.DataFrame()
        if iteration_times is not None:
            df = pd.DataFrame({
                "iteration": list(iteration_times.keys()),
                "iteration_time_s": list(iteration_times.values()),
            })
        if total_runtime is not None:
            df.loc[len(df)] = ["TOTAL", total_runtime]
        df.to_excel(writer, sheet_name="runtime", index=False)

    writer.close()
