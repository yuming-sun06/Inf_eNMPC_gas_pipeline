import json
import os

import pyomo.contrib.mpc as mpc
import pyomo.environ as pyo
from idaes.core.util.model_statistics import degrees_of_freedom

from model import buildNonLinearModel, init_network_default
from terminal import (
    css_terminal_constraints,
    css_terminal_constraints_soft,
    load_css,
    update_controller_obj,
)
from utils import (
    Tee,
    debug_gas_model,
    dynamic_demand_calculation,
    import_data_from_excel,
    write_data_to_excel,
)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def set_tracking_objective(
        m, ocss_file_path, coeff_p=1.0, coeff_P=1.0,
        terminal_coeff=0.0):
    """Use pressure/power tracking for every controller warm-up solve."""
    load_css(m, ocss_file_path)
    if m.component("tracking_p") is None:
        include_last = len(m.Times) == 1
        m.tracking_p = pyo.Expression(
            expr=coeff_p*sum(
                (m.interm_p[p, vol, t] - m.interm_p_ocss[p, vol, t])**2
                for p, vol in m.Pipes_VolExtrR_interm
                for t in m.Times
                if include_last or t != m.Times.last()
            )
        )
        m.tracking_P = pyo.Expression(
            expr=coeff_P*sum(
                (m.compressor_P[s, t] - m.compressor_P_ocss[s, t])**2
                for s in m.Stations
                for t in m.Times
                if include_last or t != m.Times.last()
            )
        )
    objective = m.tracking_p + m.tracking_P
    if terminal_coeff:
        objective += terminal_coeff*sum(
            m.terminal_p_slack_abs[p, vol]
            for p, vol in m.Pipes_VolExtrR_interm
        )
    m.ObjFun.deactivate()
    if m.component("obj") is not None:
        m.del_component("obj")
    m.obj = pyo.Objective(expr=objective)
    return m


#==============================================================================
""" dynamic_demand """
#==============================================================================

def load_dynamic_demand(m, num_time_periods=1, epsilon=0):
    demand_profiles = dynamic_demand_calculation(
        m,
        num_time_periods=num_time_periods,
        epsilon=epsilon,
    )

    for s in m.wCons:
        if s[0].startswith("sink"):
            for index, t in enumerate(m.Times):
                m.wCons[s[0], 0, t].fix(demand_profiles[s[0]][index])

    return demand_profiles


#==============================================================================
""" run_model """
#==============================================================================

def run_model(
        horizon=24,
        num_time_periods=1,
        network_data_path=None,
        input_data_path=None,
        options_data_path=None,
        ocss_file_path=None,
        periodic_constraints=False,
        calculating_css=False,
        unfix_P_at_t0=False,
        soft=False,
        final_tol=None,
        tracking=False,
        coeff_p=1.0,
        coeff_P=1.0,
        terminal_coeff=1e6):
    if network_data_path is None:
        network_data_path = os.path.join(_BASE_DIR, "test_kai", "networkData.xlsx")
    if input_data_path is None:
        input_data_path = os.path.join(_BASE_DIR, "test_kai", "inputData_longer_horizon.xlsx")
    if options_data_path is None:
        options_data_path = os.path.join(_BASE_DIR, "test_kai", "Options.json")
    if ocss_file_path is None:
        ocss_file_path = os.path.join(_BASE_DIR, "css.xlsx")

    # network and input data
    networkData, inputData = import_data_from_excel(network_data_path, input_data_path)

    #Load options file
    with open(options_data_path, 'r') as file:
        Options = json.load(file)

    #==============================================================================
    """ steady_state_model """
    #==============================================================================

    Options['dynamic'] = False # no time derivative, only one time point: T0+dt
    Options['T'] = Options['T0'] + Options['dt']/3600
    m_steady = buildNonLinearModel(
        networkData, inputData, Options)
    
    # Initialization to default
    m_steady = init_network_default(m_steady, p_default=55e5)

    # Match ABC/Test initialization: solve the initial steady state at 40 bar.
    m_steady.pSource["source_1", :].fix(40)

    t0_steady = m_steady.Times.first()
    initial_beta = {
        "compressorStation_1": 1.2,
        "compressorStation_2": 1.2,
        "compressorStation_3": 1.1,
    }
    for station in m_steady.Stations:
        m_steady.compressor_beta[station, t0_steady].fix(
            initial_beta.get(str(station), 1.0)
        )

    if tracking:
        set_tracking_objective(
            m_steady, ocss_file_path, coeff_p, coeff_P)

    ##################### 1st solve: steady state ###########################
    print("\n================ 1st solve: steady state ================")
    print("degrees_of_freedom =", degrees_of_freedom(m_steady))

    ipopt = pyo.SolverFactory('ipopt')
    ipopt.options['linear_solver'] = 'ma57'
    res_steady = ipopt.solve(m_steady, tee=True)
    try:
        pyo.assert_optimal_termination(res_steady)
    except:
        debug_gas_model(m_steady)

    #==============================================================================
    """ dynamic_model """
    #==============================================================================

    Options['dynamic'] = True
    Options['T'] = horizon

    m_dyn = buildNonLinearModel(
        networkData, inputData, Options)
    
    # Initialize to default
    m_dyn = init_network_default(m_dyn, p_default=55e5)

    # Fix pressure source
    m_dyn.pSource["source_1", :].fix()

    # Fix initial state --> to steady state
    t0 = m_dyn.Times.first()
    for p, vol in m_dyn.Pipes_VolExtrR_interm.data():
        m_dyn.interm_p[p, vol, t0] = m_steady.interm_p[p, vol, t0]
    m_dyn.interm_p[:, :, t0].fix()

    for c in m_dyn.Stations:
        m_dyn.compressor_P[c, t0] = m_steady.compressor_P[c, t0]
    if unfix_P_at_t0:
        m_dyn.compressor_P[:, t0].unfix()
        print("Initial controller compressor_P at t=0 is unfixed")
    else:
        m_dyn.compressor_P[:, t0].fix()

    if tracking:
        set_tracking_objective(
            m_dyn, ocss_file_path, coeff_p, coeff_P)

    ##################### 2nd solve: dynamic state ##########################
    print("\n================ 2nd solve: dynamic state ===============")
    print("degrees_of_freedom =", degrees_of_freedom(m_dyn))

    ipopt = pyo.SolverFactory("ipopt")
    ipopt.options["linear_solver"] = "ma57"
    res_dyn = ipopt.solve(m_dyn, tee=True)
    try:
        pyo.assert_optimal_termination(res_dyn)
    except:
        debug_gas_model(m_dyn)

    # Load dynamic demand profile
    load_dynamic_demand(m_dyn, num_time_periods=num_time_periods)

    # only the hard controller raises this beta upper bound #######################
    if not soft:
        m_dyn.compressor_beta["compressorStation_1", :].setub(4)

    #==============================================================================
    """ terminal_constraints """
    #==============================================================================

    #If we are not calculating css then the initial state of the plant hould not be unfixed
    if calculating_css:
        m_dyn.interm_p[:, :, t0].unfix()
        m_dyn.compressor_P[:, t0].unfix()

        def _terminal_controls(m, c):
            tf = m.Times.last()
            t0 = m.Times.first()
            return m.compressor_P[c, tf] == m.compressor_P[c, t0] 
        m_dyn.terminal_controls_constraint = pyo.Constraint(m_dyn.Stations, rule = _terminal_controls)
     
        def _terminal_pressure(m, p , vol):
            tf = m.Times.last()
            t0 = m.Times.first()
            return m.interm_p[p, vol, tf] ==m.interm_p[p, vol, t0]
        m_dyn.terminal_pressure = pyo.Constraint(m_dyn.Pipes_VolExtrR_interm, rule = _terminal_pressure)

    if periodic_constraints:
        if soft:
            m_dyn = css_terminal_constraints_soft(m_dyn, ocss_file_path)
            m_dyn = update_controller_obj(m_dyn)
        else:
            m_dyn = css_terminal_constraints(m_dyn, ocss_file_path)

    if tracking:
        set_tracking_objective(
            m_dyn,
            ocss_file_path,
            coeff_p,
            coeff_P,
            terminal_coeff=(
                terminal_coeff if periodic_constraints and soft else 0.0
            ),
        )
        
    ##################### 3rd solve: CSS optimization ######################
    print("\n==================== 3rd solve: CSS ====================")
    print("degrees_of_freedom =", degrees_of_freedom(m_dyn))

    if final_tol is not None:
        ipopt.options["tol"] = final_tol
    ipopt.options["mu_init"] = 1e-6
    ipopt.options["bound_push"] = 1e-6
    res_css = ipopt.solve(m_dyn, tee=True)
    pyo.assert_optimal_termination(res_css)

    return m_steady, m_dyn


#==============================================================================
""" main """
#==============================================================================

def main(soft=False):
    ocss_output_path = os.path.join(_BASE_DIR, "css.xlsx")

    m_steady, m_dyn = run_model(
        horizon=6,
        num_time_periods=1,
        periodic_constraints=False,
        calculating_css=True,
        soft=soft,
    )

    # ---- Write output -------------------------------------------------- #
    m_dyn_interface = mpc.DynamicModelInterface(m_dyn, m_dyn.Times)
    sim_data = m_dyn_interface.get_data_at_time(list(m_dyn.Times))

    all_nodes_p = []
    for n in m_dyn.Nodes:
        if str(n).startswith('sink'):
            all_nodes_p.append(m_dyn.node_p[n, :])

    sheets_keys_dict = {
        "compressor power": [m_dyn.compressor_P[s, :] for s in m_dyn.Stations],
        "compressor beta": [m_dyn.compressor_beta[s, :] for s in m_dyn.Stations],
        "wCons": [m_dyn.wCons[s, 0, :] for s in m_dyn.Nodes if str(s).startswith('sink')],
        "node pressure": all_nodes_p,
        "interm_w": [m_dyn.interm_w[p, vol, :] for p, vol in m_dyn.Pipes_VolExtrC_interm],
        "interm_p": [m_dyn.interm_p[p, vol, :] for p, vol in m_dyn.Pipes_VolExtrR_interm],
        "wSource": [m_dyn.wSource[s, :] for s in m_dyn.NodesSources],
        "pSource": [m_dyn.pSource[s, :] for s in m_dyn.NodesSources],
        "pipe_rho": [m_dyn.pipe_rho[p, vol, :] for p, vol in m_dyn.Pipes_VolExtrR],
    }
    write_data_to_excel(sim_data, m_dyn, sheets_keys_dict, ocss_output_path)


if __name__ == "__main__":
    soft = False
    log_path = os.path.join(_BASE_DIR, "css_calculation.log")
    tee = Tee(log_path)
    try:
        print("soft =", soft)
        main(soft=soft)
        print("Log saved to:", log_path)
    finally:
        tee.close()
