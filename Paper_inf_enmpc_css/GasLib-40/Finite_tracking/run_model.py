import os

import pyomo.contrib.mpc as mpc

from css_calculation import run_model as calculate_model
from utils import Tee, write_data_to_excel

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


#==============================================================================
""" run_model """
#==============================================================================

def run_model(
        horizon=120,
        num_time_periods=5,
        network_data_path=None,
        input_data_path=None,
        options_data_path=None,
        ocss_file_path=None,
        initial_css_file_path=None,
        periodic_constraints=False,
        calculating_css=False,
        soft=False,
        final_tol=1e-5,
        fix_initial_compressor_power=False,
        tracking=False,
        coeff_p=1.0,
        coeff_P=1.0,
        coeff_pSource3=1.0,
        terminal_penalty_coeff=1e6):
    return calculate_model(
        horizon=horizon,
        num_time_periods=num_time_periods,
        network_data_path=network_data_path,
        input_data_path=input_data_path,
        options_data_path=options_data_path,
        ocss_file_path=ocss_file_path,
        initial_css_file_path=initial_css_file_path,
        periodic_constraints=periodic_constraints,
        calculating_css=calculating_css,
        soft=soft,
        final_tol=final_tol,
        fix_initial_compressor_power=fix_initial_compressor_power,
        tracking=tracking,
        coeff_p=coeff_p,
        coeff_P=coeff_P,
        coeff_pSource3=coeff_pSource3,
        terminal_penalty_coeff=terminal_penalty_coeff,
    )


#==============================================================================
""" write_results """
#==============================================================================

def write_results(m_dyn, output_path):
    interface = mpc.DynamicModelInterface(m_dyn, m_dyn.Times)
    sim_data = interface.get_data_at_time(list(m_dyn.Times))

    sheets_keys_dict = {
        "compressor power": [m_dyn.compressor_P[s, :] for s in m_dyn.Stations],
        "compressor beta": [m_dyn.compressor_beta[s, :] for s in m_dyn.Stations],
        "wCons": [
            m_dyn.wCons[s, 0, :]
            for s in m_dyn.Nodes
            if str(s).startswith("sink")
        ],
        "node pressure": [
            m_dyn.node_p[n, :]
            for n in m_dyn.Nodes
            if str(n).startswith("sink")
        ],
        "interm_w": [
            m_dyn.interm_w[p, vol, :]
            for p, vol in m_dyn.Pipes_VolExtrC_interm
        ],
        "interm_p": [
            m_dyn.interm_p[p, vol, :]
            for p, vol in m_dyn.Pipes_VolExtrR_interm
        ],
        "wSource": [m_dyn.wSource[s, :] for s in m_dyn.NodesSources],
        "pSource": [m_dyn.pSource[s, :] for s in m_dyn.NodesSources],
        "pipe_rho": [
            m_dyn.pipe_rho[p, vol, :]
            for p, vol in m_dyn.Pipes_VolExtrR
        ],
    }
    write_data_to_excel(sim_data, m_dyn, sheets_keys_dict, output_path)


#==============================================================================
""" main """
#==============================================================================

def main(soft=False):
    output_path = os.path.join(_BASE_DIR, "run_model.xlsx")
    _, m_dyn = run_model(
        horizon=36,
        num_time_periods=3,
        initial_css_file_path=os.path.join(_BASE_DIR, "css_41.xlsx"),
        ocss_file_path=os.path.join(_BASE_DIR, "css.xlsx"),
        periodic_constraints=True,
        calculating_css=False,
        soft=soft,
    )
    write_results(m_dyn, output_path)


if __name__ == "__main__":
    soft = True
    log_path = os.path.join(_BASE_DIR, "run_model.log")
    tee = Tee(log_path)
    try:
        print("soft =", soft)
        main(soft=soft)
        print("Log saved to:", log_path)
    finally:
        tee.close()
