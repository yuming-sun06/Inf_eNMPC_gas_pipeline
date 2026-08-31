import pandas as pd
import pyomo.environ as pyo


def set_initial_state_from_css(m, css_file_path):
    """Set the complete t=0 guess and fixed initial states from a CSS file."""
    sheet_components = {
        "compressor power": "compressor_P",
        "compressor beta": "compressor_beta",
        "wCons": "wCons",
        "node pressure": "node_p",
        "interm_w": "interm_w",
        "interm_p": "interm_p",
        "wSource": "wSource",
        "pSource": "pSource",
        "pipe_rho": "pipe_rho",
        "inlet_w": "inlet_w",
        "outlet_w": "outlet_w",
        "u": "u",
        "u2": "u2",
    }
    sheets = pd.read_excel(
        css_file_path,
        sheet_name=list(sheet_components),
        index_col=0,
    )
    t0 = m.Times.first()
    assigned = 0

    for sheet_name, component_name in sheet_components.items():
        component = getattr(m, component_name)
        row = sheets[sheet_name].iloc[0]
        for index in component:
            index_tuple = index if isinstance(index, tuple) else (index,)
            if index_tuple[-1] != t0:
                continue
            spatial_index = index_tuple[:-1]
            if len(spatial_index) == 1:
                column = f"{component_name}['{spatial_index[0]}', :]"
            elif len(spatial_index) == 2:
                column = (
                    f"{component_name}['{spatial_index[0]}', "
                    f"{spatial_index[1]}, :]"
                )
            else:
                continue
            if column not in row.index:
                continue
            component[index].set_value(float(row[column]))
            assigned += 1

    m.interm_p[:, :, t0].fix()
    # m.compressor_P[:, t0].fix() ####################################
    print(f"Initial t=0 loaded from: {css_file_path} ({assigned} values)")
    print(
        "Initial source pressures: "
        + ", ".join(
            f"{source}={pyo.value(m.pSource[source, t0]):.5f} bar"
            for source in m.NodesSources
        )
    )
    return m


#==============================================================================
""" load_css """
#==============================================================================

def load_css(m, ocss_file_path):
    sheets = pd.read_excel(
        ocss_file_path,
        sheet_name=[
            "interm_p",
            "compressor beta",
            "compressor power",
            "pSource",
            "wSource",
        ],
        index_col=0,
    )
    css_period = len(sheets["interm_p"].index) - 1

    m.interm_p_ocss = pyo.Var(m.Pipes_VolExtrR_interm, m.Times)
    for p, vol in m.Pipes_VolExtrR_interm:
        column = "interm_p['" + str(p) + "', " + str(vol) + ", :]"
        for t in m.Times:
            m.interm_p_ocss[p, vol, t].set_value(
                sheets["interm_p"].loc[t % css_period, column]
            )

    m.compressor_beta_ocss = pyo.Var(m.Stations, m.Times)
    for station in m.Stations:
        column = "compressor_beta['" + str(station) + "', :]"
        for t in m.Times:
            m.compressor_beta_ocss[station, t].set_value(
                sheets["compressor beta"].loc[t % css_period, column]
            )

    m.compressor_P_ocss = pyo.Var(m.Stations, m.Times)
    for station in m.Stations:
        column = "compressor_P['" + str(station) + "', :]"
        for t in m.Times:
            m.compressor_P_ocss[station, t].set_value(
                sheets["compressor power"].loc[t % css_period, column]
            )

    m.pSource_ocss = pyo.Var(m.NodesSources, m.Times)
    m.wSource_ocss = pyo.Var(m.NodesSources, m.Times)
    for source in m.NodesSources:
        p_column = "pSource['" + str(source) + "', :]"
        w_column = "wSource['" + str(source) + "', :]"
        for t in m.Times:
            m.pSource_ocss[source, t].set_value(
                sheets["pSource"].loc[t % css_period, p_column]
            )
            m.wSource_ocss[source, t].set_value(
                sheets["wSource"].loc[t % css_period, w_column]
            )

    m.interm_p_ocss.fix()
    m.compressor_beta_ocss.fix()
    m.compressor_P_ocss.fix()
    m.pSource_ocss.fix()
    m.wSource_ocss.fix()
    return m


#==============================================================================
""" hard_terminal_constraint """
#==============================================================================

def css_terminal_constraints(m, ocss_file_path=None):
    load_css(m, ocss_file_path)
    tf = m.Times.last()
    t0 = m.Times.first()

    def _terminal_pipe_pressure(m, p, vol):
        return m.interm_p[p, vol, tf] == m.interm_p_ocss[p, vol, t0]
    m.terminal_pipe_pressure = pyo.Constraint(m.Pipes_VolExtrR_interm, rule=_terminal_pipe_pressure)

    return m


#==============================================================================
""" soft_terminal_constraint """
#==============================================================================

def css_terminal_constraints_soft(m, ocss_file_path=None):
    load_css(m, ocss_file_path)
    tf = m.Times.last()
    t0 = m.Times.first()

    m.terminal_p_slack = pyo.Var(
        m.Pipes_VolExtrR_interm,
        initialize=0,
        domain=pyo.Reals,
    )

    def _terminal_pipe_pressure(m, p, vol):
        return (
            m.interm_p[p, vol, tf]
            == m.interm_p_ocss[p, vol, t0] + m.terminal_p_slack[p, vol]
        )

    m.terminal_pipe_pressure = pyo.Constraint(
        m.Pipes_VolExtrR_interm,
        rule=_terminal_pipe_pressure,
    )
    return m


#==============================================================================
""" soft_terminal_penalty """
#==============================================================================

def update_controller_obj(m, coeff=1e6):
    if m.component("terminal_p_slack_abs") is None:
        m.terminal_p_slack_abs = pyo.Var(
            m.Pipes_VolExtrR_interm,
            initialize=0,
            domain=pyo.NonNegativeReals,
        )

        def _slack_abs_positive(m, p, vol):
            return m.terminal_p_slack_abs[p, vol] >= m.terminal_p_slack[p, vol]

        def _slack_abs_negative(m, p, vol):
            return m.terminal_p_slack_abs[p, vol] >= -m.terminal_p_slack[p, vol]

        m.terminal_p_slack_abs_positive = pyo.Constraint(
            m.Pipes_VolExtrR_interm,
            rule=_slack_abs_positive,
        )
        m.terminal_p_slack_abs_negative = pyo.Constraint(
            m.Pipes_VolExtrR_interm,
            rule=_slack_abs_negative,
        )

    if m.component("obj") is not None:
        m.del_component("obj")
    m.ObjFun.deactivate()
    m.obj = pyo.Objective(
        expr=m.ObjFun
        + coeff * sum(
            m.terminal_p_slack_abs[p, vol]
            for p, vol in m.Pipes_VolExtrR_interm
        )
    )
    return m
