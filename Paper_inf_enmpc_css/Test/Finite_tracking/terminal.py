import pandas as pd
import pyomo.environ as pyo


#==============================================================================
""" load_css """
#==============================================================================

def load_css(m, ocss_file_path):
    if m.component("interm_p_ocss") is not None:
        return m

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
