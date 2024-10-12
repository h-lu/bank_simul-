# Filename: advanced_central_bank_simulator.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# 更新默认参数
@st.cache_data
def get_default_params():
    params = {
        'C0': 100,  # 自主消费
        'C1': 0.8,  # 边际消费倾向
        'I0': 100,  # 自主投资
        'I1': 0.2,  # 投资对利率的敏感度
        'G': 100,   # 政府支出
        'T': 100,   # 税收
        'NX0': 50,  # 自主净出口
        'NX1': 0.1, # 净出口对汇率的敏感度
        'L0': 0.2,  # 货币需求收入弹性
        'L1': 0.1,  # 货币需求利率弹性
        'L2': 0.05, # 货币需求汇率弹性
        'M': 1000,  # 货币供给
        'P': 1,     # 初始价格水平
        'Yn': 1000, # 潜在产出
        'un': 0.05, # 自然失业率
        'alpha': 2, # 奥肯系数
        'beta': 0.5,# 菲利普斯曲线斜率
        'pi_e': 0.02,    # 初始预期通胀率
        'kappa': 0.05,   # 价格调整速度
        'rho': 0.7,      # 预期调整参数
        'phi_pi': 1.5,   # 泰勒规则通胀系数
        'phi_y': 0.5,    # 泰勒规则产出缺口系数
        'r_star': 0.02,  # 自然利率
        'g': 0.02,       # 长期增长率
        'i_foreign': 0.02, # 国外利率
    }
    return params

# 利率平价条件
def interest_rate_parity(i, i_foreign, E, E_expected):
    return i - i_foreign - (E_expected - E) / E

# 预期形成
def form_expectations(pi_e, pi, rho):
    return rho * pi_e + (1 - rho) * pi

# 泰勒规则
def taylor_rule(pi, Y, Yn, r_star, phi_pi, phi_y):
    return r_star + phi_pi * (pi - 0.02) + phi_y * (Y - Yn) / Yn

# 长期增长
def long_term_growth(Yn, g):
    return Yn * (1 + g)

def solve_model(params, G, T, M, i, E_external=None, shock_type=None, shock_value=0.0):
    Yn = params['Yn']
    P = params['P']
    pi_e = params['pi_e']
    
    def equations(vars):
        Y, i_var, E_var, P_new, pi_e_new = vars
        
        # IS曲线
        C = params['C0'] + params['C1'] * (Y - T)
        I = params['I0'] - params['I1'] * i_var
        NX = params['NX0'] - params['NX1'] * E_var
        eq1 = Y - (C + I + G + NX)
        
        # LM曲线
        eq2 = M / P_new - (params['L0'] * Y - params['L1'] * i_var + params['L2'] * E_var)
        
        # BP曲线 (利率平价条件)
        E_expected = E_var * (1 + pi_e_new)
        eq3 = interest_rate_parity(i_var, params['i_foreign'], E_var, E_expected)
        
        # 价格调整
        pi = (P_new - P) / P
        eq4 = P_new - P * (1 + params['kappa'] * (Y - Yn) / Yn)
        
        # 预期形成
        eq5 = pi_e_new - form_expectations(pi_e, pi, params['rho'])
        
        return [eq1, eq2, eq3, eq4, eq5]
    
    initial_guess = [Yn, i, 1.0, P, pi_e]
    
    try:
        Y_sol, i_sol, E_sol, P_sol, pi_e_sol = fsolve(equations, initial_guess)
    except:
        Y_sol, i_sol, E_sol, P_sol, pi_e_sol = [np.nan] * 5
    
    u_sol = params['un'] - params['alpha'] * (Y_sol - Yn) / Yn
    pi_sol = (P_sol - P) / P
    
    return {
        'Y': Y_sol,
        'i': i_sol,
        'E': E_sol,
        'P': P_sol,
        'pi_e': pi_e_sol,
        'u': u_sol,
        'pi': pi_sol,
    }

def simulate_dynamic(params, G, T, M_initial, i_initial, E_initial, P_initial, pi_e_initial, E_external=None, shock_type=None, shock_value=0.0, periods=20):
    Y_history = [params['Yn']]
    i_history = [i_initial]
    E_history = [E_initial]
    P_history = [P_initial]
    pi_e_history = [pi_e_initial]
    u_history = [params['un']]
    pi_history = [0]
    M = M_initial
    
    for t in range(periods):
        if shock_type == '技术冲击':
            params['Yn'] *= (1 + shock_value)
        elif shock_type == '需求冲击':
            params['C0'] *= (1 + shock_value)
        elif shock_type == '货币政策冲击':
            M *= (1 + shock_value)
        
        result = solve_model(params, G, T, M, i_history[-1], E_external)
        
        Y_history.append(result['Y'])
        i_history.append(result['i'])
        E_history.append(result['E'])
        P_history.append(result['P'])
        pi_e_history.append(result['pi_e'])
        u_history.append(result['u'])
        pi_history.append(result['pi'])
        
        params['Yn'] = long_term_growth(params['Yn'], params['g'])
    
    history = {
        '产出 (Y)': Y_history,
        '利率 (i)': i_history,
        '汇率 (E)': E_history,
        '价格水平 (P)': P_history,
        '预期通胀率 (π_e)': pi_e_history,
        '失业率 (u)': u_history,
        '通胀率 (π)': pi_history,
    }
    
    return history

def get_variable_explanation(variable):
    explanations = {
        '产出 (Y)': """
        当产出(Y)增加时：
        - 利率(i)可能上升，因为更高的收入会增加货币需求。
        - 价格水平(P)可能上升，因为需求增加。
        - 失业率(u)可能下降，因为更多的劳动力被雇佣。
        - 通胀率(π)可能上升，因为需求压力增加。
        """,
        '利率 (i)': """
        当利率(i)上升时：
        - 产出(Y)可能下降，因为投资和消费减少。
        - 汇率(E)可能升值，因为资本流入增加。
        - 价格水平(P)可能下降，因为总需求减少。
        - 失业率(u)可能上升���因为经济活动减少。
        """,
        '汇率 (E)': """
        当汇率(E)升值时：
        - 产出(Y)可能下降，因为净出口减少。
        - 价格水平(P)可能下降，因为进口商品变得更便宜。
        - 通胀率(π)可能下降，因为进口通胀压力减小。
        """,
        '价格水平 (P)': """
        当价格水平(P)上升时：
        - 产出(Y)可能下降，因为实际余额效应。
        - 利率(i)可能上升，因为货币需求增加。
        - 汇率(E)可能贬值，因为本国商品相对变贵。
        - 失业率(u)可能上升，因为实际工资下降导致劳动需求减少。
        """,
        '预期通胀率 (π_e)': """
        当预期通胀率(π_e)上升时：
        - 利率(i)可能上升，因为人们要求更高的名义利率。
        - 价格水平(P)可能上升，因为工资和价格设定行为改变。
        - 实际产出(Y)可能短期内增加，但长期可能不变或下降。
        """,
        '失业率 (u)': """
        当失业率(u)上升时：
        - 产出(Y)可能下降，因为总需求减少。
        - 价格水平(P)和通胀率(π)可能下降，因为工资压力减小。
        - 利率(i)可能下降，因为中央银行可能采取扩张性货币政策。
        """,
        '通胀率 (π)': """
        当通胀率(π)上升时：
        - 利率(i)可能上升，因为中央银行可能采取紧缩性货币政策。
        - 汇率(E)可能贬值，因为本国货币购买力下降。
        - 失业率(u)可能短期内下降，但长期可能上升（菲利普斯曲线）。
        - 预期通胀率(π_e)可能上升，因为人们调整对未来通胀的预期。
        """
    }
    return explanations.get(variable, "没有该变量的具体解释。")

# Streamlit 应用界面
st.title('中央银行政策影响模拟器')

# 参数设置
params = get_default_params()

# 用户输入
col1, col2, col3 = st.columns(3)

with col1:
    G = st.number_input('政府支出 (G)', value=float(params['G']), step=10.0, format="%.1f")
    T = st.number_input('税收 (T)', value=float(params['T']), step=10.0, format="%.1f")
    initial_money_supply = st.number_input('初始货币供给 (M)', value=float(params['M']), step=100.0, format="%.1f")

with col2:
    initial_interest_rate = st.number_input('初始利率 (i)', value=0.05, step=0.01, format="%.2f")
    exchange_rate_option = st.selectbox('汇率制度', ['内生', '外生'])
    E = st.number_input('外生汇率 (如适用)', value=1.0, step=0.1, format="%.1f") if exchange_rate_option == '外生' else 1.0

with col3:
    shock_type = st.selectbox('冲击类型', ['无冲击', '技术冲击', '需求冲击', '货币政策冲击'])
    shock_value = st.number_input('冲击幅度 (%)', value=0.0, step=1.0, format="%.1f") / 100 if shock_type != '无冲击' else 0.0

# 高级参数设置
with st.expander("高级参数设置"):
    col1, col2, col3 = st.columns(3)
    with col1:
        params['C0'] = st.number_input('自主消费 (C0)', value=float(params['C0']), step=10.0, format="%.1f")
        params['C1'] = st.number_input('边际消费倾向 (C1)', value=params['C1'], step=0.1, format="%.2f")
        params['I0'] = st.number_input('自主投资 (I0)', value=float(params['I0']), step=10.0, format="%.1f")
        params['I1'] = st.number_input('投资利率敏感度 (I1)', value=params['I1'], step=0.1, format="%.2f")
    with col2:
        params['NX0'] = st.number_input('自主净出口 (NX0)', value=float(params['NX0']), step=10.0, format="%.1f")
        params['NX1'] = st.number_input('净出口汇率敏感度 (NX1)', value=params['NX1'], step=0.1, format="%.2f")
        params['L0'] = st.number_input('货币需求收入弹性 (L0)', value=params['L0'], step=0.1, format="%.2f")
        params['L1'] = st.number_input('货币需求利率弹性 (L1)', value=params['L1'], step=0.1, format="%.2f")
    with col3:
        params['kappa'] = st.number_input('价格调整速度 (κ)', value=params['kappa'], step=0.01, format="%.2f")
        params['rho'] = st.number_input('预期调整参数 (ρ)', value=params['rho'], step=0.1, format="%.2f")
        params['phi_pi'] = st.number_input('泰勒规则通胀系数 (φ_π)', value=params['phi_pi'], step=0.1, format="%.2f")
        params['phi_y'] = st.number_input('泰勒规则产出缺口系数 (φ_y)', value=params['phi_y'], step=0.1, format="%.2f")

# 执行模拟
if st.button('运行模拟'):
    history = simulate_dynamic(
        params=params,
        G=G,
        T=T,
        M_initial=initial_money_supply,
        i_initial=initial_interest_rate,
        E_initial=E if exchange_rate_option == '外生' else 1.0,
        P_initial=params['P'],
        pi_e_initial=params['pi_e'],
        E_external=E if exchange_rate_option == '外生' else None,
        shock_type=shock_type if shock_type != '无冲击' else None,
        shock_value=shock_value,
        periods=20
    )

    # 转换为数据框
    df_history = pd.DataFrame(history)

    # 为每个经济指标创建单独的图表
    for column in df_history.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df_history.index, df_history[column], marker='o')
        
        ax.set_title(f'{column}随时间的变化')
        ax.set_xlabel('时期')
        ax.set_ylabel('数值')
        
        # 添加网格线
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 设置y轴的范围，使零点更明显
        y_min = min(0, df_history[column].min() * 1.1)
        y_max = max(0, df_history[column].max() * 1.1)
        ax.set_ylim(y_min, y_max)

        # 在Streamlit中显示图表
        st.pyplot(fig)
        
        # 添加变量解释
        st.markdown(f"### {column}的影响")
        st.markdown(get_variable_explanation(column))

    # 显示数据表格
    st.subheader("模拟结果数据")
    st.dataframe(df_history)

# 添加重置按钮
if st.button("重置模型"):
    st.rerun()

# 添加说明文字
st.markdown("""
## 模型说明

这个中央银行政策影响模拟器基于IS-LM-BP模型，并加入了菲利普斯曲线、奥肯法则等元素。它可以帮助你理解不同经济政策和外部冲击对经济的影响。

### 主要参数说明：
- **政府支出 (G)**: 政府购买商品和服务的支出。
- **税收 (T)**: 政府从经济中征收的税款。
- **初始货币供给 (M)**: 经济中的初始货币量。
- **初始利率 (i)**: 经济中的初始利率水平。
- **汇率制度**: 选择汇率是内生决定还是外生给定。
- **冲击类型和幅度**: 模拟不同类型的经济冲击及其强度。

### 高级参数：
包括消费函数、投资函数、货币需求函数等的具体参数，以及价格调整速度、预期形成等新增的动态要素。

### 输出变量：
- **产出 (Y)**: 经济的总产出或GDP。
- **利率 (i)**: 经济中的利率水平。
- **汇率 (E)**: 本国货币相对外国货币的价值。
- **价格水平 (P)**: 经济中的总体价格水平。
- **预期通胀率 (π_e)**: 经济主体预期的未来通胀率。
- **失业率 (u)**: 劳动力中失业的比例。
- **通胀率 (π)**: 价格水平的变化率。

通过调整不同的参数，你可以观察经济如何对不同的政策和冲击做出反应。这有助于理解宏观经济政策的效果和经济系统的动态特性。
""")