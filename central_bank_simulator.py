# Filename: advanced_central_bank_simulator.py

import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import plotly.graph_objects as go
import copy

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
        
        # 确保Y不为负
        Y = max(Y, 0.1)  # 设置一个很小的正数作为下限
        
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
    
    u_sol = max(0, params['un'] - params['alpha'] * (Y_sol - Yn) / Yn)
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
        
        # 添加检查和约束
        Y_sol = max(result['Y'], 0.1)  # 确保Y不为负
        i_sol = max(result['i'], 0)    # 确保利率不为负
        E_sol = max(result['E'], 0.1)  # 确保汇率不为负
        P_sol = max(result['P'], 0.1)  # 确保价格水平不为负
        
        Y_history.append(Y_sol)
        i_history.append(i_sol)
        E_history.append(E_sol)
        P_history.append(P_sol)
        pi_e_history.append(result['pi_e'])
        u_history.append(result['u'])
        pi_history.append(result['pi'])
        
        # 调整潜在产出的增长速度
        params['Yn'] = long_term_growth(params['Yn'], min(params['g'], 0.02))  # 限制最大增长率为2%
    
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
        - 利率(i)可能上，因为更高的收入会增加货需求。
        - 价格水平(P)可能上升，因为需求增加。
        - 失业率(u)可能下降，因为更多的劳动力被雇佣。
        - 通胀率(π)可能上升，因为需求压力增加。
        """,
        '利率 (i)': """
        当利率(i)上升时：
        - 产出(Y)可能下降，因为投资和消费减少。
        - 汇率(E)可升值，因为资本流入增加。
        - 价格水平(P)可能下降，因为总需求减少。
        - 失业率(u)可能上升因为经济活动减少。
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
        - 利率(i)可能上升，因人们要求更高的名义利率。
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

def generate_detailed_explanation(base_results, policy_results, G_base, G_policy, T_base, T_policy, M_base, M_policy):
    explanations = []
    
    # GDP解释
    gdp_change = (policy_results['产出 (Y)'][-1] - base_results['产出 (Y)'][-1]) / base_results['产出 (Y)'][-1] * 100
    if abs(gdp_change) < 0.1:
        explanations.append("GDP基本保持不变，政策对总产出的影响较小。")
    elif gdp_change > 0:
        explanations.append(f"GDP增加了{gdp_change:.2f}%。这可能是由于{'政府支出加' if G_policy > G_base else '货币供给增加' if M_policy > M_base else '其他因素'}刺激了总需求。")
    else:
        explanations.append(f"GDP减少了{abs(gdp_change):.2f}%。这可能是由于{'政府支出减少' if G_policy < G_base else '货币供给减少' if M_policy < M_base else '其他因素'}抑制了总求。")

    # 利率解释
    interest_change = policy_results['利率 (i)'][-1] - base_results['利率 (i)'][-1]
    if abs(interest_change) < 0.001:
        explanations.append("利率基本保持稳定，说明货币市场的供需关系没有显著变化。")
    elif interest_change > 0:
        explanations.append(f"利率上升了{interest_change:.3f}个百分点。这可能是由于{'总需求增加导致货币需求上升' if gdp_change > 0 else '货币供给减少'}。")
    else:
        explanations.append(f"利率下降了{abs(interest_change):.3f}个百分点。这可能是由于{'总需求减少导致货币需求下降' if gdp_change < 0 else '货币供给增加'}。")

    # 价格水���解释
    price_change = (policy_results['价格水平 (P)'][-1] - base_results['价格水平 (P)'][-1]) / base_results['价格水平 (P)'][-1] * 100
    if abs(price_change) < 0.1:
        explanations.append("价格水平基本稳定，说明政策没有显著影响通胀通缩。")
    elif price_change > 0:
        explanations.append(f"价格水平上升了{price_change:.2f}%。这表明{'总需求增加导致了轻微的通胀压力' if gdp_change > 0 else '成本推动型通胀可能发生'}。")
    else:
        explanations.append(f"价格水平下降了{abs(price_change):.2f}%。这表明{'总需求减少导致了轻微的通缩压力' if gdp_change < 0 else '生产效率提高或者原材料成本下降'}。")

    # 失业率解释
    unemployment_change = policy_results['失业率 (u)'][-1] - base_results['失业率 (u)'][-1]
    if abs(unemployment_change) < 0.001:
        explanations.append("失业率基本保持不变，就业市场相对稳定。")
    elif unemployment_change > 0:
        explanations.append(f"失业率上升了{unemployment_change:.3f}个百分点。这可能是由于{'经济增速放缓' if gdp_change < 0 else '结构性因素或摩擦性失业增加'}。")
    else:
        explanations.append(f"失业率下降了{abs(unemployment_change):.3f}个百分点。这表明{'经济增长创造了更多就业机会' if gdp_change > 0 else '劳动力市场结构得到改善'}。")

    # 通胀率解释
    inflation_change = policy_results['通胀率 (π)'][-1] - base_results['通胀率 (π)'][-1]
    if abs(inflation_change) < 0.001:
        explanations.append("通胀率基本保持稳定，物价变动不大。")
    elif inflation_change > 0:
        explanations.append(f"通胀率上升了{inflation_change:.3f}个百分点。这可能是由于{'总需求增加推高了物价' if gdp_change > 0 else '成本推动或者通胀预期上升'}。")
    else:
        explanations.append(f"通胀率下降了{abs(inflation_change):.3f}个百分点。这可能是由于{'总需求减少降低了物价压力' if gdp_change < 0 else '生产效率提高或者原材料成本下降'}。")

    # 政策传导机制解释
    if G_policy != G_base:
        explanations.append(f"政府支出{'增加' if G_policy > G_base else '减少'}直接影响了IS曲线，导致其{'右移' if G_policy > G_base else '左移'}。")
    if M_policy != M_base:
        explanations.append(f"货币供给{'增加' if M_policy > M_base else '减少'}影响了LM曲线，导致其{'右移' if M_policy > M_base else '左移'}。")
    if interest_change != 0:
        explanations.append(f"利率变化可能影响国际资本流动，进而影响BP曲线。")

    # 长期效应讨论
    if gdp_change > 0:
        explanations.append("长期来看，需要关注这种增长是否可持续，以及是否会带来通胀压力。")
    elif gdp_change < 0:
        explanations.append("长期来看，需要关注经济是否能够自我调节回到潜在产出水平。")
    
    if G_policy > G_base:
        explanations.append("政府支出增加可能带来挤出效应，影响私人投资。同时需要关注财政可持续性问题。")
    
    if M_policy > M_base:
        explanations.append("货币供给增加在长期可能导致通胀预期上升，影响价格稳定。")

    # 政策建议
    explanations.append("建议:")
    if abs(gdp_change) > 2:
        explanations.append("- 密切关注经济长的可持续性和质量。")
    if abs(inflation_change) > 0.02:
        explanations.append("- 警惕通胀或通缩风险，必要时调整货币政策。")
    if abs(unemployment_change) > 0.01:
        explanations.append("- 关就业市场变化，考虑实施相应的劳动力市场政策。")
    explanations.append("- 持续监测各项经济指标，根据实际情况及时调整政策。")

    return "\n".join(explanations)

def policy_comparison_demo(params):
    st.subheader("政策对比演示")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**基准景**")
        G_base = st.number_input("政府支出 (G)", value=float(params['G']), step=10.0, key="G_base")
        T_base = st.number_input("税收 (T)", value=float(params['T']), step=10.0, key="T_base")
        M_base = st.number_input("货币供给 (M)", value=float(params['M']), step=100.0, key="M_base")
        i_base = st.number_input("初始利率 (i)", value=0.05, step=0.01, format="%.2f", key="i_base")

    with col2:
        st.markdown("**政策情景**")
        G_policy = st.number_input("政府支出 (G)", value=float(params['G'])*1.2, step=10.0, key="G_policy")
        T_policy = st.number_input("税收 (T)", value=float(params['T']), step=10.0, key="T_policy")
        M_policy = st.number_input("货币供给 (M)", value=float(params['M']), step=100.0, key="M_policy")
        i_policy = st.number_input("初始利率 (i)", value=0.05, step=0.01, format="%.2f", key="i_policy")

    if st.button("运行对比模拟"):
        base_params = copy.deepcopy(params)
        policy_params = copy.deepcopy(params)
        
        base_results = simulate_dynamic(base_params, G_base, T_base, M_base, i_base, 1.0, base_params['P'], base_params['pi_e'])
        policy_results = simulate_dynamic(policy_params, G_policy, T_policy, M_policy, i_policy, 1.0, policy_params['P'], policy_params['pi_e'])

        # 创建对比数据框
        comparison_df = pd.DataFrame({
            '指标': ['初始GDP', '最终GDP', '初始利率', '最终利率', '初始价格水平', '最终价格水平', '初始失业率', '最终失业率', '初始通胀率', '最终通胀率'],
            '基准情景': [base_results['产出 (Y)'][0], base_results['产出 (Y)'][-1], 
                        base_results['利率 (i)'][0], base_results['利率 (i)'][-1], 
                        base_results['价格水平 (P)'][0], base_results['价格水平 (P)'][-1], 
                        base_results['失业率 (u)'][0], base_results['失业率 (u)'][-1], 
                        base_results['通胀率 (π)'][0], base_results['通胀率 (π)'][-1]],
            '政策情景': [policy_results['产出 (Y)'][0], policy_results['产出 (Y)'][-1], 
                        policy_results['利率 (i)'][0], policy_results['利率 (i)'][-1], 
                        policy_results['价格水平 (P)'][0], policy_results['价格水平 (P)'][-1], 
                        policy_results['失业率 (u)'][0], policy_results['失业率 (u)'][-1], 
                        policy_results['通胀率 (π)'][0], policy_results['通胀率 (π)'][-1]]
        })

        st.subheader("模拟结果对比")
        st.table(comparison_df)

        # 为每个指标创建单独的图表
        indicators = ['产出 (Y)', '利率 (i)', '价格水平 (P)', '失业率 (u)', '通胀率 (π)']
        for indicator in indicators:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(base_results[indicator]))), 
                                     y=base_results[indicator], 
                                     mode='lines', 
                                     name='基准情景'))
            fig.add_trace(go.Scatter(x=list(range(len(policy_results[indicator]))), 
                                     y=policy_results[indicator], 
                                     mode='lines', 
                                     name='政策情景'))
            fig.update_layout(title=f'{indicator}随时间的变化', xaxis_title='时间', yaxis_title=indicator)
            st.plotly_chart(fig)

        # 使用新的解释生成函数
        detailed_explanation = generate_detailed_explanation(
            base_results, policy_results, 
            G_base, G_policy, T_base, T_policy, M_base, M_policy
        )

        st.markdown("### 详细结果解释")
        st.markdown(detailed_explanation)

        # 添加失业率解释
        st.markdown("### 失业率说明")
        st.markdown("""
        注意：模型计算的失业率可能会出现非常低的值，这是由于模型的简化性质和长期增长假设导致的。
        在实际经济中，失业率通常不会降到非常低的水平。这个结果提醒我们，模型虽然有助于理解经济
        政策的影响，但仍然是对现实的简化表示。在解释结果时，我们应该更多地关注失业率的变化趋势，
        而不是具体的数值。
        """)

# 主应用界面
st.title('中央银行政策影响模拟器')

# 创建侧边栏菜单
menu = st.sidebar.selectbox(
    "选择功能",
    ("参数设置和模拟", "政策对比演示", "模型说明")
)

if menu == "参数设置和模拟":
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

        # 为每个经济指标创建图表
        for column in df_history.columns:
            st.subheader(f'{column}随时间的变化')
            st.line_chart(df_history[column])
            
            # 添加变量解释
            st.markdown(f"### {column}的影响")
            st.markdown(get_variable_explanation(column))

        # 显示数据表格
        st.subheader("模拟结果数据")
        st.dataframe(df_history)

    # 添加重置按钮
    if st.button("重置模型"):
        st.rerun()

elif menu == "政策对比演示":
     # 参数设置
    params = get_default_params()
    
    policy_comparison_demo(params)

elif menu == "模型说明":
    st.markdown("## 模型说明")

    st.markdown("这个中央银行政策影响模拟器基于多个宏观经济模型,包括IS-LM-BP模型、菲利普斯曲线、奥肯法则等。下面是对各个模型的详细解释:")

    st.markdown("### 1. IS-LM-BP 模型")

    st.markdown("#### IS 曲线(投资-储蓄均衡):")
    st.latex(r'''
    \begin{align*}
    Y &= C + I + G + NX \\
    C &= C_0 + C_1(Y - T) \\
    I &= I_0 - I_1i \\
    NX &= NX_0 - NX_1E
    \end{align*}
    ''')

    st.markdown("这个模型描述了商品市场的均衡。Y是总产出,C是消费,I是投资,G是政府支出,NX是净出口。消费取决于可支配收入(Y-T),投资受利率(i)影响,净出口受汇率(E)影响。")

    st.markdown("#### LM 曲线(流动性偏好-货币供给均衡):")
    st.latex(r'\frac{M}{P} = L_0Y - L_1i + L_2E')

    st.markdown("这描述了货币市场的均衡。M是货币供给,P是价格水平。货币需求取决于收入(Y),利率(i)和汇率(E)。")

    st.markdown("#### BP 曲线(国际收支平衡):")
    st.latex(r'i = i_{foreign} + \frac{E_{expected} - E}{E}')

    st.markdown("这是未覆盖利率平价条件,描述了国际资本流动的均衡。本国利率(i)等于国外利率(i_foreign)加上预期汇率变化。")

    st.markdown("### 2. 价格调整方程:")
    st.latex(r'P_{new} = P \left(1 + \kappa \frac{Y - Y_n}{Y_n}\right)')

    st.markdown("这个方程描述了价格如何随时间调整。当实际产出(Y)高于潜在产出(Y_n)时,价格上升;反之则下降。κ是价格调整速度。")

    st.markdown("### 3. 预期形成方程:")
    st.latex(r'\pi^e_{new} = \rho \pi^e + (1 - \rho)\pi')

    st.markdown("这个方程描述了人们如何形成对未来通胀的预期。新的预期通胀率(π^e_new)是当前预期(π^e)和实际通胀率(π)的加权平均。ρ是预期调整参数。")

    st.markdown("### 4. 泰勒规则:")
    st.latex(r'i = r^* + \phi_\pi(\pi - 0.02) + \phi_y\frac{Y - Y_n}{Y_n}')

    st.markdown("这是一个货币政策规则,描述了中央银行如何设定利率。i是名义利率,r*是自然利率,φ_π和φ_y分别是对通胀和产出缺口的反应系数。")

    st.markdown("### 5. 长期增长方程:")
    st.latex(r'Y_{n,new} = Y_n(1 + g)')

    st.markdown("这个方程描述了潜在产出的长期增长,其中g是长期增长率。")

    st.markdown("### 6. 奥肯法则:")
    st.latex(r'u = u_n - \alpha\frac{Y - Y_n}{Y_n}')

    st.markdown("这个法则描述了失业率(u)与产出缺口之间的关系。u_n是自然失业率,α是奥肯系数。")

    st.markdown("### 7. 菲利普斯曲线:")
    st.latex(r'\pi = \pi^e + \beta\frac{Y - Y_n}{Y_n}')

    st.markdown("这个曲线描述了通胀率与产出缺口之间的关系。β是菲利普斯曲线的斜率。")

    st.markdown("""
    这些模型共同构成了一个动态的宏观经济系统,能够模拟经济对各种政策和冲击的反应。模型考虑了产出、利率、汇率、价格水平、通胀预期、失业率等关键经济变量之间的相互作用。

    ### 主要特点:
    - 考虑了开放经济（包含汇率和国际贸易）
    - 包含价格粘性和通胀预期的动态调整
    - 纳入了货币政策规则（泰勒规则）
    - 考虑了长期经济增长
    - 包含了劳动市场（通过奥肯法则和菲利普斯曲线）

    通过调整不同的参数,你可以观察经济如何对不同的政策和冲击做出反应。这有助于理解宏观经济政策的效果和经济系统的动态特性。
    """)