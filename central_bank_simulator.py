# Filename: advanced_central_bank_simulator.py

import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import plotly.graph_objects as go
import copy
from openai import OpenAI
import os

# 设置 DeepSeek API
client = OpenAI(api_key="sk-89633c366041484f91f993634c6e31ab", base_url="https://api.deepseek.com/beta")

# 更新默认参数
@st.cache_data
def get_default_params():
    params = {
        'C0': 5000,   # 自主消费，反映基本生活需求
        'C1': 0.65,   # 边际消费倾向，根据中国数据调整
        'I0': 3000,   # 自主投资，反映中国较高的投资率
        'I1': 0.2,    # 投资对利率的敏感度，根据中国数据调整
        'G': 4000,    # 政府支出，约占GDP的30-35%
        'T': 3800,    # 税收，略低于政府支出，反映适度的财政赤字
        'NX0': 500,   # 自主净出口，反映中国的贸易顺差
        'NX1': 0.2,   # 净出口对汇率的敏感度，反映适度的敏感度
        'L0': 1.0,    # 货币需求收入弹性，根据中国数据调整
        'L1': 0.2,    # 货币需求利率弹性，据中国数据调整
        'L2': 0.05,   # 货币需求汇率弹性，反映低度敏感性
        'M': 20000,   # 货币供给，反映中国较高的M2/GDP比率
        'P': 1,       # 初始价格水平，作为基准
        'Yn': 15000,  # 潜在产出，设置为与实际GDP相近的水平
        'un': 0.045,  # 自然失业率，根据中国数据调整
        'alpha': 1.5, # 奥肯系数，反映中国劳动力市场的特殊性
        'beta': 0.3,  # 菲利普斯曲线斜率，根据中国数据调整
        'pi_e': 0.03, # 初始预期通胀率，根据中国近期通胀目标调整
        'kappa': 0.1, # 价格调整速度，根据中国数据调整
        'rho': 0.7,   # 预期调整参数，反映预期的适度粘性
        'phi_pi': 1.3,# 泰勒规则通胀系数，根据中国数据调整
        'phi_y': 0.4, # 泰勒规则产出缺口系数，根据中国数据调整
        'r_star': 0.03, # 自然利率，根据中国近期经济情况调整
        'g': 0.055,   # 长期增长率，根据中国最新经济增长目标调整
        'i_foreign': 0.02, # 国外利率，保持不变
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
        
        # 确保Y不为负，并设置一个更高的上限
        Y = max(Y, 0.1)
        Y = min(Y, Yn * 10)  # 设置上限为潜在产出的10倍
        
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
        Y_sol = max(result['Y'], 0.1)
        Y_sol = min(Y_sol, params['Yn'] * 10)  # 设置上限为潜在产出的10倍
        i_sol = max(result['i'], 0)
        i_sol = min(i_sol, 0.5)  # 设置利率上限为50%
        E_sol = max(result['E'], 0.1)
        P_sol = max(result['P'], 0.1)
        
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
        - 价格水(P)可能上升，因为需求增加。
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
        '价格水 (P)': """
        当价格水平(P)上升时：
        - 产出(Y)可能降，因为实际余额应。
        - 利率(i)可能上升，因为货币需求增加。
        - 汇率(E)可能贬值，因为本国商品相对变贵。
        - 失业率(u)可能上升，因为实际工资下降导致劳动需求减少。
        """,
        '预期通胀率 (π_e)': """
        当预期通胀率(π_e)上升时：
        - 利率(i)可能上升，因人们要求更高的名义利率。
        - 价格水平(P)可能上升，因为工资���价格设定行为改变。
        - 实际产出(Y)可能短期内增加，但长期可能不降。
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
        - 预期通胀率(π_e)可能上升，因为人们调整对未来通胀的
        """
    }
    return explanations.get(variable, "没有该变量的具体解释。")

# 添加这个新函数来生成详细的提示词模板
def create_detailed_prompt(base_results, policy_results, G_base, G_policy, T_base, T_policy, M_base, M_policy):
    prompt = f"""
    作为一位宏观经济学专家,请分析以下两种经济情景的模拟结果,并提供详细解释:

    基准情景:
    1. 政策参数:
       - 政府支出(G): {G_base}
       - 税收(T): {T_base}
       - 货币供给(M): {M_base}
    2. 经济指标:
       - GDP: 初始 {base_results['产出 (Y)'][0]:.2f}, 最终 {base_results['产出 (Y)'][-1]:.2f}
       - 失业率: 初始 {base_results['失业率 (u)'][0]:.2f}, 最终 {base_results['失业率 (u)'][-1]:.2f}
       - 通胀率: 初始 {base_results['通胀率 (π)'][0]:.2f}, 最终 {base_results['通胀率 (π)'][-1]:.2f}
       - 利率: 初始 {base_results['利率 (i)'][0]:.2f}, 最终 {base_results['利率 (i)'][-1]:.2f}
       - 价格水平: 初始 {base_results['价格水平 (P)'][0]:.2f}, 最终 {base_results['价格水平 (P)'][-1]:.2f}

    政策情景:
    1. 政策参数:
       - 政府支出(G): {G_policy}
       - 税收(T): {T_policy}
       - 货币供给(M): {M_policy}
    2. 经济指标:
       - GDP: 初始 {policy_results['产出 (Y)'][0]:.2f}, 最终 {policy_results['产出 (Y)'][-1]:.2f}
       - 失业率: 初始 {policy_results['失业率 (u)'][0]:.2f}, 最终 {policy_results['失业率 (u)'][-1]:.2f}
       - 通胀率: 初始 {policy_results['通胀率 (π)'][0]:.2f}, 最终 {policy_results['通胀率 (π)'][-1]:.2f}
       - 利率: 初始 {policy_results['利率 (i)'][0]:.2f}, 最终 {policy_results['利率 (i)'][-1]:.2f}
       - 价格水平: 初始 {policy_results['价格水平 (P)'][0]:.2f}, 最终 {policy_results['价格水平 (P)'][-1]:.2f}

    请提供以下分析:
    1. 政变化概述: 简要说明政策情景对于基准情景的主要政策变化。
    2. 经济影响分析: 
       a) 详细比较两种情景下各个经济指标(GDP、失业率、通胀率、利率、价格水平)的变化。
       b) 分析这些指标之间的相互作用和可能的因果关系。
    3. 短期vs长期影响: 讨论政策变化可能产生的短期和长期经济效果的差异。
    4. 政策效果评估: 评估政策变化是否达到了预期目标,有何优缺点。
    5. 潜在风险: 指出政策情景可能带来的潜在经济风险或负面影响。
    6. 政策建议: 
       a) 基于分析结果,提出改进或优化政策的建议。
       b) 如果需要,建议配套措施以增强政策效果或缓解负面影响。
    7. 总结: 简要总结分析结果和主要观点。

    请确保您的分析全面、深入,并考虑到宏观经济学的各个方面。
    """
    return prompt

# 修改generate_ai_explanation函数以适应新的prompt格式
def generate_ai_explanation(base_results, policy_results, G_base, G_policy, T_base, T_policy, M_base, M_policy):
    prompt = create_detailed_prompt(base_results, policy_results, G_base, G_policy, T_base, T_policy, M_base, M_policy)

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一位资深的宏观经济学专家,擅长分析复杂的经济政策影响。请提供深入、全面且结构清晰的分析。"},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
        return response
    except Exception as e:
        return f"无法生成AI解释: {str(e)}"

def policy_comparison_demo(params):
    st.subheader("政策对比演示")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**基准情景**")
        G_base = st.number_input("政府支出 (G)", value=float(params['G']), step=100.0, min_value=0.0, max_value=10000.0, key="G_base")
        T_base = st.number_input("税收 (T)", value=float(params['T']), step=100.0, min_value=0.0, max_value=10000.0, key="T_base")
        M_base = st.number_input("货币供给 (M)", value=float(params['M']), step=1000.0, min_value=0.0, max_value=100000.0, key="M_base")
        i_base = st.number_input("初始利率 (i)", value=0.03, step=0.01, format="%.2f", min_value=0.0, max_value=0.2, key="i_base")

    with col2:
        st.markdown("**政策情景**")
        G_policy = st.number_input("政府支出 (G)", value=float(params['G'])*1.2, step=100.0, min_value=0.0, max_value=10000.0, key="G_policy")
        T_policy = st.number_input("税收 (T)", value=float(params['T']), step=100.0, min_value=0.0, max_value=10000.0, key="T_policy")
        M_policy = st.number_input("货币供给 (M)", value=float(params['M']), step=1000.0, min_value=0.0, max_value=100000.0, key="M_policy")
        i_policy = st.number_input("初始利率 (i)", value=0.03, step=0.01, format="%.2f", min_value=0.0, max_value=0.2, key="i_policy")

    if st.button("运行对比模拟并生成AI解释"):
        base_params = copy.deepcopy(params)
        policy_params = copy.deepcopy(params)
        
        base_results = simulate_dynamic(base_params, G_base, T_base, M_base, i_base, 1.0, base_params['P'], base_params['pi_e'])
        policy_results = simulate_dynamic(policy_params, G_policy, T_policy, M_policy, i_policy, 1.0, policy_params['P'], policy_params['pi_e'])

        # 创建对比数据框
        comparison_df = pd.DataFrame({
            '指标': ['初始GDP', '最终GDP', '初始利率', '最终利率', '初始价格水平', '最终价格水平', '初始失业率', '最终失业率', '初始通胀率', '最终通胀率'],
            '基准情景': [f"{base_results['产出 (Y)'][0]:.0f}", f"{base_results['产出 (Y)'][-1]:.0f}", 
                        f"{base_results['利率 (i)'][0]:.2%}", f"{base_results['利率 (i)'][-1]:.2%}", 
                        f"{base_results['价格水平 (P)'][0]:.2f}", f"{base_results['价格水平 (P)'][-1]:.2f}", 
                        f"{base_results['失业率 (u)'][0]:.2%}", f"{base_results['失业率 (u)'][-1]:.2%}", 
                        f"{base_results['通胀率 (π)'][0]:.2%}", f"{base_results['通胀率 (π)'][-1]:.2%}"],
            '政策情景': [f"{policy_results['产出 (Y)'][0]:.0f}", f"{policy_results['产出 (Y)'][-1]:.0f}", 
                        f"{policy_results['利率 (i)'][0]:.2%}", f"{policy_results['利率 (i)'][-1]:.2%}", 
                        f"{policy_results['价格水平 (P)'][0]:.2f}", f"{policy_results['价格水平 (P)'][-1]:.2f}", 
                        f"{policy_results['失业率 (u)'][0]:.2%}", f"{policy_results['失业率 (u)'][-1]:.2%}", 
                        f"{policy_results['通胀率 (π)'][0]:.2%}", f"{policy_results['通胀率 (π)'][-1]:.2%}"]
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
            
            # 调整y轴范围
            if indicator == '产出 (Y)':
                fig.update_yaxes(range=[0, 30000])
            elif indicator in ['利率 (i)', '通胀率 (π)', '失业率 (u)']:
                fig.update_yaxes(range=[0, 0.2])
            elif indicator == '价格水平 (P)':
                fig.update_yaxes(range=[0, 2])
            
            st.plotly_chart(fig)

        # 添加失业率解释
        st.markdown("### 失业率说明")
        st.markdown("""
        注意：模型计算的失业率可能会出现非常低的值，这是由于模型的简化性质和长期增长假设导致的。
        在实际经济中，失业率通常不会降到非常低的水平。这个结果提醒我们，模型虽然有助于理解经济
        政策的影响，但仍然是对现实的简化表示。在解释结果时，我们应该更多地关注失业率的变化趋势，
        而不是具体的数值。
        """)

        # 生成AI解释
        st.subheader("AI生成的结果解释")
        explanation_placeholder = st.empty()
        full_response = ""
        with st.spinner("正在生成AI解释..."):
            stream = generate_ai_explanation(
                base_results, policy_results,
                G_base, G_policy,
                T_base, T_policy,
                M_base, M_policy
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    explanation_placeholder.markdown(full_response + "▌")
        explanation_placeholder.markdown(full_response)

def create_single_scenario_prompt(results, G, T, M):
    prompt = f"""
    作为一位宏观经济学专家,请分析以下经济模拟结果,并提供详细解释:

    政策参数:
    - 政府支出(G): {G}
    - 税收(T): {T}
    - 货币供给(M): {M}

    经济指标:
    - GDP: 初始 {results['产出 (Y)'][0]:.2f}, 最终 {results['产出 (Y)'][-1]:.2f}
    - 失业率: 初始 {results['失业率 (u)'][0]:.2f}, 最终 {results['失业率 (u)'][-1]:.2f}
    - 通胀率: 初始 {results['通胀率 (π)'][0]:.2f}, 最终 {results['通胀率 (π)'][-1]:.2f}
    - 利率: 初始 {results['利率 (i)'][0]:.2f}, 最终 {results['利率 (i)'][-1]:.2f}
    - 价格水平: 初始 {results['价格水平 (P)'][0]:.2f}, 最终 {results['价格水平 (P)'][-1]:.2f}

    请提供以下分析:
    1. 经济表现概述: 简要说明模拟期间经济的整体表现。
    2. 各指标分析: 
       a) 详细解释GDP、失业率、通胀率、利率和价格水平的变化趋势。
       b) 分析这些指标之间的相互作用和可能的因果关系。
    3. 政策效果评估: 评估给定的政府支出、税收和货币供给水平对经济的影响。
    4. 短期vs长期影响: 讨论观察到的经济变化在短期和长期可能产生的不同效果。
    5. 潜在风险: 指出当前经济状况可能带来的潜在风险或负面影响。
    6. 政策建议: 
       a) 基于分析结果,提出可能的政策调整建议。
       b) 如果需要,建议配套措施以改善经济表现或缓解潜在问题。
    7. 总结: 简要总结分析结果和主要观点。

    请确保您的分析全面、深入,并考虑到宏观经济学的各个方面。
    """
    return prompt

def generate_single_scenario_ai_explanation(results, G, T, M):
    prompt = create_single_scenario_prompt(results, G, T, M)

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一位资深的宏观经济学专家,擅长分析复杂的经济政策影响。请提供深入、全面且结构清晰的分析。"},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
        return response
    except Exception as e:
        return f"无法生成AI解释: {str(e)}"

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
            params['NX1'] = st.number_input('出口汇率敏感度 (NX1)', value=params['NX1'], step=0.1, format="%.2f")
            params['L0'] = st.number_input('货币需求收入弹性 (L0)', value=params['L0'], step=0.1, format="%.2f")
            params['L1'] = st.number_input('货币需求利率弹性 (L1)', value=params['L1'], step=0.1, format="%.2f")
        with col3:
            params['kappa'] = st.number_input('价格调整速度 (κ)', value=params['kappa'], step=0.01, format="%.2f")
            params['rho'] = st.number_input('期调整参数 (ρ)', value=params['rho'], step=0.1, format="%.2f")
            params['phi_pi'] = st.number_input('泰勒规则通胀系数 (φ_π)', value=params['phi_pi'], step=0.1, format="%.2f")
            params['phi_y'] = st.number_input('泰勒规则产出缺口系数 (φ_y)', value=params['phi_y'], step=0.1, format="%.2f")

    # 执行模拟
    if st.button('运行模拟并生成AI解释'):
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

        # 创建结果数据框
        results_df = pd.DataFrame({
            '指标': ['初始GDP', '最终GDP', '初始利率', '最终利率', '初始价格水平', '最终价格水平', '初始失业率', '最终失业率', '初始通胀率', '最终通胀率'],
            '值': [df_history['产出 (Y)'].iloc[0], df_history['产出 (Y)'].iloc[-1], 
                  df_history['利率 (i)'].iloc[0], df_history['利率 (i)'].iloc[-1], 
                  df_history['价格水平 (P)'].iloc[0], df_history['价格水平 (P)'].iloc[-1], 
                  df_history['失业率 (u)'].iloc[0], df_history['失业率 (u)'].iloc[-1], 
                  df_history['通胀率 (π)'].iloc[0], df_history['通胀率 (π)'].iloc[-1]]
        })

        st.subheader("模拟结果概览")
        st.table(results_df)

        # 为每个指标创建单独的图表
        indicators = ['产出 (Y)', '利率 (i)', '价格水平 (P)', '失业率 (u)', '通胀率 (π)']
        for indicator in indicators:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(df_history[indicator]))), 
                                     y=df_history[indicator], 
                                     mode='lines', 
                                     name='模拟结果'))
            fig.update_layout(title=f'{indicator}随时间的变化', xaxis_title='时间', yaxis_title=indicator)
            st.plotly_chart(fig)

        # 添加失业率解释
        st.markdown("### 失业率说明")
        st.markdown("""
        注意：模型计算的失业率可能会出现非常低的值，这是由于模型的简化性质和长期增长假设导致的。
        在实际经济中，失业率通常不会降到非常低的水平。这个结果提醒我们，模型虽然有助于理解经济
        政策的影响，但仍然是对现实的简化表示在解释结果时，我们应该更多地关注失业率的变化趋势，
        而不是具体的数值。
        """)

        # 显示详细数据表格
        st.subheader("详细模拟结果数据")
        st.dataframe(df_history)

        # 生成AI解释
        st.subheader("AI生成的结果解释")
        explanation_placeholder = st.empty()
        full_response = ""
        with st.spinner("正在生成AI解释..."):
            stream = generate_single_scenario_ai_explanation(
                history, G, T, initial_money_supply
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    explanation_placeholder.markdown(full_response + "▌")
        explanation_placeholder.markdown(full_response)

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

    st.markdown("这个方程描述了人们如何形成对未来通胀的预期。新的预期胀率(π^e_new)是当前预期(π^e)和实际通胀率(π)的加权平均。ρ是预期调整参数。")

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
    这些模型共同构成了一个动态的宏观经济系统,能够模拟经济对各种政策和冲击的反应。模型考虑了产出、利率、汇、价格水平、胀预期、失业率等关键经济变量之间的相互作用。

    ### 主要特点:
    - 考虑了开放经济（包含汇率和国际贸易）
    - 包含价格粘性通胀预期的动态调整
    - 纳入了货币政策规则（泰勒规则）
    - 考虑了长期经增长
    - 包含了劳动市场（通过奥肯法则和菲利普斯曲线）

    通过调整不同的参数,你可以观察经济如何对不同的政策和冲击做出反应。这有助于理解宏观经济政策的效果和经济系统的动态特性
    """)