# -*- coding: utf-8 -*-
# @Time   : 2023/4/19 18:10

from pulp import *
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

'''
define:
P_gas: 燃机的发电功率
P_gshp: 地源热泵的制冷功率
P_grid: 电网购电功率

objective function:
min(Cost_grid * P_grid + Cost_grid * P_gshp / COP + Cost_gas * 3600 * P_gas / P_eff / cal_value)

s.t.

P_gas + P_grid - P_gshp / COP = P_load  # 电平衡约束
Cool_load = P_gshp  # 冷平衡约束
P_gas / P_eff >= 0
P_gas / P_eff <= P_gas_max
P_gshp >= 0
P_gshp <= P_gshp_max
'''


def LP_mode(hour, P_load, Cool_load, Cost_grid):
    # op = ['P_gas', 'P_gshp', 'P_grid']
    # 创建问题实例，求最小极值
    prob = LpProblem("The CCHP Problem", LpMinimize)

    # 定义变量
    P_gas = LpVariable(name='P_gas', lowBound=0, upBound=330 * 0.4, cat=const.LpContinuous)
    P_gshp = LpVariable(name='P_gshp', lowBound=0, upBound=3 * 444, cat=const.LpContinuous)
    P_grid = LpVariable(name='P_grid', lowBound=0, cat=const.LpContinuous)

    # 目标函数
    prob += Cost_grid * P_grid + Cost_grid * P_gshp / 4.82 + 2.31 * 3600 * P_gas / 0.4 / 35588

    # 约束条件
    prob += P_gas + P_grid - P_gshp * (1 / 4.82) == P_load
    prob += P_gshp == Cool_load
    prob += Cool_load <= 3 * 444

    # 保存LP的信息
    # prob.writeLP('D:\zzxwork\cchp_demo\cchp.lp')

    # 求解
    prob.solve()

    # 查看解的状态
    var_dic = dict()
    var_dic['hour'] = hour
    var_dic['status'] = LpStatus[prob.status]

    # 查看各决策变量的取值
    for v in prob.variables():
        var_dic[v.name] = v.varValue

    var_dic['lowest_cost'] = value(prob.objective)
    return var_dic


def data_process():
    df = pd.read_excel('D:/zzxwork/cchp_demo/冬季运行策略图.xlsx', sheet_name='基础负荷')
    df['Cost_grid'] = [0.3023, 0.3023, 0.3023, 0.3023,
                       0.3023, 0.3023, 0.7697, 0.7697,
                       0.7697, 1.2884, 1.2884, 1.2884,
                       1.2884, 1.2884, 0.7697, 0.7697,
                       0.7697, 1.2884, 1.2884, 1.2884,
                       0.7697, 0.7697, 0.3023, 0.3023]
    df_new = df[['夏季典型日电负荷', '夏季典型日冷负荷', 'Cost_grid']]
    df_new.columns = ['summer_p', 'summer_c', 'Cost_grid']
    df_new.to_excel('D:/zzxwork/cchp_demo/data.xlsx')


def save_result():
    all_res.to_excel('D:/zzxwork/cchp_demo/output.xlsx')




if __name__ == '__main__':
    # data_process()
    df = pd.read_excel('D:/zzxwork/cchp_demo/data.xlsx')
    # x = df.loc[:, ['夏季典型日电负荷', '夏季典型日冷负荷', 'Cost_grid']]

    hour_list, s_list, p_gshp_list, p_gas_list, p_grid_list, l_c_list = [], [], [], [], [], []

    for hour, row in enumerate(df.itertuples()):
        res = LP_mode(hour, getattr(row, 'summer_p'), getattr(row, 'summer_c'), getattr(row, 'Cost_grid'))

        hour_list.append(res['hour'])
        s_list.append(res['status'])
        p_gshp_list.append(res['P_gshp'])
        p_gas_list.append(res['P_gas'])
        p_grid_list.append(res['P_grid'])
        l_c_list.append(res['lowest_cost'])
    all_res = pd.DataFrame({'hour': hour_list, 'status': s_list, 'P_gshp': p_gshp_list, 'P_gas': p_gas_list,
               'P_grid': p_grid_list, 'lowest_cost': l_c_list})
    print(all_res)
    # save_result()

















