import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import urllib.request
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpInteger, PULP_CBC_CMD

# 日本語フォント設定（Streamlit Cloud用）
def setup_japanese_font():
    url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/Japanese/NotoSansCJKjp-Regular.otf"
    save_path = "NotoSansCJKjp-Regular.otf"

    # ファイルがなければDL
    if not os.path.exists(save_path):
        urllib.request.urlretrieve(url, save_path)

    fm.fontManager.addfont(save_path)
    plt.rcParams['font.family'] = 'Noto Sans CJK JP'

setup_japanese_font()

# ページ設定
st.set_page_config(layout="wide", page_title="食品ロス削減シミュレーション")

# シミュレーションモデル
class FoodSupplySimulation:
    def __init__(self, strategy='LP', random_seed=42, 
                 demand_std_scale=1.0, waste_risk_weight=0.1,
                 shelf_life_dict=None, supply_mean=35):
        
        self.strategy = strategy
        # 対象店舗
        self.shops = ['大学会館店', 'つくば駅前店', 'ひたち野牛久店', '研究学園店']
        self.items = ['トマト', '牛乳', 'パン']
        self.rng = np.random.default_rng(random_seed)
        
        # 在庫データ
        self.current_stock = pd.DataFrame(columns=['stock_id', 'item', 'stock_quantity', 'remaining_shelf_life'])
        self.next_stock_id = 1
        
        # 集計用
        self.total_waste = 0
        self.total_sales = 0
        
        # 曜日変動（土日は多め）
        self.WEEKLY_DEMAND_PATTERN = [1.0, 0.9, 0.9, 1.0, 1.2, 1.4, 1.3]
        
        # パラメータ
        self.demand_std_scale = demand_std_scale
        self.waste_risk_weight = waste_risk_weight
        self.shelf_life_dict = shelf_life_dict if shelf_life_dict else {'トマト': 5, '牛乳': 7, 'パン': 4}
        self.supply_mean = supply_mean

    # 納品期限（コンビニは厳しい）
    def get_min_shelf_life(self, shop):
        if shop in ['大学会館店', 'つくば駅前店']: return 3
        return 1

    # 入荷処理
    def add_stock(self, day):
        # 日曜は入荷なし
        if (day - 1) % 7 == 6: return 
        
        for item in self.items:
            qty = max(0, int(self.rng.normal(self.supply_mean, 10)))
            
            # 入荷ラグを考慮
            full_life = self.shelf_life_dict[item]
            delay = int(self.rng.exponential(1.2))
            life = max(1, full_life - delay)
            
            if qty > 0:
                new_stock = pd.DataFrame([{
                    'stock_id': self.next_stock_id,
                    'item': item,
                    'stock_quantity': qty,
                    'remaining_shelf_life': life
                }])
                self.current_stock = pd.concat([self.current_stock, new_stock], ignore_index=True)
                self.next_stock_id += 1

    # 需要生成
    def generate_demand(self, day):
        weekday = (day - 1) % 7
        factor = self.WEEKLY_DEMAND_PATTERN[weekday]
        rows = []
        
        for shop in self.shops:
            scale = {'大学会館店': 1.5, 'つくば駅前店': 1.0, 'ひたち野牛久店': 0.6, '研究学園店': 0.8}[shop]
            
            for item in self.items:
                base = {'トマト': 8, '牛乳': 6, 'パン': 8}[item]
                std_dev = 4 * self.demand_std_scale
                
                qty = max(0, int(self.rng.normal(base * scale * factor, std_dev)))
                
                if qty > 0:
                    rows.append({'retail_store': shop, 'item': item, 'demand_quantity': qty})
        return pd.DataFrame(rows)

    # 数理最適化(LP)
    def solve_lp(self, demand):
        stock = self.current_stock.copy()
        
        # 賞味期限が近いものを優先するための重み
        stock['risk'] = 1 / (stock['remaining_shelf_life'] + self.waste_risk_weight)
        
        prob = LpProblem("Distribution_Optimization", LpMinimize)
        x = {}

        for i in stock.index:
            for j in demand.index:
                if stock.at[i, 'item'] != demand.at[j, 'item']: continue
                if stock.at[i, 'remaining_shelf_life'] < self.get_min_shelf_life(demand.at[j, 'retail_store']): continue
                
                x[i, j] = LpVariable(f"x_{i}_{j}", 0, None, LpInteger)
        
        # 目的関数
        prob += lpSum(
            (stock.at[i, 'stock_quantity'] - lpSum(x[i, j] for j in demand.index if (i, j) in x))
            * stock.at[i, 'risk']
            for i in stock.index
        )
        
        # 制約条件
        for i in stock.index:
            prob += lpSum(x[i, j] for j in demand.index if (i, j) in x) <= stock.at[i, 'stock_quantity']
        for j in demand.index:
            prob += lpSum(x[i, j] for i in stock.index if (i, j) in x) <= demand.at[j, 'demand_quantity']
            
        prob.solve(PULP_CBC_CMD(msg=0))
        return {(i, j): v.value() for (i, j), v in x.items() if v.value() > 0}

    # FIFO / Random
    def solve_fifo(self, demand, random=False):
        shipment = {}
        stock = self.current_stock.copy()
        
        demand_idx = demand.index.tolist()
        self.rng.shuffle(demand_idx)
        
        for item in self.items:
            stock_idx = stock[stock['item'] == item].index.tolist()
            
            if random:
                self.rng.shuffle(stock_idx)
            else:
                stock_idx.sort(key=lambda i: stock.at[i, 'remaining_shelf_life'])
            
            for j in demand_idx:
                if demand.at[j, 'item'] != item: continue
                need = demand.at[j, 'demand_quantity']
                min_life = self.get_min_shelf_life(demand.at[j, 'retail_store'])
                
                for i in stock_idx:
                    if need <= 0: break
                    if stock.at[i, 'stock_quantity'] <= 0: continue
                    if stock.at[i, 'remaining_shelf_life'] < min_life: continue
                    
                    amount = min(need, stock.at[i, 'stock_quantity'])
                    shipment[i, j] = shipment.get((i, j), 0) + amount
                    stock.at[i, 'stock_quantity'] -= amount
                    need -= amount
        return shipment

    # 1日分のステップ実行
    def step(self, day):
        self.add_stock(day)
        
        # 廃棄
        expired = self.current_stock['remaining_shelf_life'] <= 0
        waste_today = self.current_stock.loc[expired, 'stock_quantity'].sum()
        self.total_waste += waste_today
        self.current_stock = self.current_stock[~expired]
        
        # 需要と配送
        demand = self.generate_demand(day)
        shipment = {}
        
        if self.strategy == 'LP':
            shipment = self.solve_lp(demand)
        elif self.strategy == 'FIFO':
            shipment = self.solve_fifo(demand, random=False)
        elif self.strategy == 'Random':
            shipment = self.solve_fifo(demand, random=True)
            
        # 更新
        shipped_today = 0
        for (i, j), qty in shipment.items():
            self.current_stock.at[i, 'stock_quantity'] -= qty
            shipped_today += qty
            
        self.total_sales += shipped_today
        self.current_stock['remaining_shelf_life'] -= 1
        self.current_stock = self.current_stock[self.current_stock['stock_quantity'] > 0]
        
        return self.total_waste, self.total_sales

# メイン処理
def main():
    st.title("社会工学類 課題制作：食品ロス削減シミュレーション")
    st.markdown("スーパーマーケットにおける配送戦略（ランダム・FIFO・LP最適化）の違いが、食品ロス削減に与える影響を比較分析する。")
    
    # サイドバー（パラメータ）
    st.sidebar.header("シミュレーション条件設定")
    
    with st.sidebar.expander("基本条件", expanded=True):
        simulation_days = st.slider("期間 (日)", 10, 100, 30, 5)
        random_seed = st.number_input("乱数シード", value=42)

    with st.sidebar.expander("商品設定 (最大賞味期限)", expanded=True):
        col1, col2, col3 = st.columns(3)
        life_dict = {
            'トマト': col1.number_input("トマト", 3, 10, 5),
            '牛乳':   col2.number_input("牛乳", 3, 15, 7),
            'パン':   col3.number_input("パン", 2, 8, 4)
        }

    with st.sidebar.expander("需給バランス", expanded=True):
        supply_mean = st.slider("平均入荷数", 20, 60, 35)
        demand_std = st.slider("需要のばらつき倍率", 0.0, 3.0, 1.0)

    with st.sidebar.expander("アルゴリズム設定"):
        risk_weight = st.slider("LPリスク感度係数", 0.01, 1.0, 0.1)

    # 実行
    if st.sidebar.button("シミュレーション開始", type="primary"):
        strategies = ['Random', 'FIFO', 'LP']
        colors = {'Random': 'gray', 'FIFO': 'blue', 'LP': 'red'}
        results = {s: {'days': [], 'waste': [], 'sales': []} for s in strategies}

        progress = st.progress(0)
        status = st.empty()

        for i, s in enumerate(strategies):
            status.text(f"計算中... {s}")
            sim = FoodSupplySimulation(
                strategy=s, 
                random_seed=random_seed,
                demand_std_scale=demand_std,
                waste_risk_weight=risk_weight,
                shelf_life_dict=life_dict,
                supply_mean=supply_mean
            )
            for day in range(1, simulation_days + 1):
                waste, sales = sim.step(day)
                results[s]['days'].append(day)
                results[s]['waste'].append(waste)
                results[s]['sales'].append(sales)
            progress.progress((i + 1) / len(strategies))
        
        status.text("完了")
        progress.empty()

        # 集計
        summary = []
        base_waste = results['Random']['waste'][-1]
        
        for s in strategies:
            waste = results[s]['waste'][-1]
            sales = results[s]['sales'][-1]
            total = waste + sales
            
            rate = (waste / total * 100) if total > 0 else 0
            improv = (base_waste - waste) / base_waste * 100 if s != 'Random' else 0

            summary.append([
                s, int(sales), int(waste), 
                f"{rate:.1f}%", 
                f"▲{improv:.1f}%" if improv > 0 else "-"
            ])
            
        df_sum = pd.DataFrame(summary, columns=['戦略', '累積売上', '累積廃棄', '廃棄率', '削減率'])

        col_L, col_R = st.columns([1, 2])
        
        with col_L:
            st.subheader("集計結果")
            st.table(df_sum)
            best = df_sum.iloc[df_sum['累積廃棄'].idxmin()]['戦略']
            st.info(f"最も効果的な戦略: **{best}**")
            
            if supply_mean > 45: st.warning("考察: 入荷過多の傾向あり")
            elif supply_mean < 25: st.warning("考察: 機会損失の可能性あり")

        with col_R:
            st.subheader("推移グラフ")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
            plt.subplots_adjust(hspace=0.3)

            for s in strategies:
                ax1.plot(results[s]['days'], results[s]['waste'], label=s, color=colors[s], marker='o', markersize=4)
                ax2.plot(results[s]['days'], results[s]['sales'], label=s, color=colors[s], linestyle='--')
            
            ax1.set_title("累積廃棄数 (低いほど良い)")
            ax1.set_ylabel("個数")
            ax1.grid(True, linestyle='--', alpha=0.6)
            ax1.legend()

            ax2.set_title("累積販売数 (高いほど良い)")
            ax2.set_xlabel("日数")
            ax2.set_ylabel("個数")
            ax2.grid(True, linestyle='--', alpha=0.6)
            ax2.legend()
            
            st.pyplot(fig)

if __name__ == "__main__":
    main()
