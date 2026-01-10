import streamlit as st
import pandas as pd
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpInteger, PULP_CBC_CMD
import numpy as np
import matplotlib.pyplot as plt
import platform
import japanize_matplotlib # 日本語フォント対応

# ----------------------------------------
# 0. Streamlitページ設定
# ----------------------------------------
st.set_page_config(layout="wide", page_title="食品ロス削減シミュレーション")

# ----------------------------------------
# 1. シミュレーションクラス（パラメータ受取を強化）
# ----------------------------------------
class FoodSupplySimulation:
    def __init__(self, strategy='LP', random_seed=42, 
                 demand_std_scale=1.0, waste_risk_weight=0.1,
                 shelf_life_dict=None, supply_mean=35):
        
        self.strategy = strategy
        self.shops = ['A店', 'B店', 'C店', 'D店']
        self.items = ['トマト', '牛乳', 'パン']
        self.rng = np.random.default_rng(random_seed)
        self.current_stock = pd.DataFrame(
            columns=['stock_id', 'item', 'stock_quantity', 'remaining_shelf_life']
        )
        self.next_stock_id = 1
        self.total_waste = 0
        self.total_sales = 0
        self.WEEKLY_DEMAND_PATTERN = [1.0, 0.9, 0.9, 1.0, 1.2, 1.4, 1.3]
        
        # --- 外部から受け取るパラメータ ---
        self.demand_std_scale = demand_std_scale
        self.waste_risk_weight = waste_risk_weight
        # 賞味期限設定（デフォルト値を設定）
        self.shelf_life_dict = shelf_life_dict if shelf_life_dict else {'トマト': 5, '牛乳': 7, 'パン': 4}
        # 入荷量の平均値
        self.supply_mean = supply_mean

    def get_min_shelf_life(self, shop):
        # コンビニ(AB)は鮮度厳守、スーパー(CD)は少し古くても置くイメージ
        if shop in ['A店', 'B店']: return 3
        return 1

    def add_stock(self, day):
        # 日曜日は入荷なし
        if (day - 1) % 7 == 6: return 
        
        for item in self.items:
            # ★入荷量をコントロールパネルの値(supply_mean)に基づいて決定
            qty = max(0, int(self.rng.normal(self.supply_mean, 10)))
            
            # ★賞味期限をコントロールパネルの値から取得
            full_life = self.shelf_life_dict[item]
            
            # 入荷時点で少し時間が経過しているラグを表現
            delay = int(self.rng.exponential(1.2))
            life = max(1, full_life - delay)
            
            if qty > 0:
                self.current_stock = pd.concat([
                    self.current_stock,
                    pd.DataFrame([{
                        'stock_id': self.next_stock_id,
                        'item': item,
                        'stock_quantity': qty,
                        'remaining_shelf_life': life
                    }])
                ], ignore_index=True)
                self.next_stock_id += 1

    def generate_demand(self, day):
        weekday = (day - 1) % 7
        factor = self.WEEKLY_DEMAND_PATTERN[weekday]
        rows = []
        for shop in self.shops:
            scale = {'A店': 1.5, 'B店': 1.0, 'C店': 0.6, 'D店': 0.8}[shop]
            for item in self.items:
                base = {'トマト': 8, '牛乳': 6, 'パン': 8}[item]
                std_dev = 4 * self.demand_std_scale 
                qty = max(0, int(self.rng.normal(base * scale * factor, std_dev)))
                if qty > 0:
                    rows.append({'retail_store': shop, 'item': item, 'demand_quantity': qty})
        return pd.DataFrame(rows)

    def solve_lp(self, demand):
        stock = self.current_stock.copy()
        stock['risk'] = 1 / (stock['remaining_shelf_life'] + self.waste_risk_weight)
        
        prob = LpProblem("LP", LpMinimize)
        x = {}
        for i in stock.index:
            for j in demand.index:
                if stock.at[i, 'item'] != demand.at[j, 'item']: continue
                if stock.at[i, 'remaining_shelf_life'] < self.get_min_shelf_life(demand.at[j, 'retail_store']): continue
                x[i, j] = LpVariable(f"x_{i}_{j}", 0, None, LpInteger)
        
        # 目的関数：廃棄リスクが高いものを優先して出荷
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

    def step(self, day):
        self.add_stock(day)
        expired = self.current_stock['remaining_shelf_life'] <= 0
        waste_today = self.current_stock.loc[expired, 'stock_quantity'].sum()
        self.total_waste += waste_today
        self.current_stock = self.current_stock[~expired]
        demand = self.generate_demand(day)
        shipment = {}
        if self.strategy == 'LP':
            shipment = self.solve_lp(demand)
        elif self.strategy == 'FIFO':
            shipment = self.solve_fifo(demand, random=False)
        elif self.strategy == 'Random':
            shipment = self.solve_fifo(demand, random=True)
        shipped_today = 0
        for (i, j), qty in shipment.items():
            self.current_stock.at[i, 'stock_quantity'] -= qty
            shipped_today += qty
        self.total_sales += shipped_today
        self.current_stock['remaining_shelf_life'] -= 1
        self.current_stock = self.current_stock[self.current_stock['stock_quantity'] > 0]
        return self.total_waste, self.total_sales

# ----------------------------------------
# 2. Streamlit UI構築
# ----------------------------------------
def main():
    st.title("🍎 食品サプライチェーン最適化シミュレーター")
    
    # --- サイドバー：コントロールパネル ---
    st.sidebar.header("🛠 コントロールパネル")
    
    # グループ1: 基本設定
    with st.sidebar.expander("① 基本設定", expanded=True):
        simulation_days = st.slider("シミュレーション日数", 10, 100, 30, 5)
        random_seed = st.number_input("乱数シード (結果の固定)", value=42)

    # グループ2: 商品パラメータ（ここを追加）
    with st.sidebar.expander("② 商品設定 (賞味期限)", expanded=True):
        st.caption("各商品の最大賞味期限(日)を設定します")
        col_p1, col_p2, col_p3 = st.columns(3)
        life_tomato = col_p1.number_input("トマト", 3, 10, 5)
        life_milk = col_p2.number_input("牛乳", 3, 15, 7)
        life_bread = col_p3.number_input("パン", 2, 8, 4)
        
        shelf_life_dict = {'トマト': life_tomato, '牛乳': life_milk, 'パン': life_bread}

    # グループ3: 需給バランス（ここを追加）
    with st.sidebar.expander("③ 需給バランス調整", expanded=True):
        supply_mean = st.slider("1回あたりの平均入荷数", 
                                min_value=20, max_value=60, value=35, 
                                help="数値を大きくすると「作りすぎ」の状態になります。")
        
        demand_std_scale = st.slider("需要のばらつき倍率", 
                                     0.0, 3.0, 1.0, 
                                     help="1.0が通常。大きくすると客足が予測不能になります。")

    # グループ4: アルゴリズム設定
    with st.sidebar.expander("④ LPアルゴリズム詳細"):
        waste_risk_weight = st.slider("リスク感度パラメータ", 
                                      0.01, 1.0, 0.1, 
                                      help="小さいほど、賞味期限切れ間近の商品を優先的に出荷します。")

    run_button = st.sidebar.button("シミュレーション実行", type="primary")

    # --- メイン処理 ---
    if run_button:
        strategies = ['Random', 'FIFO', 'LP']
        colors = {'Random': 'gray', 'FIFO': 'blue', 'LP': 'red'}
        results = {s: {'days': [], 'waste': [], 'sales': []} for s in strategies}

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, s in enumerate(strategies):
            status_text.text(f"戦略 {s} を計算中...")
            # GUIからのパラメータを全て渡す
            sim = FoodSupplySimulation(
                strategy=s, 
                random_seed=random_seed,
                demand_std_scale=demand_std_scale,
                waste_risk_weight=waste_risk_weight,
                shelf_life_dict=shelf_life_dict, # 新しい設定
                supply_mean=supply_mean          # 新しい設定
            )
            for day in range(1, simulation_days + 1):
                waste, sales = sim.step(day)
                results[s]['days'].append(day)
                results[s]['waste'].append(waste)
                results[s]['sales'].append(sales)
            progress_bar.progress((i + 1) / len(strategies))
        
        status_text.text("完了！")
        progress_bar.empty()

        # --- 集計データの作成 ---
        summary_data = []
        for s in strategies:
            final_waste = results[s]['waste'][-1]
            final_sales = results[s]['sales'][-1]
            total_items = final_waste + final_sales
            waste_rate = (final_waste / total_items * 100) if total_items > 0 else 0
            
            base_waste = results['Random']['waste'][-1]
            improvement = 0
            if s != 'Random':
                improvement = (base_waste - final_waste) / base_waste * 100

            summary_data.append([
                s, 
                int(final_sales), 
                int(final_waste), 
                f"{waste_rate:.1f}%", 
                f"▲{improvement:.1f}%" if improvement > 0 else "-"
            ])
        df_summary = pd.DataFrame(summary_data, columns=['戦略', '累積売上', '累積廃棄', '廃棄率', '削減率'])

        # --- 結果表示 ---
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📊 集計結果")
            st.table(df_summary)
            
            best_strat = df_summary.iloc[df_summary['累積廃棄'].idxmin()]['戦略']
            st.info(f"最も廃棄が少なかった戦略: **{best_strat}**")
            
            # 実験のヒントを表示
            if supply_mean > 45:
                st.warning("⚠️ 入荷量が多すぎます。どの戦略でも廃棄が増える傾向にあります。")
            elif supply_mean < 25:
                st.warning("⚠️ 入荷量が少なすぎます。廃棄は減りますが、売上機会を逃しています。")

        with col2:
            st.subheader("📈 推移グラフ")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
            plt.subplots_adjust(hspace=0.3)

            # 1. 累積廃棄グラフ
            for s in strategies:
                ax1.plot(results[s]['days'], results[s]['waste'], 
                         label=s, color=colors[s], marker='o', markersize=4)
            ax1.set_title("累積フードロス発生量 (低いほど良い)")
            ax1.set_ylabel("累積廃棄数 (個)")
            ax1.grid(True, linestyle='--', alpha=0.6)
            ax1.legend()

            # 2. 累積売上グラフ
            for s in strategies:
                ax2.plot(results[s]['days'], results[s]['sales'], 
                         label=s, color=colors[s], linestyle='--')
            ax2.set_title("累積販売数 (高いほど良い)")
            ax2.set_xlabel("経過日数")
            ax2.set_ylabel("累積販売数 (個)")
            ax2.grid(True, linestyle='--', alpha=0.6)
            ax2.legend()
            
            st.pyplot(fig)

if __name__ == "__main__":
    main()