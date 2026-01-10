import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import urllib.request
from pulp import LpProblem, LpVariable, LpMinimize, LpMaximize, lpSum, LpInteger, PULP_CBC_CMD

# ---------------------------------------------------------
# 1. フォント設定
# ---------------------------------------------------------
def setup_japanese_font():
    url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/Japanese/NotoSansCJKjp-Regular.otf"
    save_path = "NotoSansCJKjp-Regular.otf"
    if not os.path.exists(save_path):
        urllib.request.urlretrieve(url, save_path)
    fm.fontManager.addfont(save_path)
    plt.rcParams['font.family'] = 'Noto Sans CJK JP'

setup_japanese_font()

# ---------------------------------------------------------
# 2. アプリ設定
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="食品サプライチェーン経営シミュレーター")

# ---------------------------------------------------------
# 3. シミュレーションモデル (価格弾力性 対応版)
# ---------------------------------------------------------
class RealWorldSupplySimulation:
    def __init__(self, 
                 strategy, 
                 shop_config_df,      
                 item_config_df,      
                 random_seed=42, 
                 demand_std_scale=1.0, 
                 transport_threshold=5,
                 transport_cost_unit=10):
        
        self.strategy = strategy
        self.rng = np.random.default_rng(random_seed)
        
        # 1. 店舗情報
        self.shops = shop_config_df['店舗名'].tolist()
        self.shop_scales = dict(zip(shop_config_df['店舗名'], shop_config_df['規模倍率']))

        # 2. 商品情報 (弾力性パラメータを追加)
        self.items = item_config_df['商品名'].tolist()
        self.item_props = {}
        for _, row in item_config_df.iterrows():
            self.item_props[row['商品名']] = {
                'life': int(row['賞味期限(日)']),
                'base_demand': int(row['基本需要(個)']),
                'target_stock': int(row['発注基準(個)']),
                'price': int(row['販売単価(円)']),
                'base_price': int(row['基準価格(円)']),    # ★追加: 需要の基準となる価格
                'elasticity': float(row['価格弾力性']),    # ★追加: 感度
                'cost': int(row['仕入れ原価(円)']),
                'disposal': int(row['廃棄コスト(円)'])
            }

        # 在庫データ
        self.current_stock = pd.DataFrame(columns=[
            'stock_id', 'retail_store', 'item', 'stock_quantity', 'remaining_shelf_life'
        ])
        self.next_stock_id = 1
        
        # 累計KPI
        self.total_sales_amount = 0
        self.total_procurement_cost = 0
        self.total_disposal_cost = 0
        self.total_transport_cost = 0
        self.total_waste_count = 0
        
        # 日次計算用
        self.daily_procurement_cost = 0
        self.daily_sales_amount = 0
        self.daily_transport_cost = 0
        self.daily_disposal_cost = 0
        
        self.WEEKLY_DEMAND_PATTERN = [1.0, 0.9, 0.9, 1.0, 1.2, 1.4, 1.3]
        self.demand_std_scale = demand_std_scale
        
        # 転送パラメータ
        self.transport_threshold = transport_threshold
        self.transport_cost_unit = transport_cost_unit

    # ★修正: 価格弾力性を考慮した需要計算
    def get_expected_demand(self, shop, item, day):
        weekday = (day - 1) % 7
        factor = self.WEEKLY_DEMAND_PATTERN[weekday]
        
        # 1. 店舗規模 × 商品基本需要
        scale = self.shop_scales[shop]
        base_demand = self.item_props[item]['base_demand']
        
        # 2. 価格弾力性による補正
        # 需要 = 基本需要 * (設定価格 / 基準価格) ^ (-弾力性)
        current_price = self.item_props[item]['price']
        base_price = self.item_props[item]['base_price']
        elasticity = self.item_props[item]['elasticity']
        
        # ゼロ除算回避
        if base_price <= 0: base_price = 1
        
        price_ratio = current_price / base_price
        # 弾力性が反映された需要倍率
        price_factor = price_ratio ** (-elasticity)
        
        return base_demand * scale * factor * price_factor

    # ---------------------------------------------------------
    # 入荷プロセス (Inbound)
    # ---------------------------------------------------------
    def inbound_process(self, day):
        if (day - 1) % 7 == 6: return 

        new_rows = []
        for shop in self.shops:
            for item in self.items:
                # --- 追加計算: 現在の価格設定に基づく需要倍率を取得 ---
                current_price = self.item_props[item]['price']
                base_price = self.item_props[item]['base_price']
                elasticity = self.item_props[item]['elasticity']
                if base_price <= 0: base_price = 1
                price_ratio = current_price / base_price
                price_factor = price_ratio ** (-elasticity)
                # ---------------------------------------------------

                # 発注基準にも価格による需要変動(price_factor)を掛ける
                base_target = self.item_props[item]['target_stock']
                scale = self.shop_scales[shop]
                
                # ★ ここを変更: price_factor を掛ける
                target_level = base_target * scale * price_factor
                
                if self.strategy == 'Random':
                    order_qty = int(self.rng.uniform(target_level * 0.5, target_level * 1.5))
                else:
                    # 発注点方式
                    current_stock_df = self.current_stock[
                        (self.current_stock['retail_store'] == shop) & 
                        (self.current_stock['item'] == item)
                    ]
                    current_qty = current_stock_df['stock_quantity'].sum()
                    needed_qty = target_level - current_qty
                    order_qty = max(0, int(self.rng.normal(needed_qty, target_level * 0.05)))
                
                if order_qty > 0:
                    props = self.item_props[item]
                    delay = int(self.rng.exponential(1.0))
                    life = max(1, props['life'] - delay)
                    
                    new_rows.append({
                        'stock_id': self.next_stock_id,
                        'retail_store': shop,
                        'item': item,
                        'stock_quantity': order_qty,
                        'remaining_shelf_life': life
                    })
                    self.next_stock_id += 1
                    
                    cost = order_qty * props['cost']
                    self.daily_procurement_cost += cost
                    self.total_procurement_cost += cost
        
        if new_rows:
            self.current_stock = pd.concat([self.current_stock, pd.DataFrame(new_rows)], ignore_index=True)

    # ---------------------------------------------------------
    # 転送プロセス (Transshipment)
    # ---------------------------------------------------------
    def run_transshipment(self, day):
        if self.strategy in ['Random', 'FIFO']: return 0
        if self.strategy == 'LP': return self.run_lp_optimization(day)
        if self.strategy == 'New Optimization': return self.run_heuristic_optimization(day)
        return 0

    # LP転送ロジック
    def run_lp_optimization(self, day):
        transferred_count = 0
        new_transferred_stock = []
        self.current_stock.reset_index(drop=True, inplace=True)

        for item in self.items:
            balances = {}
            valid_indices = {}
            
            for shop in self.shops:
                stock_df = self.current_stock[
                    (self.current_stock['retail_store'] == shop) & 
                    (self.current_stock['item'] == item)
                ]
                current_qty = stock_df['stock_quantity'].sum()
                next_demand = self.get_expected_demand(shop, item, day + 1)
                
                valid_stock = stock_df[stock_df['remaining_shelf_life'] >= 2]
                valid_indices[shop] = valid_stock.index.tolist()
                
                balance = current_qty - next_demand
                balances[shop] = int(balance)

            senders = [s for s, b in balances.items() if b > 0]
            receivers = [r for r, b in balances.items() if b < 0]
            
            if not senders or not receivers: continue

            prob = LpProblem(f"Transshipment_{item}_{day}", LpMaximize)
            x = LpVariable.dicts("route", (senders, receivers), 0, None, LpInteger)
            
            unit_price = self.item_props[item]['price']
            # 利益最大化 (売上確保価値 - 輸送コスト)
            prob += lpSum([x[s][r] * (unit_price - self.transport_cost_unit) for s in senders for r in receivers])
            
            for s in senders:
                prob += lpSum([x[s][r] for r in receivers]) <= balances[s]
            for r in receivers:
                prob += lpSum([x[s][r] for s in senders]) <= abs(balances[r])

            prob.solve(PULP_CBC_CMD(msg=0))
            
            for s in senders:
                for r in receivers:
                    amount = x[s][r].value()
                    if amount and amount > 0:
                        amount = int(amount)
                        transferred_count += amount
                        
                        t_cost = amount * self.transport_cost_unit
                        self.daily_transport_cost += t_cost
                        self.total_transport_cost += t_cost
                        
                        remaining = amount
                        for idx in valid_indices[s]:
                            if remaining <= 0: break
                            if idx not in self.current_stock.index: continue
                            have = self.current_stock.at[idx, 'stock_quantity']
                            if have <= 0: continue
                            
                            take = min(have, remaining)
                            self.current_stock.at[idx, 'stock_quantity'] -= take
                            remaining -= take
                            
                            original_row = self.current_stock.loc[idx]
                            new_row = {
                                'stock_id': self.next_stock_id,
                                'retail_store': r,
                                'item': item,
                                'stock_quantity': take,
                                'remaining_shelf_life': original_row['remaining_shelf_life']
                            }
                            new_transferred_stock.append(new_row)
                            self.next_stock_id += 1
                            
        if new_transferred_stock:
            self.current_stock = pd.concat([self.current_stock, pd.DataFrame(new_transferred_stock)], ignore_index=True)

        return transferred_count

    # New Model転送ロジック
    def run_heuristic_optimization(self, day):
        transferred_count = 0
        new_transferred_stock = []
        self.current_stock.reset_index(drop=True, inplace=True)

        for item in self.items:
            # --- ★ 追加: コスト対効果の判定 ---
            # 輸送費が (売価 + 廃棄回避コスト) を上回るなら、転送しない方がマシ（赤字になる）
            unit_price = self.item_props[item]['price']
            disposal_cost = self.item_props[item]['disposal']
            
            # 転送による経済的価値 = 売上の確保 + 廃棄コストの回避
            economic_value = unit_price + disposal_cost
            
            if self.transport_cost_unit > economic_value:
                continue # 輸送費が高すぎて割に合わないためスキップ
            # ------------------------------------

            senders = []
            receivers = []
            
            for shop in self.shops:
                stock_df = self.current_stock[
                    (self.current_stock['retail_store'] == shop) & 
                    (self.current_stock['item'] == item)
                ]
                current_qty = stock_df['stock_quantity'].sum()
                next_demand = self.get_expected_demand(shop, item, day + 1)
                
                safety_stock = next_demand * 0.2 
                balance = current_qty - (next_demand + safety_stock)
                
                if balance > 0:
                    valid_stock = stock_df[stock_df['remaining_shelf_life'] >= 2]
                    sendable = valid_stock['stock_quantity'].sum()
                    surplus = max(0, sendable - (next_demand + safety_stock))
                    if surplus > 0:
                        senders.append({'shop': shop, 'qty': surplus, 'df_index': valid_stock.index.tolist()})
                        
                elif current_qty < next_demand:
                    shortage = next_demand - current_qty
                    urgency = shortage / (next_demand + 1)
                    receivers.append({'shop': shop, 'qty': shortage, 'urgency': urgency})

            receivers.sort(key=lambda x: x['urgency'], reverse=True)
            senders.sort(key=lambda x: x['qty'], reverse=True)
            
            for receiver in receivers:
                for sender in senders:
                    if sender['qty'] <= 0 or receiver['qty'] <= 0: continue
                    
                    amount = min(sender['qty'], receiver['qty'])
                    if amount < self.transport_threshold: continue
                    
                    transferred_count += amount
                    sender['qty'] -= amount
                    receiver['qty'] -= amount
                    
                    t_cost = amount * self.transport_cost_unit
                    self.daily_transport_cost += t_cost
                    self.total_transport_cost += t_cost
                    
                    remaining = amount
                    for idx in sender['df_index']:
                        if remaining <= 0: break
                        if idx not in self.current_stock.index: continue
                        have = self.current_stock.at[idx, 'stock_quantity']
                        if have <= 0: continue

                        take = min(have, remaining)
                        self.current_stock.at[idx, 'stock_quantity'] -= take
                        remaining -= take
                        
                        original_row = self.current_stock.loc[idx]
                        new_row = {
                            'stock_id': self.next_stock_id,
                            'retail_store': receiver['shop'],
                            'item': item,
                            'stock_quantity': take,
                            'remaining_shelf_life': original_row['remaining_shelf_life']
                        }
                        new_transferred_stock.append(new_row)
                        self.next_stock_id += 1
                            
        if new_transferred_stock:
            self.current_stock = pd.concat([self.current_stock, pd.DataFrame(new_transferred_stock)], ignore_index=True)

        return transferred_count

    def step(self, day):
        self.daily_procurement_cost = 0
        self.daily_sales_amount = 0
        self.daily_transport_cost = 0
        self.daily_disposal_cost = 0
        
        self.inbound_process(day)
        
        sold_today = 0
        demand_rows = []
        for shop in self.shops:
            for item in self.items:
                # ★ここでも価格弾力性が効いた需要を使用
                expected = self.get_expected_demand(shop, item, day)
                qty = max(0, int(self.rng.normal(expected, 4 * self.demand_std_scale)))
                if qty > 0:
                    demand_rows.append({'shop': shop, 'item': item, 'qty': qty})
        
        self.current_stock.reset_index(drop=True, inplace=True)
        
        for d in demand_rows:
            shop, item, need = d['shop'], d['item'], d['qty']
            targets = self.current_stock[
                (self.current_stock['retail_store'] == shop) & 
                (self.current_stock['item'] == item)
            ].sort_values('remaining_shelf_life')
            
            for idx, stock in targets.iterrows():
                if need <= 0: break
                if stock['remaining_shelf_life'] < 1: continue 
                if stock['stock_quantity'] <= 0: continue

                have = stock['stock_quantity']
                sell = min(need, have)
                self.current_stock.at[idx, 'stock_quantity'] -= sell
                sold_today += sell
                need -= sell
                
                self.daily_sales_amount += sell * self.item_props[item]['price']

        transferred = self.run_transshipment(day)

        expired = self.current_stock['remaining_shelf_life'] <= 0
        waste_count_today = 0
        
        expired_rows = self.current_stock[expired]
        for _, row in expired_rows.iterrows():
            qty = row['stock_quantity']
            item = row['item']
            if qty > 0:
                waste_count_today += qty
                self.daily_disposal_cost += qty * self.item_props[item]['disposal']
            
        self.total_waste_count += waste_count_today
        self.total_disposal_cost += self.daily_disposal_cost
        
        self.current_stock = self.current_stock[
            (self.current_stock['stock_quantity'] > 0) & 
            (self.current_stock['remaining_shelf_life'] > 0)
        ]
        self.current_stock['remaining_shelf_life'] -= 1
        
        daily_profit = self.daily_sales_amount - self.daily_procurement_cost - self.daily_disposal_cost - self.daily_transport_cost
        
        return waste_count_today, daily_profit

# ---------------------------------------------------------
# 4. メインUI
# ---------------------------------------------------------
def main():
    st.title("食品サプライチェーン経営シミュレーター")
    st.markdown("""
    4つの戦略(Random, FIFO, LP, New Optimization)を比較検証します。
    **「価格弾力性」** により、販売単価を変更すると需要が変動するリアルな市場原理を導入済みです。
    """)

    st.sidebar.header("経営パラメータ設定")
    
    with st.sidebar.expander("① 商品・店舗マスタ設定", expanded=True):
        st.caption("「基準価格」より高く売ると需要が減り、安く売ると増えます。")
        
        # ★基準価格と価格弾力性を追加
        default_items_data = {
            '商品名': ['トマト', '牛乳', 'パン'],
            '賞味期限(日)': [5, 7, 4],
            '基本需要(個)': [8, 6, 8],
            '発注基準(個)': [20, 15, 20],
            '販売単価(円)': [120, 200, 150],  # 現在の設定価格
            '基準価格(円)': [120, 200, 150],  # 需要計算の基準となる価格
            '価格弾力性': [1.5, 0.8, 1.2],    # 1.0より大きいと価格に敏感
            '仕入れ原価(円)': [60, 140, 70],
            '廃棄コスト(円)': [10, 20, 5]
        }
        df_items_default = pd.DataFrame(default_items_data)
        
        edited_items_df = st.data_editor(
            df_items_default, 
            num_rows="dynamic", 
            key="editor_items",
            column_config={
                "販売単価(円)": st.column_config.NumberColumn(format="¥%d"),
                "基準価格(円)": st.column_config.NumberColumn(format="¥%d"),
                "仕入れ原価(円)": st.column_config.NumberColumn(format="¥%d"),
                "廃棄コスト(円)": st.column_config.NumberColumn(format="¥%d"),
                "価格弾力性": st.column_config.NumberColumn(help="1.0:標準, >1:敏感, <1:鈍感")
            }
        )

        default_shops_data = {
            '店舗名': ['大学会館店', 'つくば駅前店', 'ひたち野牛久店', '研究学園店'],
            '規模倍率': [1.5, 1.0, 0.6, 0.8]
        }
        df_shops_default = pd.DataFrame(default_shops_data)
        
        edited_shops_df = st.data_editor(
            df_shops_default, 
            num_rows="dynamic",
            key="editor_shops"
        )

    with st.sidebar.expander("② シミュレーション条件", expanded=False):
        days = st.slider("期間 (日)", 10, 365, 30)
        demand_std = st.slider("需要のばらつき倍率", 0.0, 2.0, 1.0)
        threshold = st.slider("転送閾値 (New Model用)", 1, 10, 5)
        cost_unit = st.number_input("1個あたりの輸送コスト (円)", value=30)
        
        # --- シード値コントロールを追加 ---
        seed_val = st.number_input("乱数シード", value=42, step=1, help="同じ値にすると結果が再現されます")

    if st.sidebar.button("4戦略比較を実行", type="primary"):
        if edited_shops_df.empty or edited_items_df.empty:
            st.error("設定が必要です。")
            return

        strategies = ['Random', 'FIFO', 'LP', 'New Optimization']
        colors = {'Random': 'gray', 'FIFO': 'blue', 'LP': 'orange', 'New Optimization': 'red'}
        
        results = {}
        progress = st.progress(0)
        
        for i, strat in enumerate(strategies):
            # --- シード値を引数として渡す ---
            sim = RealWorldSupplySimulation(
                strategy=strat,
                shop_config_df=edited_shops_df,
                item_config_df=edited_items_df,
                random_seed=seed_val,  # ★ここをUIからの値に変更
                demand_std_scale=demand_std,
                transport_threshold=threshold,
                transport_cost_unit=cost_unit
            )
            
            daily_waste = []
            cumulative_profit = []
            current_cum_profit = 0
            
            for d in range(1, days + 1):
                w, p = sim.step(d)
                daily_waste.append(w)
                current_cum_profit += p
                cumulative_profit.append(current_cum_profit)
            
            gross_profit = sim.total_sales_amount - sim.total_procurement_cost
            final_profit = gross_profit - sim.total_disposal_cost - sim.total_transport_cost
            
            results[strat] = {
                "Profit": final_profit,
                "Sales": sim.total_sales_amount,
                "WasteCount": sim.total_waste_count,
                "WasteCost": sim.total_disposal_cost,
                "TransportCost": sim.total_transport_cost,
                "DailyWaste": daily_waste,
                "CumProfit": cumulative_profit
            }
            progress.progress((i + 1) / len(strategies))
        
        progress.empty()
        
        # --- 結果表示 ---
        st.subheader("📊 戦略別 損益比較")
        
        summary_data = []
        for s in strategies:
            r = results[s]
            summary_data.append({
                "戦略": s,
                "最終利益": f"¥{int(r['Profit']):,}",
                "売上高": f"¥{r['Sales']:,}",
                "廃棄個数": f"{r['WasteCount']}個",
                "廃棄コスト": f"¥{r['WasteCost']:,}",
                "輸送コスト": f"¥{r['TransportCost']:,}"
            })
        st.table(pd.DataFrame(summary_data))
        
        # --- グラフ ---
        st.subheader("📈 シミュレーション推移")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        plt.subplots_adjust(hspace=0.3)

        for s in strategies:
            alpha = 0.5 if s == 'Random' else 1.0
            width = 2.5 if s == 'New Optimization' else 1.5
            ax1.plot(results[s]["CumProfit"], label=s, color=colors[s], alpha=alpha, linewidth=width)
            ax2.plot(results[s]["DailyWaste"], label=s, color=colors[s], alpha=alpha, linewidth=width)
        
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_title("累積利益の推移 (高いほど良い)")
        ax1.set_ylabel("利益 (円)")
        ax1.set_xlabel("経過日数")
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()
        
        ax2.set_title("日次廃棄数の推移 (低いほど良い)")
        ax2.set_ylabel("廃棄数 (個)")
        ax2.set_xlabel("経過日数")
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()

        st.pyplot(fig)
        
        best_profit = max(results, key=lambda x: results[x]['Profit'])
        st.info(f"""
        **分析結果:** 最も利益が高かった戦略は **{best_profit}** です。
        表の「販売単価」を「基準価格」より高く設定して試してみてください。
        弾力性が高い商品は需要が減り、利益が悪化する様子が確認できます。
        """)

if __name__ == "__main__":
    main()
