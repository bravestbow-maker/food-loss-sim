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
                'base_price': int(row['基準価格(円)']),    # 基準価格
                'elasticity': float(row['価格弾力性']),    # 価格弾力性
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
        
        # サービスレベル計算用
        self.total_demand_qty = 0
        self.total_sold_qty = 0
        
        # 日次計算用
        self.daily_procurement_cost = 0
        self.daily_sales_amount = 0
        self.daily_transport_cost = 0
        self.daily_disposal_cost = 0
        self.daily_profit = 0  # 追加
        
        self.WEEKLY_DEMAND_PATTERN = [1.0, 0.9, 0.9, 1.0, 1.2, 1.4, 1.3]
        self.demand_std_scale = demand_std_scale
        
        # 転送パラメータ
        self.transport_threshold = transport_threshold
        self.transport_cost_unit = transport_cost_unit

    # 価格弾力性を考慮した需要計算
    def get_expected_demand(self, shop, item, day):
        weekday = (day - 1) % 7
        factor = self.WEEKLY_DEMAND_PATTERN[weekday]
        
        # 1. 店舗規模 × 商品基本需要
        scale = self.shop_scales[shop]
        base_demand = self.item_props[item]['base_demand']
        
        # 2. 価格弾力性による補正
        current_price = self.item_props[item]['price']
        base_price = self.item_props[item]['base_price']
        elasticity = self.item_props[item]['elasticity']
        
        if base_price <= 0: base_price = 1
        
        price_ratio = current_price / base_price
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
                # 需要予測に基づく発注量の調整
                current_price = self.item_props[item]['price']
                base_price = self.item_props[item]['base_price']
                elasticity = self.item_props[item]['elasticity']
                if base_price <= 0: base_price = 1
                price_ratio = current_price / base_price
                price_factor = price_ratio ** (-elasticity)

                base_target = self.item_props[item]['target_stock']
                scale = self.shop_scales[shop]
                
                target_level = base_target * scale * price_factor
                
                # 発注点方式 (Randomを削除し統一)
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
        # Random削除に伴い条件変更
        if self.strategy == 'FIFO': return 0
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
            # --- ★ コスト対効果の判定 ---
            unit_price = self.item_props[item]['price']
            disposal_cost = self.item_props[item]['disposal']
            economic_value = unit_price + disposal_cost
            
            if self.transport_cost_unit > economic_value:
                continue # 輸送費が高すぎて割に合わないためスキップ

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
                expected = self.get_expected_demand(shop, item, day)
                qty = max(0, int(self.rng.normal(expected, 4 * self.demand_std_scale)))
                if qty > 0:
                    demand_rows.append({'shop': shop, 'item': item, 'qty': qty})
                    # 総需要数のカウント
                    self.total_demand_qty += qty
        
        self.current_stock.reset_index(drop=True, inplace=True)
        
        for d in demand_rows:
            shop, item, need = d['shop'], d['item'], d['qty']
            
            # --- ★ FF (Fresh First) 実装部分 ---
            # ascending=False に変更: 賞味期限が「長い（新しい）」順に並べ替え
            # これにより、顧客は最も新鮮なものを優先して購入する挙動となる
            targets = self.current_stock[
                (self.current_stock['retail_store'] == shop) & 
                (self.current_stock['item'] == item)
            ].sort_values('remaining_shelf_life', ascending=False)
            
            for idx, stock in targets.iterrows():
                if need <= 0: break
                if stock['remaining_shelf_life'] < 1: continue 
                if stock['stock_quantity'] <= 0: continue

                have = stock['stock_quantity']
                sell = min(need, have)
                self.current_stock.at[idx, 'stock_quantity'] -= sell
                sold_today += sell
                # 総販売数のカウント
                self.total_sold_qty += sell
                
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
        
        self.daily_profit = self.daily_sales_amount - self.daily_procurement_cost - self.daily_disposal_cost - self.daily_transport_cost
        
        return waste_count_today, self.daily_profit

# ---------------------------------------------------------
# 4. メインUI
# ---------------------------------------------------------
def main():
    st.title("食品サプライチェーン経営シミュレーター")
    
    # --- 解説パネルの追加 ---
    with st.expander("📖 シミュレーションの仕組みと戦略の解説"):
        st.markdown("""
        ### 1. 経済モデル：価格弾力性
        商品は価格によって需要が変動します。「基準価格」より高く売ると需要は減少し、安く売ると増加します。
        
        **需要計算式:** $$需要 = 基本需要 \\times \\left( \\frac{販売単価}{基準価格} \\right)^{-\\text{価格弾力性}}$$
        
        ---
        ### 2. 戦略の違い
        このシミュレーションでは3つの在庫管理戦略を比較します。
        
        1.  **FIFO (先入先出・発注点方式)**
            * 毎朝、減った在庫分をきっちり発注して補充します。
            * 店舗間の在庫転送は行いません。
            * **特徴:** 基本的な管理手法ですが、需要の急変動には弱く、店ごとの過不足を解消できません。

        2.  **LP (線形計画法・最適化)**
            * 数理最適化ソルバー(`PuLP`)を使用します。
            * 全店舗の在庫状況を見て、「利益が最大（輸送コストも考慮）」になるように最適な在庫転送ルートを計算します。
            * **特徴:** 理論上の「最強の経営」ですが、計算コストがかかります。

        3.  **New Optimization (ヒューリスティック・独自戦略)**
            * 「余っている店」から「足りない店」へ、ルールベースで融通（転送）します。
            * **重要:** 「輸送コスト」が「商品の利益＋廃棄回避額」を上回る場合は、転送せずに廃棄を選択する賢いコスト判定を行います。
            * **特徴:** 高速な計算で、LPに近い利益を出そうとする実用的な戦略です。
            
        **※顧客行動モデル:**
        本シミュレーションでは**「FF (Fresh First)」**を採用しています。
        顧客は**「賞味期限が新しいもの」**を優先して購入するため、棚には古い商品が残りやすく、廃棄リスクが高い過酷な環境設定となっています。
        """)

    st.markdown("""
    左側のサイドバーでパラメータを調整し、「3戦略比較を実行」ボタンを押してください。
    """)

    st.sidebar.header("経営パラメータ設定")
    
    with st.sidebar.expander("① 商品・店舗マスタ設定", expanded=True):
        st.caption("「基準価格」より高く売ると需要が減り、安く売ると増えます。")
        
        default_items_data = {
            '商品名': ['トマト', '牛乳', 'パン'],
            '賞味期限(日)': [5, 7, 4],
            '基本需要(個)': [8, 6, 8],
            '発注基準(個)': [20, 15, 20],
            '販売単価(円)': [120, 200, 150],
            '基準価格(円)': [120, 200, 150],
            '価格弾力性': [1.5, 0.8, 1.2],
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
        seed_val = st.number_input("乱数シード", value=42, step=1, help="同じ値にすると結果が再現されます")

    if st.sidebar.button("3戦略比較を実行", type="primary"):
        if edited_shops_df.empty or edited_items_df.empty:
            st.error("設定が必要です。")
            return

        strategies = ['FIFO', 'LP', 'New Optimization']
        colors = {'FIFO': 'blue', 'LP': 'orange', 'New Optimization': 'red'}
        
        results = {}
        progress = st.progress(0)
        
        for i, strat in enumerate(strategies):
            sim = RealWorldSupplySimulation(
                strategy=strat,
                shop_config_df=edited_shops_df,
                item_config_df=edited_items_df,
                random_seed=seed_val,
                demand_std_scale=demand_std,
                transport_threshold=threshold,
                transport_cost_unit=cost_unit
            )
            
            daily_waste = []
            cumulative_profit = []
            daily_profits = [] # 追加: 日次利益の保存用
            current_cum_profit = 0
            
            for d in range(1, days + 1):
                w, p = sim.step(d)
                daily_waste.append(w)
                daily_profits.append(p)
                current_cum_profit += p
                cumulative_profit.append(current_cum_profit)
            
            gross_profit = sim.total_sales_amount - sim.total_procurement_cost
            final_profit = gross_profit - sim.total_disposal_cost - sim.total_transport_cost
            
            # サービス率の計算
            service_level = (sim.total_sold_qty / sim.total_demand_qty * 100) if sim.total_demand_qty > 0 else 0
            
            results[strat] = {
                "Profit": final_profit,
                "Sales": sim.total_sales_amount,
                "ProcurementCost": sim.total_procurement_cost,
                "WasteCount": sim.total_waste_count,
                "WasteCost": sim.total_disposal_cost,
                "TransportCost": sim.total_transport_cost,
                "DailyWaste": daily_waste,
                "CumProfit": cumulative_profit,
                "DailyProfits": daily_profits,
                "ServiceLevel": service_level
            }
            progress.progress((i + 1) / len(strategies))
        
        progress.empty()
        
        # --- 結果表示 (Summary Table) ---
        st.subheader(" 戦略別 損益・KPI比較")
        
        summary_data = []
        for s in strategies:
            r = results[s]
            summary_data.append({
                "戦略": s,
                "最終利益": f"¥{int(r['Profit']):,}",
                "サービス率": f"{r['ServiceLevel']:.1f}%",
                "売上高": f"¥{r['Sales']:,}",
                "廃棄コスト": f"¥{r['WasteCost']:,}",
                "輸送コスト": f"¥{r['TransportCost']:,}"
            })
        st.table(pd.DataFrame(summary_data))
        
        # --- 比較モデル詳細検討 (Advanced Analysis) ---
        st.markdown("---")
        st.subheader(" 比較モデルの検討（詳細分析）")
        
        col_analysis_1, col_analysis_2 = st.columns(2)
        
        # 1. コスト構造分析 (Stacked Bar Chart)
        with col_analysis_1:
            st.markdown("##### コスト構造の比較")
            st.caption("利益を生むためには、廃棄と輸送のバランスが重要です。")
            
            fig_cost, ax_cost = plt.subplots(figsize=(6, 4))
            bar_width = 0.6
            x_pos = np.arange(len(strategies))
            
            # データの準備
            procurements = [results[s]['ProcurementCost'] for s in strategies]
            wastes = [results[s]['WasteCost'] for s in strategies]
            transports = [results[s]['TransportCost'] for s in strategies]
            profits = [results[s]['Profit'] for s in strategies]
            
            # 負の利益対応: 利益がマイナスの場合は0として積み上げ、別途表示等を検討（ここでは簡易表示）
            pos_profits = [max(0, p) for p in profits]

            p1 = ax_cost.bar(x_pos, procurements, bar_width, label='仕入', color='#a6cee3')
            p2 = ax_cost.bar(x_pos, wastes, bar_width, bottom=procurements, label='廃棄', color='#e31a1c')
            p3 = ax_cost.bar(x_pos, transports, bar_width, bottom=np.array(procurements)+np.array(wastes), label='輸送', color='#ff7f00')
            p4 = ax_cost.bar(x_pos, pos_profits, bar_width, bottom=np.array(procurements)+np.array(wastes)+np.array(transports), label='利益', color='#33a02c')

            ax_cost.set_xticks(x_pos)
            ax_cost.set_xticklabels(strategies, fontsize=9)
            ax_cost.set_ylabel("金額 (円)")
            ax_cost.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
            ax_cost.grid(axis='y', linestyle='--', alpha=0.4)
            
            st.pyplot(fig_cost)

        # 2. 利益の安定性分析 (Box Plot)
        with col_analysis_2:
            st.markdown("##### 利益の安定性 (リスク分析)")
            st.caption("日々の利益のばらつき（箱ひげ図）。箱が小さく高い位置にあるのが理想です。")
            
            fig_risk, ax_risk = plt.subplots(figsize=(6, 4))
            
            data_to_plot = [results[s]['DailyProfits'] for s in strategies]
            
            ax_risk.boxplot(data_to_plot, labels=strategies, patch_artist=True,
                            boxprops=dict(facecolor="lightblue", color="blue"),
                            medianprops=dict(color="red"))
            
            ax_risk.set_ylabel("日次利益 (円)")
            ax_risk.grid(axis='y', linestyle='--', alpha=0.4)
            st.pyplot(fig_risk)

        # --- 基本グラフ (Trend) ---
        st.markdown("---")
        st.subheader("シミュレーション推移")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        plt.subplots_adjust(hspace=0.3)

        for s in strategies:
            alpha = 1.0
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
        
        # 結論の動的生成
        best_strat = max(results, key=lambda x: results[x]['Profit'])
        worst_strat = min(results, key=lambda x: results[x]['Profit'])
        st.info(f"""
        **💡 分析結果サマリー:**
        最も優れた成果を出したのは **{best_strat}** です。
        
        * **利益最大:** {best_strat} (¥{int(results[best_strat]['Profit']):,})
        * **サービス率:** {results[best_strat]['ServiceLevel']:.1f}%
        * **廃棄削減:** {best_strat}の廃棄コストは {worst_strat} と比較して大幅に抑制されています。
        
        詳細分析の「コスト構造」を見ると、LPやNew Optimizationは「輸送コスト」をかけてでも「廃棄」を防ぐことで、結果的に利益を最大化していることが分かります。
        また、本シミュレーションでは顧客が新しい商品を優先的に購入する**FF (Fresh First)** モデルを採用しているため、古い在庫が残りやすく、適切な在庫転送を行わないFIFO戦略では廃棄が増加する傾向にあります。
        """)

if __name__ == "__main__":
    main()
