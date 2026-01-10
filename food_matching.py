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
# 3. シミュレーションモデル (4戦略対応版)
# ---------------------------------------------------------
class RealWorldSupplySimulation:
    def __init__(self, 
                 strategy,           # ★戦略 (Random, FIFO, LP, New Model)
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

        # 2. 商品情報
        self.items = item_config_df['商品名'].tolist()
        self.item_props = {}
        for _, row in item_config_df.iterrows():
            self.item_props[row['商品名']] = {
                'life': int(row['賞味期限(日)']),
                'base_demand': int(row['基本需要(個)']),
                'target_stock': int(row['発注基準(個)']),
                'price': int(row['販売単価(円)']),
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

    def get_expected_demand(self, shop, item, day):
        weekday = (day - 1) % 7
        factor = self.WEEKLY_DEMAND_PATTERN[weekday]
        scale = self.shop_scales[shop]
        base = self.item_props[item]['base_demand']
        return base * scale * factor

    # ---------------------------------------------------------
    # 入荷プロセス (Inbound)
    # ---------------------------------------------------------
    def inbound_process(self, day):
        if (day - 1) % 7 == 6: return 

        new_rows = []
        for shop in self.shops:
            for item in self.items:
                # 戦略による発注精度の違い
                base_target = self.item_props[item]['target_stock']
                scale = self.shop_scales[shop]
                target_level = base_target * scale
                
                if self.strategy == 'Random':
                    # Random戦略: 在庫状況を見ずに適当に発注する (精度が低い)
                    # 目標在庫の 0.5倍〜1.5倍 の量をランダムに入荷
                    order_qty = int(self.rng.uniform(target_level * 0.5, target_level * 1.5))
                else:
                    # FIFO, LP, New Model: 発注点方式 (賢い発注)
                    # 現在在庫を確認して、足りない分だけ発注
                    current_stock_df = self.current_stock[
                        (self.current_stock['retail_store'] == shop) & 
                        (self.current_stock['item'] == item)
                    ]
                    current_qty = current_stock_df['stock_quantity'].sum()
                    needed_qty = target_level - current_qty
                    # 誤差を含める
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
        # RandomとFIFOは転送を行わない
        if self.strategy in ['Random', 'FIFO']: return 0
        
        # LP戦略: 数理最適化ソルバーで転送決定
        if self.strategy == 'LP':
            return self.run_lp_optimization(day)
            
        # New Model: 提案手法（ヒューリスティック）で転送決定
        if self.strategy == 'New Optimization':
            return self.run_heuristic_optimization(day)
            
        return 0

    # ★LP転送ロジック (Solver使用)
    def run_lp_optimization(self, day):
        transferred_count = 0
        new_transferred_stock = []
        self.current_stock.reset_index(drop=True, inplace=True)

        for item in self.items:
            # 1. 各店舗の余剰と不足を計算
            balances = {} # shop -> +/- qty
            valid_indices = {} # shop -> list of indices
            
            for shop in self.shops:
                stock_df = self.current_stock[
                    (self.current_stock['retail_store'] == shop) & 
                    (self.current_stock['item'] == item)
                ]
                current_qty = stock_df['stock_quantity'].sum()
                next_demand = self.get_expected_demand(shop, item, day + 1)
                
                # 転送可能な在庫 (期限2日以上)
                valid_stock = stock_df[stock_df['remaining_shelf_life'] >= 2]
                valid_indices[shop] = valid_stock.index.tolist()
                
                # LPは「全体最適」なので、明日の需要に対する過不足をそのまま使う
                balance = current_qty - next_demand
                balances[shop] = int(balance)

            senders = [s for s, b in balances.items() if b > 0]
            receivers = [r for r, b in balances.items() if b < 0]
            
            if not senders or not receivers: continue

            # 2. LP問題の定義
            prob = LpProblem(f"Transshipment_{item}_{day}", LpMaximize)
            
            # 変数: x[sender][receiver] = 移動量
            x = LpVariable.dicts("route", (senders, receivers), 0, None, LpInteger)
            
            # 目的関数: 転送による「救済価値」 - 「輸送コスト」
            # 救済価値 = 販売単価 (売れるようになるから)
            unit_price = self.item_props[item]['price']
            prob += lpSum([x[s][r] * (unit_price - self.transport_cost_unit) for s in senders for r in receivers])
            
            # 制約条件
            for s in senders:
                # 送る量は余剰分を超えない
                prob += lpSum([x[s][r] for r in receivers]) <= balances[s]
                
            for r in receivers:
                # 受け取る量は不足分を超えない
                prob += lpSum([x[s][r] for s in senders]) <= abs(balances[r])

            # ソルバー実行
            prob.solve(PULP_CBC_CMD(msg=0))
            
            # 3. 結果の適用
            for s in senders:
                for r in receivers:
                    amount = x[s][r].value()
                    if amount and amount > 0:
                        amount = int(amount)
                        transferred_count += amount
                        
                        t_cost = amount * self.transport_cost_unit
                        self.daily_transport_cost += t_cost
                        self.total_transport_cost += t_cost
                        
                        # 在庫移動処理
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

    # ★New Model転送ロジック (提案手法: プロアクティブ + 閾値)
    def run_heuristic_optimization(self, day):
        transferred_count = 0
        new_transferred_stock = []
        self.current_stock.reset_index(drop=True, inplace=True)

        for item in self.items:
            senders = []
            receivers = []
            
            for shop in self.shops:
                stock_df = self.current_stock[
                    (self.current_stock['retail_store'] == shop) & 
                    (self.current_stock['item'] == item)
                ]
                current_qty = stock_df['stock_quantity'].sum()
                next_demand = self.get_expected_demand(shop, item, day + 1)
                
                # 安全在庫係数 (これより多くないと送らない = 実用的なバッファ)
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
                    
                    # ★閾値制御 (LPには無い、実用的な制約)
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
        
        # 1. 入荷
        self.inbound_process(day)
        
        # 2. 販売
        sold_today = 0
        demand_rows = []
        for shop in self.shops:
            for item in self.items:
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

        # 3. 転送 (戦略に応じて実行)
        transferred = self.run_transshipment(day)

        # 4. 廃棄
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
    4つの戦略を比較検証します：
    1. **Random**: 発注が不正確、転送なし
    2. **FIFO**: 発注は正確(発注点方式)、転送なし [基準]
    3. **LP**: 発注は正確、**数理最適化**による理想的な転送
    4. **New Optimization**: 発注は正確、**提案手法(閾値+予測)**による現実的な転送
    """)

    st.sidebar.header("経営パラメータ設定")
    
    with st.sidebar.expander("① 商品・店舗マスタ設定", expanded=True):
        st.caption("発注基準＝目標在庫レベル")
        
        default_items_data = {
            '商品名': ['トマト', '牛乳', 'パン'],
            '賞味期限(日)': [5, 7, 4],
            '基本需要(個)': [8, 6, 8],
            '発注基準(個)': [20, 15, 20],
            '販売単価(円)': [120, 200, 150],
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
                "仕入れ原価(円)": st.column_config.NumberColumn(format="¥%d"),
                "廃棄コスト(円)": st.column_config.NumberColumn(format="¥%d"),
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
        days = st.slider("期間 (日)", 10, 60, 30)
        demand_std = st.slider("需要のばらつき倍率", 0.0, 2.0, 1.0)
        threshold = st.slider("転送閾値 (New Model用)", 1, 10, 5)
        cost_unit = st.number_input("1個あたりの輸送コスト (円)", value=30)

    if st.sidebar.button("4戦略比較を実行", type="primary"):
        if edited_shops_df.empty or edited_items_df.empty:
            st.error("設定が必要です。")
            return

        # 4つの戦略を定義
        strategies = ['Random', 'FIFO', 'LP', 'New Optimization']
        colors = {'Random': 'gray', 'FIFO': 'blue', 'LP': 'orange', 'New Optimization': 'red'}
        
        results = {}
        progress = st.progress(0)
        
        for i, strat in enumerate(strategies):
            sim = RealWorldSupplySimulation(
                strategy=strat,
                shop_config_df=edited_shops_df,
                item_config_df=edited_items_df,
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
        
        # データフレーム作成
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
            # Randomはばらつきが大きいので少し薄くする
            alpha = 0.5 if s == 'Random' else 1.0
            width = 2.5 if s == 'New Optimization' else 1.5
            
            ax1.plot(results[s]["CumProfit"], label=s, color=colors[s], alpha=alpha, linewidth=width)
            ax2.plot(results[s]["DailyWaste"], label=s, color=colors[s], alpha=alpha, linewidth=width)

        ax1.set_title("累積利益の推移")
        ax1.set_ylabel("利益 (円)")
        ax1.set_xlabel("経過日数")
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()
        
        ax2.set_title("日次廃棄数の推移")
        ax2.set_ylabel("廃棄数 (個)")
        ax2.set_xlabel("経過日数")
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()

        st.pyplot(fig)
        
        # 考察コメント
        best_profit = max(results, key=lambda x: results[x]['Profit'])
        st.info(f"""
        **分析結果:** 最も利益が高かった戦略は **{best_profit}** です。
        LPは理論上の最適解を出しますが、輸送コストや閾値を考慮する New Optimization も、
        現実的な制約の中で高いパフォーマンスを発揮できているか確認してください。
        """)

if __name__ == "__main__":
    main()
