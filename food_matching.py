import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import urllib.request

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
# 3. シミュレーションモデル (ロジック修正版)
# ---------------------------------------------------------
class RealWorldSupplySimulation:
    def __init__(self, 
                 shop_config_df,     
                 item_config_df,     
                 random_seed=42, 
                 demand_std_scale=1.0, 
                 enable_transshipment=False, 
                 transport_threshold=5,
                 transport_cost_unit=10):
        
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
                'target_stock': int(row['発注基準(個)']), # ★意味変更: 目標在庫レベル
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
        
        self.enable_transshipment = enable_transshipment
        self.transport_threshold = transport_threshold
        self.transport_cost_unit = transport_cost_unit

    def get_expected_demand(self, shop, item, day):
        weekday = (day - 1) % 7
        factor = self.WEEKLY_DEMAND_PATTERN[weekday]
        scale = self.shop_scales[shop]
        base = self.item_props[item]['base_demand']
        return base * scale * factor

    # ---------------------------------------------------------
    # ★ロジック修正: 発注点方式 (Order-Up-To Policy)
    # 現在の在庫を確認し、目標在庫(target_stock)まで補充する
    # ---------------------------------------------------------
    def inbound_process(self, day):
        if (day - 1) % 7 == 6: return # 日曜発注なし

        new_rows = []
        for shop in self.shops:
            for item in self.items:
                # 1. 現在の有効在庫数を確認
                current_stock_df = self.current_stock[
                    (self.current_stock['retail_store'] == shop) & 
                    (self.current_stock['item'] == item)
                ]
                current_qty = current_stock_df['stock_quantity'].sum()
                
                # 2. 目標在庫レベル (店舗規模に応じて調整)
                base_target = self.item_props[item]['target_stock']
                scale = self.shop_scales[shop]
                target_level = base_target * scale
                
                # 3. 発注量の計算 (目標 - 現在)
                # 足りない分だけ発注する。マイナスなら発注しない。
                needed_qty = target_level - current_qty
                
                # 発注量のゆらぎ (オペレーション誤差)
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
                    
                    # コスト加算
                    cost = order_qty * props['cost']
                    self.daily_procurement_cost += cost
                    self.total_procurement_cost += cost
        
        if new_rows:
            self.current_stock = pd.concat([self.current_stock, pd.DataFrame(new_rows)], ignore_index=True)

    def run_transshipment(self, day):
        if not self.enable_transshipment: return 0
        
        transferred_count = 0
        new_transferred_stock = []
        
        # インデックスリセット (エラー防止)
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
                
                # 安全在庫係数 (これより多くないと送らない)
                safety_stock = next_demand * 0.2 
                balance = current_qty - (next_demand + safety_stock)
                
                if balance > 0:
                    # 送り手: 賞味期限2日以上のみ
                    valid_stock = stock_df[stock_df['remaining_shelf_life'] >= 2]
                    sendable = valid_stock['stock_quantity'].sum()
                    surplus = max(0, sendable - (next_demand + safety_stock))
                    
                    if surplus > 0:
                        # indexリストを保持
                        senders.append({'shop': shop, 'qty': surplus, 'df_index': valid_stock.index.tolist()})
                        
                elif current_qty < next_demand:
                    # 受け手: 明日の分が足りない
                    shortage = next_demand - current_qty
                    urgency = shortage / (next_demand + 1)
                    receivers.append({'shop': shop, 'qty': shortage, 'urgency': urgency})

            # マッチング
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
                    # 送り手の在庫を減らす
                    for idx in sender['df_index']:
                        if remaining <= 0: break
                        
                        # current_stockから現在の値を取得
                        if idx not in self.current_stock.index: continue
                        have = self.current_stock.at[idx, 'stock_quantity']
                        
                        if have <= 0: continue

                        take = min(have, remaining)
                        self.current_stock.at[idx, 'stock_quantity'] -= take
                        remaining -= take
                        
                        # 新しい行を作成 (受け手用)
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
        
        # 1. 入荷 (修正済み: 足りない分だけ発注)
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
                
                have = stock['stock_quantity']
                if have <= 0: continue

                sell = min(need, have)
                self.current_stock.at[idx, 'stock_quantity'] -= sell
                sold_today += sell
                need -= sell
                
                self.daily_sales_amount += sell * self.item_props[item]['price']

        # 3. 転送
        transferred = self.run_transshipment(day)

        # 4. 廃棄
        expired = self.current_stock['remaining_shelf_life'] <= 0
        waste_count_today = 0
        
        expired_rows = self.current_stock[expired]
        for _, row in expired_rows.iterrows():
            qty = row['stock_quantity']
            item = row['item']
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
    **修正版ロジック搭載**: 「発注点方式」により、売れた分だけ補充するリアルな在庫管理を実現。
    在庫の垂れ流しを防いだ上で、転送による最適化効果を検証します。
    """)

    st.sidebar.header("経営パラメータ設定")
    
    with st.sidebar.expander("① 商品・店舗マスタ設定", expanded=True):
        st.caption("「発注基準」は**目標在庫レベル(Order-Up-To Level)**として機能します。")
        
        # 発注基準を少し大きめに修正(在庫バッファを持たせるため)
        default_items_data = {
            '商品名': ['トマト', '牛乳', 'パン'],
            '賞味期限(日)': [5, 7, 4],
            '基本需要(個)': [8, 6, 8],
            '発注基準(個)': [20, 15, 20],      # 目標在庫数 (1日あたりの需要の2~3倍程度が目安)
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
        threshold = st.slider("転送閾値 (個)", 1, 10, 5)
        cost_unit = st.number_input("1個あたりの輸送コスト (円)", value=30)

    if st.sidebar.button("経営分析を開始", type="primary"):
        if edited_shops_df.empty or edited_items_df.empty:
            st.error("店舗と商品は設定が必要です。")
            return

        scenarios = [("従来モデル", False), ("提案モデル", True)]
        results = []
        progress = st.progress(0)
        
        for i, (name, enable) in enumerate(scenarios):
            sim = RealWorldSupplySimulation(
                shop_config_df=edited_shops_df,
                item_config_df=edited_items_df,
                demand_std_scale=demand_std,
                enable_transshipment=enable,
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
            
            # 最終利益
            gross_profit = sim.total_sales_amount - sim.total_procurement_cost
            final_profit = gross_profit - sim.total_disposal_cost - sim.total_transport_cost
            
            results.append({
                "Name": name,
                "Profit": final_profit,
                "Sales": sim.total_sales_amount,
                "WasteCount": sim.total_waste_count,
                "WasteCost": sim.total_disposal_cost,
                "TransportCost": sim.total_transport_cost,
                "DailyWaste": daily_waste,
                "CumProfit": cumulative_profit
            })
            progress.progress((i + 1) / len(scenarios))
        
        progress.empty()
        
        base = results[0]
        prop = results[1]
        profit_diff = prop["Profit"] - base["Profit"]
        
        # --- P/L ---
        st.subheader("💰 損益計算書 (P/L) 比較")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("① 従来モデル 最終利益", f"¥{int(base['Profit']):,}")
        col2.metric("② 提案モデル 最終利益", f"¥{int(prop['Profit']):,}")
        delta_color = "normal" if profit_diff > 0 else "inverse"
        col3.metric("利益改善額 (②-①)", f"¥{int(profit_diff):,}", delta_color=delta_color)

        detail_data = {
            "項目": ["売上高", "仕入原価", "廃棄コスト", "輸送コスト", "【最終利益】", "(参考)廃棄個数"],
            "従来モデル": [
                f"¥{base['Sales']:,}", f"¥-{int(base['Sales'] - base['Profit'] - base['WasteCost']):,}",
                f"¥-{base['WasteCost']:,}", "¥0", f"**¥{base['Profit']:,}**", f"{base['WasteCount']}個"
            ],
            "提案モデル": [
                f"¥{prop['Sales']:,}", f"¥-{int(prop['Sales'] - prop['Profit'] - prop['WasteCost'] - prop['TransportCost']):,}",
                f"¥-{prop['WasteCost']:,}", f"¥-{prop['TransportCost']:,}", f"**¥{prop['Profit']:,}**", f"{prop['WasteCount']}個"
            ]
        }
        st.table(pd.DataFrame(detail_data))

        # --- Graph ---
        st.subheader("📈 シミュレーション推移")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        plt.subplots_adjust(hspace=0.3)

        ax1.plot(base["CumProfit"], label="従来モデル", linestyle='--', color='gray')
        ax1.plot(prop["CumProfit"], label="提案モデル", color='green', linewidth=2)
        ax1.set_title("累積利益の推移 (在庫適正化済み)")
        ax1.set_ylabel("利益 (円)")
        ax1.set_xlabel("経過日数")
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()
        
        ax2.plot(base["DailyWaste"], label="従来モデル", linestyle='--', color='gray')
        ax2.plot(prop["DailyWaste"], label="提案モデル", color='red', linewidth=2)
        ax2.set_title("日次廃棄数の推移")
        ax2.set_ylabel("廃棄数 (個)")
        ax2.set_xlabel("経過日数")
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()

        st.pyplot(fig)

if __name__ == "__main__":
    main()
