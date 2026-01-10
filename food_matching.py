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
st.set_page_config(layout="wide", page_title="食品サプライチェーン動的シミュレーター")

# ---------------------------------------------------------
# 3. シミュレーションモデル (設定受取対応版)
# ---------------------------------------------------------
class RealWorldSupplySimulation:
    def __init__(self, 
                 shop_config_df,     # ★変更: 店舗設定DFを受け取る
                 item_config_df,     # ★変更: 商品設定DFを受け取る
                 random_seed=42, 
                 demand_std_scale=1.0, 
                 supply_mean=35,
                 enable_transshipment=False, 
                 transport_threshold=5,
                 transport_cost_unit=10):
        
        self.rng = np.random.default_rng(random_seed)
        
        # 設定データフレームから情報を抽出
        # 1. 店舗情報
        self.shops = shop_config_df['店舗名'].tolist()
        self.shop_scales = dict(zip(shop_config_df['店舗名'], shop_config_df['規模倍率']))

        # 2. 商品情報
        self.items = item_config_df['商品名'].tolist()
        self.item_props = {}
        for _, row in item_config_df.iterrows():
            self.item_props[row['商品名']] = {
                'life': int(row['賞味期限(日)']),
                'base': int(row['基本需要(個)'])
            }

        # 在庫データ
        self.current_stock = pd.DataFrame(columns=[
            'stock_id', 'retail_store', 'item', 'stock_quantity', 'remaining_shelf_life'
        ])
        self.next_stock_id = 1
        
        # KPI
        self.total_waste_count = 0
        self.total_sales_count = 0
        self.total_transport_cost = 0 
        
        self.WEEKLY_DEMAND_PATTERN = [1.0, 0.9, 0.9, 1.0, 1.2, 1.4, 1.3]
        self.demand_std_scale = demand_std_scale
        self.supply_mean = supply_mean
        
        self.enable_transshipment = enable_transshipment
        self.transport_threshold = transport_threshold
        self.transport_cost_unit = transport_cost_unit

    def get_expected_demand(self, shop, item, day):
        weekday = (day - 1) % 7
        factor = self.WEEKLY_DEMAND_PATTERN[weekday]
        
        scale = self.shop_scales[shop]
        base = self.item_props[item]['base']
        
        return base * scale * factor

    def inbound_process(self, day):
        if (day - 1) % 7 == 6: return 

        new_rows = []
        for shop in self.shops:
            for item in self.items:
                expected = self.get_expected_demand(shop, item, day)
                order_qty = max(0, int(self.rng.normal(expected * (self.supply_mean/30), 5)))
                
                if order_qty > 0:
                    full_life = self.item_props[item]['life']
                    delay = int(self.rng.exponential(1.0))
                    life = max(1, full_life - delay)
                    
                    new_rows.append({
                        'stock_id': self.next_stock_id,
                        'retail_store': shop,
                        'item': item,
                        'stock_quantity': order_qty,
                        'remaining_shelf_life': life
                    })
                    self.next_stock_id += 1
        
        if new_rows:
            self.current_stock = pd.concat([self.current_stock, pd.DataFrame(new_rows)], ignore_index=True)

    def run_transshipment(self, day):
        if not self.enable_transshipment: return 0
        
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
                balance = current_qty - next_demand
                
                if balance > 0:
                    valid_stock = stock_df[stock_df['remaining_shelf_life'] >= 2]
                    sendable = valid_stock['stock_quantity'].sum()
                    surplus = max(0, sendable - next_demand)
                    if surplus > 0:
                        senders.append({'shop': shop, 'qty': surplus, 'df_index': valid_stock.index.tolist()})
                        
                elif balance < 0:
                    shortage = abs(balance)
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
                    self.total_transport_cost += amount * self.transport_cost_unit
                    
                    remaining = amount
                    for idx in sender['df_index']:
                        if remaining <= 0: break
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
        self.inbound_process(day)
        
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

        self.total_sales_count += sold_today
        transferred = self.run_transshipment(day)

        expired = self.current_stock['remaining_shelf_life'] <= 0
        waste_today = self.current_stock.loc[expired, 'stock_quantity'].sum()
        self.total_waste_count += waste_today
        
        self.current_stock = self.current_stock[
            (self.current_stock['stock_quantity'] > 0) & 
            (self.current_stock['remaining_shelf_life'] > 0)
        ]
        self.current_stock['remaining_shelf_life'] -= 1
        
        return waste_today, transferred

# ---------------------------------------------------------
# 4. メインUI
# ---------------------------------------------------------
def main():
    st.title("動的サプライチェーンシミュレーション (詳細設定版)")
    st.markdown("店舗リストや商品スペック（賞味期限・需要）を自由に変更してシミュレーションできます。")

    st.sidebar.header("条件設定")
    
    # --- ネットワーク構成 (編集可能テーブル) ---
    with st.sidebar.expander("① ネットワーク詳細設定", expanded=True):
        st.caption("下表を直接編集して、店舗や商品を追加・変更できます。")
        
        # 1. 店舗設定データの作成
        default_shops_data = {
            '店舗名': ['大学会館店', 'つくば駅前店', 'ひたち野牛久店', '研究学園店'],
            '規模倍率': [1.5, 1.0, 0.6, 0.8]
        }
        df_shops_default = pd.DataFrame(default_shops_data)
        
        st.markdown("**店舗設定**")
        edited_shops_df = st.data_editor(
            df_shops_default, 
            num_rows="dynamic", # 行の追加削除を許可
            key="editor_shops"
        )
        
        # 2. 商品設定データの作成
        default_items_data = {
            '商品名': ['トマト', '牛乳', 'パン', 'おにぎり', '弁当'],
            '賞味期限(日)': [5, 7, 4, 1, 1],
            '基本需要(個)': [8, 6, 8, 20, 15]
        }
        df_items_default = pd.DataFrame(default_items_data)
        
        st.markdown("**商品設定**")
        edited_items_df = st.data_editor(
            df_items_default, 
            num_rows="dynamic", # 行の追加削除を許可
            key="editor_items"
        )

    with st.sidebar.expander("② 基本設定", expanded=False):
        days = st.slider("シミュレーション期間 (日)", 10, 60, 30)
        supply_mean = st.slider("基本入荷基準値", 20, 50, 30)
        demand_std = st.slider("需要のばらつき倍率", 0.0, 2.0, 1.0)
    
    with st.sidebar.expander("③ 転送・コスト設定", expanded=False):
        threshold = st.slider("転送閾値 (これ以下は送らない)", 1, 10, 5)
        cost_unit = st.number_input("1個あたりの輸送コスト (円)", value=30)

    if st.sidebar.button("検証開始", type="primary"):
        # 入力チェック
        if edited_shops_df.empty or edited_items_df.empty:
            st.error("店舗と商品は少なくとも1つ以上設定してください。")
            return

        scenarios = [("従来モデル", False), ("提案モデル", True)]
        results = []
        progress = st.progress(0)
        
        for i, (name, enable) in enumerate(scenarios):
            sim = RealWorldSupplySimulation(
                shop_config_df=edited_shops_df, # ★テーブルデータを渡す
                item_config_df=edited_items_df, # ★テーブルデータを渡す
                supply_mean=supply_mean,
                demand_std_scale=demand_std,
                enable_transshipment=enable,
                transport_threshold=threshold,
                transport_cost_unit=cost_unit
            )
            daily_waste = []
            for d in range(1, days + 1):
                w, _ = sim.step(d)
                daily_waste.append(w)
            
            results.append({
                "Name": name,
                "Waste": sim.total_waste_count,
                "TransportCost": sim.total_transport_cost,
                "DailyWaste": daily_waste
            })
            progress.progress((i + 1) / len(scenarios))
        
        progress.empty()
        
        base = results[0]
        prop = results[1]
        
        waste_diff = base["Waste"] - prop["Waste"]
        rate = (waste_diff / base["Waste"] * 100) if base["Waste"] > 0 else 0
        
        WASTE_COST = 100
        cost_saving = (waste_diff * WASTE_COST) - prop["TransportCost"]

        col1, col2, col3 = st.columns(3)
        col1.metric("廃棄削減数", f"▲{int(waste_diff)}個", f"{rate:.1f}% 削減")
        col2.metric("輸送コスト", f"{int(prop['TransportCost']):,} 円", f"商品数:{len(edited_items_df)}")
        col3.metric("経済効果", f"{int(cost_saving):,} 円", "廃棄削減 - 輸送費")

        st.subheader("日次廃棄量の推移")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(base["DailyWaste"], label="従来モデル", linestyle='--', color='gray')
        ax.plot(prop["DailyWaste"], label="提案モデル", color='red', linewidth=2)
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
