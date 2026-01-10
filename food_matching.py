import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import urllib.request

# ---------------------------------------------------------
# 1. フォント設定 (Streamlit Cloud対応)
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
# 3. シミュレーションモデル (動的リスト対応版)
# ---------------------------------------------------------
class RealWorldSupplySimulation:
    def __init__(self, 
                 shop_list,          # ★引数追加: 店舗リスト
                 item_list,          # ★引数追加: 商品リスト
                 random_seed=42, 
                 demand_std_scale=1.0, 
                 supply_mean=35,
                 enable_transshipment=False, 
                 transport_threshold=5,
                 transport_cost_unit=10):
        
        self.rng = np.random.default_rng(random_seed)
        
        # ★ユーザー入力されたリストを使用
        self.shops = shop_list
        self.items = item_list
        
        # ★店舗・商品の特性を自動生成 (ハードコード廃止)
        # 既存の名前なら固定値、新しい名前ならランダム生成する柔軟な設計
        self.shop_scales = {}
        for shop in self.shops:
            # デフォルト設定(既知の店)
            defaults = {'大学会館店': 1.5, 'つくば駅前店': 1.0, 'ひたち野牛久店': 0.6, '研究学園店': 0.8}
            # 未知の店なら0.5~1.5倍の範囲でランダム設定
            self.shop_scales[shop] = defaults.get(shop, self.rng.uniform(0.5, 1.5))

        self.item_props = {}
        for item in self.items:
            # デフォルト設定(既知の商品)
            # base:基本需要, life:賞味期限
            defaults = {
                'トマト': {'base': 8, 'life': 5},
                '牛乳':   {'base': 6, 'life': 7},
                'パン':   {'base': 8, 'life': 4}
            }
            if item in defaults:
                self.item_props[item] = defaults[item]
            else:
                # 未知の商品ならランダム生成
                # 需要: 5~12, 賞味期限: 2~7日
                self.item_props[item] = {
                    'base': self.rng.integers(5, 12),
                    'life': self.rng.integers(2, 7)
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
        
        # ★自動生成されたプロパティ辞書から取得
        scale = self.shop_scales[shop]
        base = self.item_props[item]['base']
        
        return base * scale * factor

    def inbound_process(self, day):
        if (day - 1) % 7 == 6: return 

        new_rows = []
        for shop in self.shops:
            for item in self.items:
                expected = self.get_expected_demand(shop, item, day)
                # 入荷量のゆらぎ
                order_qty = max(0, int(self.rng.normal(expected * (self.supply_mean/30), 5)))
                
                if order_qty > 0:
                    # ★辞書から賞味期限を取得
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
    st.title("動的サプライチェーンシミュレーション (実用運用版)")
    st.markdown("""
    先行研究 (Chen et al., Olsson) に基づく「動的転送」モデル。
    店舗や商品を自由に追加して、ネットワーク規模の変化による影響を検証できます。
    """)

    st.sidebar.header("条件設定")
    
    # ★追加: 編集可能な店舗・商品リスト
    with st.sidebar.expander("① ネットワーク構成 (編集可能)", expanded=True):
        default_shops = "大学会館店, つくば駅前店, ひたち野牛久店, 研究学園店"
        default_items = "トマト, 牛乳, パン"
        
        shops_input = st.text_area("店舗名 (カンマ区切り)", value=default_shops, help="店舗を増やすと計算時間が増えます")
        items_input = st.text_area("商品名 (カンマ区切り)", value=default_items, help="新しい商品を追加すると賞味期限はランダム設定されます")
        
        # リストに変換
        shop_list = [s.strip() for s in shops_input.split(',') if s.strip()]
        item_list = [s.strip() for s in items_input.split(',') if s.strip()]

    with st.sidebar.expander("② 基本設定", expanded=False):
        days = st.slider("シミュレーション期間 (日)", 10, 60, 30)
        supply_mean = st.slider("基本入荷基準値", 20, 50, 30)
        demand_std = st.slider("需要のばらつき倍率", 0.0, 2.0, 1.0)
    
    with st.sidebar.expander("③ 転送・コスト設定", expanded=False):
        threshold = st.slider("転送閾値 (これ以下は送らない)", 1, 10, 5)
        cost_unit = st.number_input("1個あたりの輸送コスト (円)", value=30)

    if st.sidebar.button("検証開始", type="primary"):
        if not shop_list or not item_list:
            st.error("店舗名と商品名は少なくとも1つ以上入力してください。")
            return

        scenarios = [("従来モデル", False), ("提案モデル", True)]
        results = []
        progress = st.progress(0)
        
        for i, (name, enable) in enumerate(scenarios):
            sim = RealWorldSupplySimulation(
                shop_list=shop_list,  # ★入力を渡す
                item_list=item_list,  # ★入力を渡す
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
        col2.metric("輸送コスト", f"{int(prop['TransportCost']):,} 円", f"店舗数:{len(shop_list)}")
        col3.metric("経済効果", f"{int(cost_saving):,} 円", "廃棄削減 - 輸送費")

        st.subheader("日次廃棄量の推移")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(base["DailyWaste"], label="従来モデル", linestyle='--', color='gray')
        ax.plot(prop["DailyWaste"], label="提案モデル", color='red', linewidth=2)
        ax.legend()
        st.pyplot(fig)
        
        # 店舗数に応じたコメント
        if len(shop_list) > 6:
            st.info("💡 ヒント: 店舗数が多いほど、在庫転送のマッチング機会が増え、削減効果が高まりやすい傾向があります（スケールメリット）。")

if __name__ == "__main__":
    main()
