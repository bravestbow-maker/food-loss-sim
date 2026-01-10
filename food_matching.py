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
# 3. シミュレーションモデル (修正版: バグ修正済み)
# ---------------------------------------------------------
class RealWorldSupplySimulation:
    def __init__(self, 
                 random_seed=42, 
                 demand_std_scale=1.0, 
                 supply_mean=35,
                 enable_transshipment=False, 
                 transport_threshold=5,
                 transport_cost_unit=10):
        
        self.shops = ['大学会館店', 'つくば駅前店', 'ひたち野牛久店', '研究学園店']
        self.items = ['トマト', '牛乳', 'パン']
        self.rng = np.random.default_rng(random_seed)
        
        # 在庫データ
        self.current_stock = pd.DataFrame(columns=[
            'stock_id', 'retail_store', 'item', 'stock_quantity', 'remaining_shelf_life'
        ])
        self.next_stock_id = 1
        
        self.total_waste_count = 0
        self.total_sales_count = 0
        self.total_transport_cost = 0 
        
        self.WEEKLY_DEMAND_PATTERN = [1.0, 0.9, 0.9, 1.0, 1.2, 1.4, 1.3]
        self.demand_std_scale = demand_std_scale
        self.shelf_life_dict = {'トマト': 5, '牛乳': 7, 'パン': 4}
        self.supply_mean = supply_mean
        
        self.enable_transshipment = enable_transshipment
        self.transport_threshold = transport_threshold
        self.transport_cost_unit = transport_cost_unit

    def get_expected_demand(self, shop, item, day):
        weekday = (day - 1) % 7
        factor = self.WEEKLY_DEMAND_PATTERN[weekday]
        scale = {'大学会館店': 1.5, 'つくば駅前店': 1.0, 'ひたち野牛久店': 0.6, '研究学園店': 0.8}[shop]
        base = {'トマト': 8, '牛乳': 6, 'パン': 8}[item]
        return base * scale * factor

    def inbound_process(self, day):
        if (day - 1) % 7 == 6: return 

        new_rows = [] # 追加用リスト
        for shop in self.shops:
            for item in self.items:
                expected = self.get_expected_demand(shop, item, day)
                order_qty = max(0, int(self.rng.normal(expected * (self.supply_mean/30), 5)))
                
                if order_qty > 0:
                    full_life = self.shelf_life_dict[item]
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

    # ---------------------------------------------------------
    # ★修正箇所: 安全な転送処理 (IndexError回避)
    # ---------------------------------------------------------
    def run_transshipment(self, day):
        if not self.enable_transshipment: return 0
        
        transferred_count = 0
        # ★追加分を一時保存するリスト（ループ中のconcatを避けるため）
        new_transferred_stock = []
        
        # 処理前にインデックスをリセットして整理する（重要）
        self.current_stock.reset_index(drop=True, inplace=True)

        for item in self.items:
            senders = []
            receivers = []
            
            for shop in self.shops:
                # 該当商品のインデックスを取得
                stock_df = self.current_stock[
                    (self.current_stock['retail_store'] == shop) & 
                    (self.current_stock['item'] == item)
                ]
                current_qty = stock_df['stock_quantity'].sum()
                next_demand = self.get_expected_demand(shop, item, day + 1)
                balance = current_qty - next_demand
                
                if balance > 0:
                    # 送り手
                    valid_stock = stock_df[stock_df['remaining_shelf_life'] >= 2]
                    sendable = valid_stock['stock_quantity'].sum()
                    surplus = max(0, sendable - next_demand)
                    if surplus > 0:
                        senders.append({'shop': shop, 'qty': surplus, 'df_index': valid_stock.index.tolist()})
                        
                elif balance < 0:
                    # 受け手
                    shortage = abs(balance)
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
                    self.total_transport_cost += amount * self.transport_cost_unit
                    
                    remaining = amount
                    # 送り手の在庫を減らす（既存行の更新）
                    for idx in sender['df_index']:
                        if remaining <= 0: break
                        # ここで現在の値を取得して確認
                        have = self.current_stock.at[idx, 'stock_quantity']
                        
                        if have <= 0: continue # 既に無い場合スキップ

                        take = min(have, remaining)
                        self.current_stock.at[idx, 'stock_quantity'] -= take
                        remaining -= take
                        
                        # ★新しい行（受け手の在庫）を作成リストに追加
                        # 元の行情報をコピーして使う
                        original_row = self.current_stock.loc[idx]
                        new_row = {
                            'stock_id': self.next_stock_id,
                            'retail_store': receiver['shop'], # 店を変更
                            'item': item,
                            'stock_quantity': take, # 移動した分
                            'remaining_shelf_life': original_row['remaining_shelf_life']
                        }
                        new_transferred_stock.append(new_row)
                        self.next_stock_id += 1
                            
        # ★ループ終了後にまとめて追加（これでインデックスずれを防ぐ）
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
        
        # 処理前にインデックスリセット
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
        
        # 転送処理
        transferred = self.run_transshipment(day)

        # 廃棄 & 更新
        expired = self.current_stock['remaining_shelf_life'] <= 0
        waste_today = self.current_stock.loc[expired, 'stock_quantity'].sum()
        self.total_waste_count += waste_today
        
        # 0以下の行を削除して整理
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
    先行研究 (Chen et al., Olsson) に基づき、**「所在管理」「品質制約」「輸送コスト」** を厳密に組み込んだモデル。
    「従来型（転送なし）」と「提案手法（プロアクティブ転送）」を比較検証する。
    """)

    st.sidebar.header("条件設定")
    
    with st.sidebar.expander("① 基本設定", expanded=True):
        days = st.slider("シミュレーション期間 (日)", 10, 60, 30)
        supply_mean = st.slider("基本入荷基準値", 20, 50, 30)
        demand_std = st.slider("需要のばらつき倍率", 0.0, 2.0, 1.0)
    
    with st.sidebar.expander("② 転送・コスト設定 (先行研究)", expanded=True):
        threshold = st.slider("転送閾値 (これ以下は送らない)", 1, 10, 5)
        cost_unit = st.number_input("1個あたりの輸送コスト (円)", value=30)

    if st.sidebar.button("検証開始", type="primary"):
        scenarios = [("従来モデル", False), ("提案モデル", True)]
        results = []
        progress = st.progress(0)
        
        for i, (name, enable) in enumerate(scenarios):
            sim = RealWorldSupplySimulation(
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
        col2.metric("輸送コスト", f"{int(prop['TransportCost']):,} 円", f"@{cost_unit}円")
        col3.metric("経済効果 (損益)", f"{int(cost_saving):,} 円", "廃棄削減益 - 輸送費")

        st.subheader("日次廃棄量の推移")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(base["DailyWaste"], label="従来モデル", linestyle='--', color='gray')
        ax.plot(prop["DailyWaste"], label="提案モデル", color='red', linewidth=2)
        ax.legend()
        st.pyplot(fig)
        
        if cost_saving > 0:
            st.success(f"成功: 輸送コストを上回る廃棄削減効果を確認。実用性あり。")
        else:
            st.warning(f"課題: 廃棄は減ったが、輸送コスト過多。閾値の見直しが必要。")

if __name__ == "__main__":
    main()
