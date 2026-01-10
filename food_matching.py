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
# 3. シミュレーションモデル (実用運用版)
# ---------------------------------------------------------
class RealWorldSupplySimulation:
    def __init__(self, 
                 random_seed=42, 
                 demand_std_scale=1.0, 
                 supply_mean=35,
                 enable_transshipment=False, # 転送を行うか？
                 transport_threshold=5,      # 何個以上なら送るか？(閾値)
                 transport_cost_unit=10):    # 1個あたりの輸送コスト(円)
        
        self.shops = ['大学会館店', 'つくば駅前店', 'ひたち野牛久店', '研究学園店']
        self.items = ['トマト', '牛乳', 'パン']
        self.rng = np.random.default_rng(random_seed)
        
        # ★重要変更点1: 在庫データに「retail_store」を追加 (所在管理)
        self.current_stock = pd.DataFrame(columns=[
            'stock_id', 'retail_store', 'item', 'stock_quantity', 'remaining_shelf_life'
        ])
        self.next_stock_id = 1
        
        # KPI管理
        self.total_waste_count = 0
        self.total_sales_count = 0
        self.total_transport_cost = 0 # 輸送コスト累計(円)
        
        # 設定パラメータ
        self.WEEKLY_DEMAND_PATTERN = [1.0, 0.9, 0.9, 1.0, 1.2, 1.4, 1.3]
        self.demand_std_scale = demand_std_scale
        self.shelf_life_dict = {'トマト': 5, '牛乳': 7, 'パン': 4}
        self.supply_mean = supply_mean
        
        # 先行研究パラメータ
        self.enable_transshipment = enable_transshipment
        self.transport_threshold = transport_threshold
        self.transport_cost_unit = transport_cost_unit

    # 需要期待値の計算 (予測に使用)
    def get_expected_demand(self, shop, item, day):
        weekday = (day - 1) % 7
        factor = self.WEEKLY_DEMAND_PATTERN[weekday]
        scale = {'大学会館店': 1.5, 'つくば駅前店': 1.0, 'ひたち野牛久店': 0.6, '研究学園店': 0.8}[shop]
        base = {'トマト': 8, '牛乳': 6, 'パン': 8}[item]
        return base * scale * factor

    # 朝の入荷処理 (各店舗へ個別に納品)
    def inbound_process(self, day):
        if (day - 1) % 7 == 6: return # 日曜は入荷なし

        for shop in self.shops:
            for item in self.items:
                # 発注量の決定 (需要予測 + 少しの余分) - ここで「誤差」が生まれる
                expected = self.get_expected_demand(shop, item, day)
                # 平均入荷量パラメータを加味
                order_qty = max(0, int(self.rng.normal(expected * (self.supply_mean/30), 5)))
                
                if order_qty > 0:
                    full_life = self.shelf_life_dict[item]
                    delay = int(self.rng.exponential(1.0)) # 配送遅延
                    life = max(1, full_life - delay)
                    
                    self._add_stock_record(shop, item, order_qty, life)

    def _add_stock_record(self, shop, item, qty, life):
        new_row = {
            'stock_id': self.next_stock_id,
            'retail_store': shop, # ★店舗を指定して追加
            'item': item,
            'stock_quantity': qty,
            'remaining_shelf_life': life
        }
        self.current_stock = pd.concat([self.current_stock, pd.DataFrame([new_row])], ignore_index=True)
        self.next_stock_id += 1

    # ---------------------------------------------------------
    # ★先行研究に基づく「プロアクティブ転送」ロジック
    # ---------------------------------------------------------
    def run_transshipment(self, day):
        if not self.enable_transshipment: return 0
        
        transferred_count = 0
        
        for item in self.items:
            # 1. 送り手(Sender)と受け手(Receiver)の探索
            senders = []
            receivers = []
            
            for shop in self.shops:
                # 現在の在庫
                stock_df = self.current_stock[
                    (self.current_stock['retail_store'] == shop) & 
                    (self.current_stock['item'] == item)
                ]
                current_qty = stock_df['stock_quantity'].sum()
                
                # 明日の需要予測 (Proactive: 未来を見る)
                next_demand = self.get_expected_demand(shop, item, day + 1)
                
                balance = current_qty - next_demand
                
                if balance > 0:
                    # 余剰あり: 送り手候補
                    # ★重要制約: 賞味期限が残り2日未満のものは送らない (輸送リスク)
                    valid_stock = stock_df[stock_df['remaining_shelf_life'] >= 2]
                    sendable = valid_stock['stock_quantity'].sum()
                    
                    # 自分の明日の分は確保
                    surplus = max(0, sendable - next_demand)
                    
                    if surplus > 0:
                        senders.append({'shop': shop, 'qty': surplus, 'df_index': valid_stock.index})
                        
                elif balance < 0:
                    # 不足あり: 受け手候補
                    shortage = abs(balance)
                    urgency = shortage / (next_demand + 1)
                    receivers.append({'shop': shop, 'qty': shortage, 'urgency': urgency})

            # 2. マッチング (緊急度が高い順)
            receivers.sort(key=lambda x: x['urgency'], reverse=True)
            senders.sort(key=lambda x: x['qty'], reverse=True)
            
            for receiver in receivers:
                for sender in senders:
                    if sender['qty'] <= 0 or receiver['qty'] <= 0: continue
                    
                    amount = min(sender['qty'], receiver['qty'])
                    
                    # ★重要制約: 閾値 (Threshold)
                    # まとまった量でなければ輸送コスト倒れになるので送らない
                    if amount < self.transport_threshold: continue
                    
                    # 転送実行
                    transferred_count += amount
                    sender['qty'] -= amount
                    receiver['qty'] -= amount
                    
                    # コスト加算
                    self.total_transport_cost += amount * self.transport_cost_unit
                    
                    # データ更新 (所在地の書き換え)
                    remaining = amount
                    for idx in sender['df_index']:
                        if remaining <= 0: break
                        have = self.current_stock.at[idx, 'stock_quantity']
                        
                        if have <= remaining:
                            self.current_stock.at[idx, 'retail_store'] = receiver['shop']
                            remaining -= have
                        else:
                            # 分割処理
                            self.current_stock.at[idx, 'stock_quantity'] -= remaining
                            new_row = self.current_stock.loc[idx].copy()
                            new_row['stock_quantity'] = remaining
                            new_row['retail_store'] = receiver['shop']
                            new_row['stock_id'] = self.next_stock_id
                            self.next_stock_id += 1
                            self.current_stock = pd.concat([self.current_stock, pd.DataFrame([new_row])], ignore_index=True)
                            remaining = 0
                            
        return transferred_count

    # 1日のステップ実行
    def step(self, day):
        # 1. 朝: 入荷
        self.inbound_process(day)
        
        # 2. 日中: 販売
        sold_today = 0
        demand_rows = []
        for shop in self.shops:
            for item in self.items:
                expected = self.get_expected_demand(shop, item, day)
                qty = max(0, int(self.rng.normal(expected, 4 * self.demand_std_scale)))
                if qty > 0:
                    demand_rows.append({'shop': shop, 'item': item, 'qty': qty})
        
        # 需要に対して在庫を引き当てる (FIFO)
        for d in demand_rows:
            shop, item, need = d['shop'], d['item'], d['qty']
            targets = self.current_stock[
                (self.current_stock['retail_store'] == shop) & 
                (self.current_stock['item'] == item)
            ].sort_values('remaining_shelf_life')
            
            for idx, stock in targets.iterrows():
                if need <= 0: break
                if stock['remaining_shelf_life'] < 1: continue # 期限切れは売れない
                
                have = stock['stock_quantity']
                sell = min(need, have)
                self.current_stock.at[idx, 'stock_quantity'] -= sell
                sold_today += sell
                need -= sell

        self.total_sales_count += sold_today

        # 3. 夕方: 店舗間転送
        transferred = self.run_transshipment(day)

        # 4. 夜: 廃棄 & 日付更新
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
    先行研究 (Chen et al., Olsson) に基づき、**「所在管理」「品質制約」「輸送コスト」** を厳密に組み込んだモデル。
    「従来型（転送なし）」と「提案手法（プロアクティブ転送）」を比較検証する。
    """)

    # --- パラメータ設定 ---
    st.sidebar.header("条件設定")
    
    with st.sidebar.expander("① 基本設定", expanded=True):
        days = st.slider("シミュレーション期間 (日)", 10, 60, 30)
        supply_mean = st.slider("基本入荷基準値", 20, 50, 30)
        demand_std = st.slider("需要のばらつき倍率", 0.0, 2.0, 1.0)
    
    with st.sidebar.expander("② 転送・コスト設定 (先行研究)", expanded=True):
        threshold = st.slider("転送閾値 (これ以下は送らない)", 1, 10, 5, help="Olssonの閾値制御")
        cost_unit = st.number_input("1個あたりの輸送コスト (円)", value=30, help="人件費・燃料費")

    if st.sidebar.button("検証開始", type="primary"):
        # 2つのシナリオを比較実行
        scenarios = [
            ("従来モデル (転送なし)", False),
            ("提案モデル (転送あり)", True)
        ]
        
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
                "Sales": sim.total_sales_count,
                "TransportCost": sim.total_transport_cost,
                "DailyWaste": daily_waste
            })
            progress.progress((i + 1) / len(scenarios))
        
        progress.empty()
        
        # --- 結果表示 ---
        base = results[0]
        prop = results[1]
        
        # 1. 廃棄削減効果
        waste_diff = base["Waste"] - prop["Waste"]
        rate = (waste_diff / base["Waste"] * 100) if base["Waste"] > 0 else 0
        
        # 2. 経済効果 (簡易計算: 廃棄損1個100円 - 輸送費)
        WASTE_COST_PER_UNIT = 100 # 廃棄単価(円)
        base_loss = base["Waste"] * WASTE_COST_PER_UNIT
        prop_loss = (prop["Waste"] * WASTE_COST_PER_UNIT) + prop["TransportCost"]
        cost_saving = base_loss - prop_loss

        col1, col2, col3 = st.columns(3)
        col1.metric("廃棄削減数", f"▲{int(waste_diff)}個", f"{rate:.1f}% 削減")
        col2.metric("輸送コスト発生", f"{int(prop['TransportCost']):,} 円", f"単価 {cost_unit}円")
        col3.metric("最終経済効果 (損益改善)", f"{int(cost_saving):,} 円", 
                    help="廃棄コスト(100円/個)の削減分から、輸送コストを引いた実質利益")

        # グラフ
        st.subheader("日次廃棄量の推移")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(base["DailyWaste"], label="従来モデル", linestyle='--', color='gray')
        ax.plot(prop["DailyWaste"], label="提案モデル", color='red', linewidth=2)
        ax.set_ylabel("廃棄数 (個)")
        ax.set_xlabel("経過日数")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # 考察の自動生成
        if cost_saving > 0:
            st.success(f"""
            **検証成功:** 輸送コスト({prop['TransportCost']:,}円)をかけて在庫を転送した結果、
            それを上回る廃棄コスト削減効果が得られました。
            Olssonの研究にある通り、適切な閾値({threshold}個)の設定が利益最大化に貢献しています。
            """)
        else:
            st.error(f"""
            **検証課題:** 廃棄は減りましたが、輸送コストがかかりすぎて経済的にはマイナスです。
            「転送閾値」をもっと上げるか、より効率的な配送計画が必要です。
            """)

if __name__ == "__main__":
    main()
