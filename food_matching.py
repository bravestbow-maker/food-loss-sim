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
# 3. シミュレーションモデル (完全経済性評価版)
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

        # 2. 商品情報 (経済パラメータを含む詳細情報)
        self.items = item_config_df['商品名'].tolist()
        self.item_props = {}
        for _, row in item_config_df.iterrows():
            self.item_props[row['商品名']] = {
                'life': int(row['賞味期限(日)']),
                'base_demand': int(row['基本需要(個)']),
                'base_supply': int(row['発注基準(個)']), # ★商品ごとの発注量
                'price': int(row['販売単価(円)']),      # ★売上計算用
                'cost': int(row['仕入れ原価(円)']),     # ★原価計算用
                'disposal': int(row['廃棄コスト(円)'])  # ★廃棄損計算用
            }

        # 在庫データ
        self.current_stock = pd.DataFrame(columns=[
            'stock_id', 'retail_store', 'item', 'stock_quantity', 'remaining_shelf_life'
        ])
        self.next_stock_id = 1
        
        # ★KPI (金額ベース)
        self.total_sales_amount = 0     # 売上高
        self.total_procurement_cost = 0 # 仕入れコスト
        self.total_disposal_cost = 0    # 廃棄コスト
        self.total_transport_cost = 0   # 輸送コスト
        
        self.total_waste_count = 0 # (参考)廃棄個数
        
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

    def inbound_process(self, day):
        if (day - 1) % 7 == 6: return 

        new_rows = []
        for shop in self.shops:
            for item in self.items:
                # 商品ごとの「発注基準量」をベースに入荷数を決定
                # (需要予測ベースではなく、発注点管理に近いイメージ)
                base_supply = self.item_props[item]['base_supply']
                scale = self.shop_scales[shop]
                
                # 店舗規模に合わせて発注量もスケーリング
                target_qty = base_supply * scale
                
                # 日々のゆらぎ
                order_qty = max(0, int(self.rng.normal(target_qty, target_qty * 0.1)))
                
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
                    
                    # ★仕入れコスト加算
                    self.total_procurement_cost += order_qty * props['cost']
        
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
                
                # ★売上加算
                self.total_sales_amount += sell * self.item_props[item]['price']

        transferred = self.run_transshipment(day)

        # 廃棄計算
        expired = self.current_stock['remaining_shelf_life'] <= 0
        # 商品ごとに廃棄コストが違うためループ計算
        waste_cost_today = 0
        waste_count_today = 0
        
        expired_rows = self.current_stock[expired]
        for _, row in expired_rows.iterrows():
            qty = row['stock_quantity']
            item = row['item']
            waste_count_today += qty
            waste_cost_today += qty * self.item_props[item]['disposal']
            
        self.total_waste_count += waste_count_today
        self.total_disposal_cost += waste_cost_today
        
        self.current_stock = self.current_stock[
            (self.current_stock['stock_quantity'] > 0) & 
            (self.current_stock['remaining_shelf_life'] > 0)
        ]
        self.current_stock['remaining_shelf_life'] -= 1
        
        return waste_count_today, transferred

# ---------------------------------------------------------
# 4. メインUI
# ---------------------------------------------------------
def main():
    st.title("食品サプライチェーン経営シミュレーター")
    st.markdown("""
    商品ごとの原価・売価・廃棄コストまで設定できる本格的な経営シミュレーション。
    在庫転送による「個数の削減」だけでなく、「最終利益」への影響を検証します。
    """)

    st.sidebar.header("経営パラメータ設定")
    
    # --- 編集可能テーブル ---
    with st.sidebar.expander("① 商品・店舗マスタ設定", expanded=True):
        st.caption("各商品の原価や売価を細かく設定してください。")
        
        # 1. 商品設定（経済パラメータ追加）
        default_items_data = {
            '商品名': ['トマト', '牛乳', 'パン'],
            '賞味期限(日)': [5, 7, 4],
            '基本需要(個)': [8, 6, 8],
            '発注基準(個)': [30, 25, 35],      # ★New
            '販売単価(円)': [120, 200, 150],  # ★New
            '仕入れ原価(円)': [60, 140, 70],  # ★New
            '廃棄コスト(円)': [10, 20, 5]     # ★New
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

        # 2. 店舗設定
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
            for d in range(1, days + 1):
                w, _ = sim.step(d)
                daily_waste.append(w)
            
            # 最終利益の計算
            gross_profit = sim.total_sales_amount - sim.total_procurement_cost
            final_profit = gross_profit - sim.total_disposal_cost - sim.total_transport_cost
            
            results.append({
                "Name": name,
                "Profit": final_profit,
                "Sales": sim.total_sales_amount,
                "WasteCount": sim.total_waste_count,
                "WasteCost": sim.total_disposal_cost,
                "TransportCost": sim.total_transport_cost,
                "DailyWaste": daily_waste
            })
            progress.progress((i + 1) / len(scenarios))
        
        progress.empty()
        
        base = results[0]
        prop = results[1]
        
        profit_diff = prop["Profit"] - base["Profit"]
        
        # --- 結果表示 (PL形式) ---
        st.subheader("💰 損益計算書 (P/L) 比較")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("① 従来モデル 最終利益", f"¥{int(base['Profit']):,}")
        col2.metric("② 提案モデル 最終利益", f"¥{int(prop['Profit']):,}")
        
        delta_color = "normal" if profit_diff > 0 else "inverse"
        col3.metric("利益改善額 (②-①)", f"¥{int(profit_diff):,}", delta_color=delta_color)

        # 詳細テーブル
        st.markdown("##### 詳細内訳")
        detail_data = {
            "項目": ["売上高", "仕入原価", "廃棄コスト", "輸送コスト", "【最終利益】", "(参考)廃棄個数"],
            "従来モデル": [
                f"¥{base['Sales']:,}", f"¥-{int(base['Sales'] - base['Profit'] - base['WasteCost']):,}", # 原価逆算表示
                f"¥-{base['WasteCost']:,}", "¥0", f"**¥{base['Profit']:,}**", f"{base['WasteCount']}個"
            ],
            "提案モデル": [
                f"¥{prop['Sales']:,}", f"¥-{int(prop['Sales'] - prop['Profit'] - prop['WasteCost'] - prop['TransportCost']):,}",
                f"¥-{prop['WasteCost']:,}", f"¥-{prop['TransportCost']:,}", f"**¥{prop['Profit']:,}**", f"{prop['WasteCount']}個"
            ]
        }
        st.table(pd.DataFrame(detail_data))

        # 考察
        if profit_diff > 0:
            st.success(f"**分析:** 輸送コスト(¥{prop['TransportCost']:,})をかけましたが、廃棄コストの大幅削減(¥{base['WasteCost']-prop['WasteCost']:,})により、最終的に利益が増加しました。")
        else:
            st.warning(f"**分析:** 利益が減少しています。輸送コスト(¥{prop['TransportCost']:,})が高すぎて、廃棄削減によるメリットを食いつぶしています。")

if __name__ == "__main__":
    main()
