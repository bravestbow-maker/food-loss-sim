import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import urllib.request
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpInteger, PULP_CBC_CMD

# ---------------------------------------------------------
# 日本語フォント設定 (Streamlit Cloud対応)
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
# ページ設定
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="食品ロス削減モデル - 動的シミュレーション")

# ---------------------------------------------------------
# シミュレーションモデル (先行研究ロジック統合版)
# ---------------------------------------------------------
class FoodSupplySimulation:
    def __init__(self, strategy='LP', random_seed=42, 
                 demand_std_scale=1.0, waste_risk_weight=0.1,
                 shelf_life_dict=None, supply_mean=35,
                 enable_auction=False, transfer_threshold=5):
        
        self.strategy = strategy
        self.shops = ['大学会館店', 'つくば駅前店', 'ひたち野牛久店', '研究学園店']
        self.items = ['トマト', '牛乳', 'パン']
        self.rng = np.random.default_rng(random_seed)
        
        # 在庫データ (Wholesaler Inventory)
        self.current_stock = pd.DataFrame(columns=['stock_id', 'item', 'stock_quantity', 'remaining_shelf_life'])
        self.next_stock_id = 1
        
        # 記録用
        self.total_waste = 0
        self.total_sales = 0
        self.transferred_count = 0 # 店舗間転送数
        
        # パラメータ
        self.WEEKLY_DEMAND_PATTERN = [1.0, 0.9, 0.9, 1.0, 1.2, 1.4, 1.3]
        self.demand_std_scale = demand_std_scale
        self.waste_risk_weight = waste_risk_weight
        self.shelf_life_dict = shelf_life_dict if shelf_life_dict else {'トマト': 5, '牛乳': 7, 'パン': 4}
        self.supply_mean = supply_mean
        
        # ★先行研究に基づく拡張機能スイッチ
        self.enable_auction = enable_auction     # オークション再配分を行うか
        self.transfer_threshold = transfer_threshold # 転送の最小ロット(閾値) 

    def get_min_shelf_life(self, shop):
        if shop in ['大学会館店', 'つくば駅前店']: return 3
        return 1

    def add_stock(self, day):
        # 日曜入荷なし
        if (day - 1) % 7 == 6: return 
        for item in self.items:
            qty = max(0, int(self.rng.normal(self.supply_mean, 10)))
            full_life = self.shelf_life_dict[item]
            delay = int(self.rng.exponential(1.2))
            life = max(1, full_life - delay)
            if qty > 0:
                new_stock = pd.DataFrame([{
                    'stock_id': self.next_stock_id,
                    'item': item,
                    'stock_quantity': qty,
                    'remaining_shelf_life': life
                }])
                self.current_stock = pd.concat([self.current_stock, new_stock], ignore_index=True)
                self.next_stock_id += 1

    def generate_demand(self, day):
        weekday = (day - 1) % 7
        factor = self.WEEKLY_DEMAND_PATTERN[weekday]
        rows = []
        for shop in self.shops:
            scale = {'大学会館店': 1.5, 'つくば駅前店': 1.0, 'ひたち野牛久店': 0.6, '研究学園店': 0.8}[shop]
            for item in self.items:
                base = {'トマト': 8, '牛乳': 6, 'パン': 8}[item]
                std_dev = 4 * self.demand_std_scale
                qty = max(0, int(self.rng.normal(base * scale * factor, std_dev)))
                if qty > 0:
                    rows.append({'retail_store': shop, 'item': item, 'demand_quantity': qty})
        return pd.DataFrame(rows)

    def solve_lp(self, demand):
        # 既存のLPロジック (初期配分)
        stock = self.current_stock.copy()
        stock['risk'] = 1 / (stock['remaining_shelf_life'] + self.waste_risk_weight)
        prob = LpProblem("Dist_Opt", LpMinimize)
        x = {}
        for i in stock.index:
            for j in demand.index:
                if stock.at[i, 'item'] != demand.at[j, 'item']: continue
                if stock.at[i, 'remaining_shelf_life'] < self.get_min_shelf_life(demand.at[j, 'retail_store']): continue
                x[i, j] = LpVariable(f"x_{i}_{j}", 0, None, LpInteger)
        
        prob += lpSum((stock.at[i, 'stock_quantity'] - lpSum(x[i, j] for j in demand.index if (i, j) in x)) * stock.at[i, 'risk'] for i in stock.index)
        
        for i in stock.index: prob += lpSum(x[i, j] for j in demand.index if (i, j) in x) <= stock.at[i, 'stock_quantity']
        for j in demand.index: prob += lpSum(x[i, j] for i in stock.index if (i, j) in x) <= demand.at[j, 'demand_quantity']
        
        prob.solve(PULP_CBC_CMD(msg=0))
        return {(i, j): v.value() for (i, j), v in x.items() if v.value() > 0}

    def solve_fifo(self, demand, random=False):
        shipment = {}
        stock = self.current_stock.copy()
        demand_idx = demand.index.tolist()
        self.rng.shuffle(demand_idx)
        for item in self.items:
            stock_idx = stock[stock['item'] == item].index.tolist()
            if random: self.rng.shuffle(stock_idx)
            else: stock_idx.sort(key=lambda i: stock.at[i, 'remaining_shelf_life'])
            
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

    # ---------------------------------------------------------
    # ★先行研究[3]に基づくオークション型・在庫再配分 (Lateral Transshipment)
    # ---------------------------------------------------------
    def run_auction_redistribution(self, initial_shipment, demand_df):
        """
        初期配分後に、需要予測とのギャップに基づいて店舗間で在庫を融通する。
        「入札(Bid)」メカニズムを用い、必要度の高い店舗へ優先的に転送する。
        """
        # 1. 現在の配分状況を整理 (Store -> Item -> Quantity)
        store_inventory = {shop: {item: 0 for item in self.items} for shop in self.shops}
        shipment_map = {} # (stock_id, demand_id) -> qty の逆引き用などを保持したいが、簡易的に再構築
        
        # shipmentは {(stock_id, demand_id): qty} なので、これを展開して分析
        # どの店にどれだけ届く予定か集計
        for (s_id, d_id), qty in initial_shipment.items():
            shop = demand_df.at[d_id, 'retail_store']
            item = demand_df.at[d_id, 'item']
            store_inventory[shop][item] += qty

        # 2. オークション開催 (品目ごとに実施)
        adjusted_shipment = initial_shipment.copy()
        
        for item in self.items:
            sellers = [] # 余剰店舗
            buyers = []  # 不足店舗 (入札者)

            for shop in self.shops:
                allocated = store_inventory[shop][item]
                # その店のその商品の総需要
                shop_demand = demand_df[(demand_df['retail_store'] == shop) & (demand_df['item'] == item)]['demand_quantity'].sum()
                
                diff = allocated - shop_demand
                
                if diff > self.transfer_threshold: # 閾値[2]以上の余剰があれば「売り手」
                    sellers.append({'shop': shop, 'surplus': diff})
                elif diff < 0: # 不足していれば「買い手」として入札
                    # ★入札額(Bid)の計算 [3]
                    # 不足度が大きいほど、高い値を付ける (Bid = 不足数 * 緊急度係数)
                    shortage = abs(diff)
                    bid_price = shortage * 10 
                    buyers.append({'shop': shop, 'bid': bid_price, 'shortage': shortage})

            # 入札額が高い順にソート (優先権)
            buyers.sort(key=lambda x: x['bid'], reverse=True)

            # 3. 転送マッチング (Greedy)
            for buyer in buyers:
                if buyer['shortage'] <= 0: continue
                
                for seller in sellers:
                    if seller['surplus'] <= 0: continue
                    
                    # 転送量決定
                    transfer_qty = min(seller['surplus'], buyer['shortage'])
                    
                    # 閾値チェック (少量の移動はコスト倒れなのでしない)
                    if transfer_qty < self.transfer_threshold: continue

                    # ★配送計画の書き換え (Re-routing)
                    # seller向けの商品を、buyer向けに付け替える処理
                    # 簡易実装: adjusted_shipment の中から seller行きの配送を探して buyer行きとみなす
                    # (厳密なID付け替えは複雑なため、ここでは集計上の転送数を記録し、販売処理で調整する)
                    
                    self.transferred_count += transfer_qty
                    seller['surplus'] -= transfer_qty
                    buyer['shortage'] -= transfer_qty
                    
                    # Store在庫情報の更新 (販売計算用)
                    store_inventory[seller['shop']][item] -= transfer_qty
                    store_inventory[buyer['shop']][item] += transfer_qty

                    if buyer['shortage'] <= 0: break
        
        return store_inventory

    def step(self, day):
        self.add_stock(day)
        
        # 期限切れ廃棄
        expired = self.current_stock['remaining_shelf_life'] <= 0
        waste_today = self.current_stock.loc[expired, 'stock_quantity'].sum()
        self.total_waste += waste_today
        self.current_stock = self.current_stock[~expired]
        
        demand = self.generate_demand(day)
        shipment = {}
        
        # 1. 初期配分 (LP or FIFO)
        if self.strategy == 'LP':
            shipment = self.solve_lp(demand)
        elif self.strategy == 'FIFO':
            shipment = self.solve_fifo(demand, random=False)
        elif self.strategy == 'Random':
            shipment = self.solve_fifo(demand, random=True)

        # 2. ★オークション再配分 (プロアクティブ転送)
        # オークション有効時、店舗在庫(store_inv)は転送後の状態になる
        if self.enable_auction:
            final_store_inv = self.run_auction_redistribution(shipment, demand)
            # 販売計算のために簡易的に shipment を無視して store_inv を使うロジックへ
            # (既存コードの構造上、shipment辞書をベースに在庫を減らす必要があるため、ここで販売数を計算)
            
            sold_count = 0
            # 店舗ごとに「持ってる在庫」と「需要」を突き合わせて販売数を確定
            for shop in self.shops:
                for item in self.items:
                    qty_held = final_store_inv[shop][item]
                    qty_need = demand[(demand['retail_store']==shop) & (demand['item']==item)]['demand_quantity'].sum()
                    sold = min(qty_held, qty_need)
                    sold_count += sold
                    
                    # 元の大元の在庫(self.current_stock)から減らす処理
                    # どの在庫IDを減らすかは「古い順」に減らすと仮定(FIFO消費)
                    self._reduce_stock_by_item(item, sold)

            self.total_sales += sold_count

        else:
            # 従来通りの販売計算 (再配分なし)
            shipped_today = 0
            for (i, j), qty in shipment.items():
                self.current_stock.at[i, 'stock_quantity'] -= qty
                shipped_today += qty
            self.total_sales += shipped_today

        # 日付更新
        self.current_stock['remaining_shelf_life'] -= 1
        self.current_stock = self.current_stock[self.current_stock['stock_quantity'] > 0]
        
        return self.total_waste, self.total_sales, self.transferred_count

    def _reduce_stock_by_item(self, item, amount):
        """在庫IDに関わらず、指定アイテムを指定数だけ減らす (FIFO的消費)"""
        targets = self.current_stock[self.current_stock['item'] == item].sort_values('remaining_shelf_life')
        for idx in targets.index:
            if amount <= 0: break
            have = self.current_stock.at[idx, 'stock_quantity']
            reduce_val = min(have, amount)
            self.current_stock.at[idx, 'stock_quantity'] -= reduce_val
            amount -= reduce_val

# ---------------------------------------------------------
# メイン処理 UI
# ---------------------------------------------------------
def main():
    st.title("社会工学類 課題制作：食品ロス削減シミュレーション")
    st.markdown("""
    **研究目的:** 需要予測誤差による食品ロスを、**「動的在庫転送 (Lateral Transshipment)」** と **「オークション理論」** を用いて削減できるか検証する。
    """)
    
    # --- パラメータ ---
    st.sidebar.header("シミュレーション条件")
    
    with st.sidebar.expander("1. 基本設定", expanded=True):
        days = st.slider("期間 (日)", 10, 60, 30, 5)
        seed = st.number_input("乱数シード", value=42)

    with st.sidebar.expander("2. 先行研究ロジックの適用", expanded=True):
        use_auction = st.checkbox("オークション型 再配分を導入", value=True, help="先行研究[3]に基づく。店舗間で在庫を融通します。")
        threshold = st.slider("転送閾値 (Threshold)", 1, 10, 5, help="先行研究[2]に基づく。これ以下の不足/余剰では転送コストを考慮して移動しません。")

    with st.sidebar.expander("3. 需給バランス"):
        supply = st.slider("平均入荷数", 20, 60, 35)
        std_scale = st.slider("需要変動倍率", 0.0, 3.0, 1.0)
    
    # 実行
    if st.sidebar.button("シミュレーション開始", type="primary"):
        # 比較のため、今回は「LP (従来)」と「LP + Auction (提案)」を比較する形にする
        # または、ユーザーが選んだ設定で走らせる
        
        strategies = ['Random', 'LP'] # 比較対象
        results = {s: {'days': [], 'waste': [], 'sales': [], 'trans': []} for s in strategies}
        
        progress = st.progress(0)
        status = st.empty()

        for i, s in enumerate(strategies):
            # 戦略に応じてAuctionのON/OFFを切り替えるなど柔軟に設定
            # 今回は「LP」を選んだ時だけ、チェックボックスの状態を反映させる
            is_auction = use_auction if s == 'LP' else False
            
            label = f"{s} + 再配分" if (s == 'LP' and is_auction) else s
            status.text(f"計算中... モデル: {label}")
            
            sim = FoodSupplySimulation(
                strategy=s, 
                random_seed=seed,
                supply_mean=supply,
                demand_std_scale=std_scale,
                enable_auction=is_auction,
                transfer_threshold=threshold
            )
            
            for d in range(1, days + 1):
                waste, sales, trans = sim.step(d)
                results[s]['days'].append(d)
                results[s]['waste'].append(waste)
                results[s]['sales'].append(sales)
                results[s]['trans'].append(trans)
                
            progress.progress((i + 1) / len(strategies))
        
        status.empty()
        progress.empty()

        # --- 結果表示 ---
        st.subheader("分析結果")
        
        # サマリーテーブル作成
        summary_data = []
        for s in strategies:
            is_auction = use_auction if s == 'LP' else False
            name = f"{s} (再配分あり)" if (s == 'LP' and is_auction) else s
            
            w = results[s]['waste'][-1]
            sl = results[s]['sales'][-1]
            tr = sum(results[s]['trans'])
            total = w + sl
            rate = (w / total * 100) if total > 0 else 0
            
            summary_data.append([name, int(sl), int(w), f"{rate:.1f}%", int(tr)])
            
        df_res = pd.DataFrame(summary_data, columns=['モデル', '累積売上', '累積廃棄', '廃棄率', '総転送数'])
        st.table(df_res)
        
        # グラフ
        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots()
            for s in strategies:
                label = f"{s} (再配分あり)" if (s == 'LP' and use_auction) else s
                ax1.plot(results[s]['days'], results[s]['waste'], marker='o', markersize=4, label=label)
            ax1.set_title("累積廃棄数の推移")
            ax1.set_xlabel("日数")
            ax1.set_ylabel("廃棄数")
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig1)

        with col2:
            fig2, ax2 = plt.subplots()
            # 転送数の推移 (LPのみ)
            if use_auction:
                ax2.bar(results['LP']['days'], results['LP']['trans'], color='green', alpha=0.6, label="転送実行数")
                ax2.set_title("日ごとの店舗間転送数 (LPモデル)")
                ax2.set_xlabel("日数")
                ax2.legend()
                ax2.grid(True, axis='y', linestyle='--', alpha=0.6)
                st.pyplot(fig2)
            else:
                st.info("再配分OFFのため転送データはありません")

if __name__ == "__main__":
    main()
