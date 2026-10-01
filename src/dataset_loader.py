import json
import os
from pathlib import Path

class PeerReadLoader:
    def __init__(self, data_dir="data/"):
        """
        初始化資料集載入器。
        預設從專案根目錄的 data/ 資料夾開始遞迴向下尋找所有 .json 檔。
        """
        self.data_dir = data_dir

    def load_data(self):
        """
        利用 rglob 深度掃描資料夾內所有的 .json 檔案，
        並從中讀取 title, abstract, text 與決策 (accept/reject)。
        """
        papers = []
        
        print(f"📂 正在深度掃描 {self.data_dir} 目錄與所有子資料夾...")
        
        # 使用 rglob 遞迴找出所有 .json 檔案 (無論藏在多深的子目錄)
        json_paths = list(Path(self.data_dir).rglob("*.json"))
        
        if not json_paths:
            print("⚠️ 警告：找不到任何 JSON 檔案，請確認解壓縮路徑。")
            return papers
            
        print(f"✅ 成功尋獲 {len(json_paths)} 篇論文資料！開始載入...")

        for file_path in json_paths:
            try:
                # Python 的 open 可以直接接收 Path 物件
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # 嘗試從 metadata 中取得 Accept/Reject 標籤 (若無則跳過)
                    decision = data.get("metadata", {}).get("decision")
                    if not decision:
                        continue
                    
                    # 統一標籤：將 Accept 轉為 1，Reject 轉為 0
                    label = 1 if "Accept" in decision else 0
                    
                    # 提取論文內容 (依據 Science Parse 格式)
                    title = data.get("metadata", {}).get("title", "")
                    abstract_text = data.get("metadata", {}).get("abstractText", "")
                    
                    # 組合 text 欄位，處理 Science Parse 的 sections 結構
                    sections = data.get("metadata", {}).get("sections", [])
                    body_text = " ".join([sec.get("text", "") for sec in sections if sec.get("text")])
                    
                    # 組合完整文本
                    full_text = f"{title}\n{abstract_text}\n{body_text}"
                    
                    papers.append({
                        "text": full_text,
                        "label": label
                    })
                    
            except Exception as e:
                print(f"❌ 讀取檔案 {file_path.name} 時發生錯誤: {e}")
                continue

        print(f"🎉 成功解析並載入 {len(papers)} 篇有效標註論文。")
        return papers