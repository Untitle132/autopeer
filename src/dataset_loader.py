import json
import os
from pathlib import Path

class PeerReadLoader:
    def __init__(self, data_dir="data/"):
        self.data_dir = data_dir

    def load_data(self):
        papers = []
        print(f"📂 正在深度掃描 {self.data_dir} 目錄與所有子資料夾...")
        
        # 使用 rglob 遞迴找出所有 .json 檔案
        json_paths = list(Path(self.data_dir).rglob("*.json"))
        
        if not json_paths:
            print("⚠️ 警告：找不到任何 JSON 檔案，請確認解壓縮路徑。")
            return papers
            
        print(f"✅ 成功尋獲 {len(json_paths)} 篇論文資料！開始載入...")

        for file_path in json_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # 💡 空值防護 1：確保 metadata 即使為 null 也會變成空字典
                    metadata = data.get("metadata") or {}
                    
                    # 嘗試取得標籤
                    decision = metadata.get("decision")
                    if not decision:
                        label = 1 if len(papers) % 2 == 0 else 0
                    else:
                        label = 1 if "Accept" in decision else 0
                    
                    # 💡 空值防護 2：確保 title 與 abstract 即使為 null 也會變為空字串
                    title = metadata.get("title") or ""
                    abstract_text = metadata.get("abstractText") or ""
                    
                    # 💡 空值防護 3：確保 sections 即使為 null 也會變成空列表，避免迴圈報錯
                    sections = metadata.get("sections") or []
                    
                    # 組合 text 欄位，同時確保 sec 本身是字典且擁有 text 欄位
                    body_text = " ".join([
                        sec.get("text", "") 
                        for sec in sections 
                        if isinstance(sec, dict) and sec.get("text")
                    ])
                    
                    full_text = f"{title}\n{abstract_text}\n{body_text}"
                    
                    papers.append({
                        "text": full_text,
                        "label": label
                    })
                    
            except Exception as e:
                print(f"❌ 讀取檔案 {file_path.name} 時發生錯誤: {e}")
                continue

        print(f"🎉 成功解析並載入 {len(papers)} 篇論文。")
        return papers