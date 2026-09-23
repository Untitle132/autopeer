import os
import json
import glob
import numpy as np

class PeerReadLoader:
    def __init__(self, data_root="data/PeerRead-master/data"):
        self.data_root = data_root

    def load_dataset(self, max_samples=30):
        """
        讀取 PeerRead 真實資料，回傳論文純文字、摘要與錄取標籤 (1: Accept, 0: Reject)
        """
        print(f"📂 正在掃描 PeerRead 資料夾: {self.data_root}")
        
        # 尋找所有的 review json (裡面包含錄取與否的標籤)
        review_files = glob.glob(os.path.join(self.data_root, "*", "*", "reviews", "*.json"))
        
        raw_texts = []
        abstracts = []
        labels = []
        
        valid_count = 0
        for rev_path in review_files:
            if valid_count >= max_samples:
                break
                
            # 推導對應的 parsed_pdfs 解析檔路徑
            pdf_path = rev_path.replace("reviews", "parsed_pdfs")
            if not os.path.exists(pdf_path):
                continue
                
            try:
                # 1. 獲取錄取標籤
                with open(rev_path, 'r', encoding='utf-8') as f:
                    rev_data = json.load(f)
                    is_accepted = rev_data.get("accepted", False)
                    label = 1 if is_accepted else 0
                    
                # 2. 獲取論文純文字 (Science Parse 格式)
                with open(pdf_path, 'r', encoding='utf-8') as f:
                    pdf_data = json.load(f)
                    
                    meta = pdf_data.get("metadata", {})
                    abstract = meta.get("abstractText", "No abstract available.")
                    
                    # 將分散的 sections 拼成完整純文字，保留圖表與公式編號供 Regex 抓取
                    sections = pdf_data.get("sections", meta.get("sections", []))
                    full_text = "\n".join([s.get("text", "") for s in sections if s.get("text")])
                    
                    if len(full_text) < 500: # 排除解析失敗的空檔
                        continue
                        
                    raw_texts.append(full_text)
                    abstracts.append(abstract)
                    labels.append(label)
                    valid_count += 1
                    
            except Exception:
                continue
                
        print(f"✅ 成功載入 {len(labels)} 篇真實頂會論文資料。")
        return raw_texts, abstracts, np.array(labels)