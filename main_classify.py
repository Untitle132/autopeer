import torch  # 強制搶先載入核心，避免 Windows DLL 報錯

import os
import numpy as np
import json
from pypdf import PdfReader
from sklearn.model_selection import train_test_split
import joblib

from src.dataset_loader import PeerReadLoader
from src.extractor import ScientificPaperExtractor
from src.classifier import ScientificPaperClassifier
from src.llm_reviewer import LLMReviewer

# 定義快取檔案的路徑
CACHE_DIR = "models"
META_FILE = os.path.join(CACHE_DIR, "dataset_meta.json")

def main():
    print("🚀 啟動 AutoPeer 雙軌制自動審查系統 (實驗室快取優化版)...")
    
    loader = PeerReadLoader()
    extractor = ScientificPaperExtractor()
    classifier = ScientificPaperClassifier()
    llm_reviewer = LLMReviewer()

    os.makedirs(CACHE_DIR, exist_ok=True)

    # ==========================================
    # 1. 載入並檢查資料集狀態
    # ==========================================
    papers = loader.load_data()
    if not papers:
        print("❌ 找不到訓練資料，系統中止。")
        return
    
    current_dataset_size = len(papers)
    print(f"\n📂 目前偵測到 {current_dataset_size} 篇有效訓練論文。")

    # 檢查是否有舊的 metadata 紀錄
    need_retrain = True
    if os.path.exists(META_FILE) and os.path.exists(os.path.join(CACHE_DIR, "svm_model.pkl")):
        with open(META_FILE, 'r') as f:
            meta = json.load(f)
            # 簡易判定：若資料集總篇數與上次相同，則視為未更動
            if meta.get("dataset_size") == current_dataset_size:
                print("✅ 偵測到資料集無更動，準備載入舊有模型權重...")
                need_retrain = False
            else:
                print("🔄 偵測到資料集已更動，準備重新訓練模型...")

    # ==========================================
    # 2. 訓練與測試資料準備 (按需執行)
    # ==========================================
    if need_retrain:
        print("\n⏳ 正在抽取特徵 (這需要 GPU 算力與時間，請耐心等候 ☕)...")
        
        X_data = []
        y_data = []
        
        for i, paper in enumerate(papers):
            if (i + 1) % 10 == 0:
                print(f"   ... 已處理 {i + 1}/{current_dataset_size} 篇")
            
            full_text = paper.get("text", "")
            if not isinstance(full_text, str) or len(full_text.strip()) < 50:
                continue
                
            abstract = paper.get("abstract", "")
            if not isinstance(abstract, str) or not abstract.strip():
                abstract = full_text[:1000]
            
            try:
                # 取得 773 維特徵
                extracted_text, hybrid_vector = extractor.extract_core_content(
                    raw_text=full_text, 
                    abstract_text=abstract, 
                    top_k=30
                )
                X_data.append(hybrid_vector[0])
                y_data.append(paper["label"])
            except Exception as e:
                continue
            
        X_data = np.array(X_data)
        y_data = np.array(y_data)

        if len(X_data) == 0:
            print("❌ 特徵抽取皆失敗，無法進行訓練。")
            return

        # ✨ 切分訓練集 (80%) 與測試集 (20%)
        print(f"\n✂️ 將 {len(X_data)} 筆有效資料切分為 80% 訓練集與 20% 測試集...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
        )

        print("⚙️ 正在訓練並評估 SVM 分類器...")
        classifier.train(X_train, y_train, X_test, y_test)

        # ✨ 儲存模型與 metadata
        classifier.save_model(save_dir=CACHE_DIR)
        with open(META_FILE, 'w') as f:
            json.dump({"dataset_size": current_dataset_size}, f)
        print("💾 模型與權重已更新並儲存！")
    
    else:
        # 直接載入舊模型
        classifier.load_model(save_dir=CACHE_DIR)

    # ==========================================
    # 3. 分析目標論文 (sample_paper.pdf)
    # ==========================================
    sample_pdf_path = "data/sample_paper.pdf"
    print(f"\n🔍 正在分析目標論文: {sample_pdf_path} ...")
    
    try:
        reader = PdfReader(sample_pdf_path)
        test_text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except Exception as e:
        print(f"❌ 讀取 {sample_pdf_path} 失敗: {e}")
        return

    test_abstract = test_text[:1000]
    top_sentences, test_hybrid_vector = extractor.extract_core_content(
        raw_text=test_text,
        abstract_text=test_abstract,
        top_k=30
    )
    
    # 預測錄取率
    pred_class, accept_prob = classifier.predict(test_hybrid_vector)
    
    print("\n================ 🏆 Stage 1: SVM 快篩結果 ================")
    print(f"➔ 系統判定錄取率 (Confidence): {accept_prob * 100:.2f}%")
    
    # 模糊地帶判斷
    if 0.20 <= accept_prob <= 0.80:
        print("⚠️ 錄取率落在模糊地帶，觸發 Stage 2 (Gemini 深度審查)...")
        print("========================================================\n")
        
        print("⏳ 正在呼叫 Gemini 生成結構化深度評論...")
        review_result = llm_reviewer.generate_review(top_sentences)
        
        print("\n================ 🧠 Stage 2: Gemini 審查報告 ================")
        print(review_result)
        print("=============================================================")
    else:
        decision = "✅ 直接錄取" if accept_prob > 0.80 else "❌ 直接拒絕"
        print(f"➔ 決策結果: {decision} (分數落於極端值，無需動用 LLM 審查)")
        print("========================================================\n")

if __name__ == "__main__":
    main()