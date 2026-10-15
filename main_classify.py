import torch  # 🌟 務必放在第一行：強制搶先載入核心，避免 Windows DLL 報錯

import os
import numpy as np
from pypdf import PdfReader
from src.dataset_loader import PeerReadLoader
from src.extractor import ScientificPaperExtractor
from src.classifier import ScientificPaperClassifier
from src.llm_reviewer import LLMReviewer

def main():
    print("🚀 啟動 AutoPeer 雙軌制自動審查系統 (實驗室 GPU 火力全開版)...")
    
    loader = PeerReadLoader()
    extractor = ScientificPaperExtractor()
    classifier = ScientificPaperClassifier()
    llm_reviewer = LLMReviewer()

    # 1. 載入全部資料 (不再使用 papers[:30])
    papers = loader.load_data()
    if not papers:
        print("❌ 找不到訓練資料，系統中止。")
        return
        
    print(f"\n⏳ 正在使用 SciBERT 抽取 {len(papers)} 篇訓練論文的 773 維完整特徵...")
    print("   (提醒：即將動用 GPU 進行數千次矩陣運算，請確保 CUDA 環境正常運作 ☕)")
    
    X_train = []
    y_train = []
    
    for i, paper in enumerate(papers):
        if (i + 1) % 10 == 0:
            print(f"   ... 已處理 {i + 1}/{len(papers)} 篇")
        
        # 取得純文字與摘要 (如果 loader 沒有特別把 abstract 切出來，我們先用前 1000 個字元當錨點)
        full_text = paper.get("text", "")
        # 嘗試取得 abstract，若沒有則擷取開頭片段當作摘要錨點
        abstract = paper.get("abstract", full_text[:1000])
        
        # 💡 重點對接：呼叫 extract_core_content，傳入全文與摘要
        extracted_text, hybrid_vector = extractor.extract_core_content(
            raw_text=full_text, 
            abstract_text=abstract, 
            top_k=30
        )
        
        # 注意 hybrid_vector 的 shape 是 (1, 773)，我們取 [0] 降為 1D 放入訓練集清單
        X_train.append(hybrid_vector[0])
        y_train.append(paper["label"])
        
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print("\n⚙️ 正在訓練 SVM 分類器...")
    classifier.train(X_train, y_train, X_train, y_train)

    sample_pdf_path = "data/sample_paper.pdf"
    print(f"\n🔍 正在分析目標論文: {sample_pdf_path} ...")
    
    try:
        reader = PdfReader(sample_pdf_path)
        test_text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except Exception as e:
        print(f"❌ 讀取 {sample_pdf_path} 失敗: {e}")
        return

    # 對目標論文也執行完整的 773 維特徵抽取
    # 同樣以開頭 1000 字當作目標論文的摘要錨點
    test_abstract = test_text[:1000]
    top_sentences, test_hybrid_vector = extractor.extract_core_content(
        raw_text=test_text,
        abstract_text=test_abstract,
        top_k=30
    )
    
    # SVM 推論
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