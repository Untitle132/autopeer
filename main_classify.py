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

    # 1. 載入全部資料 (無筆電限制)
    papers = loader.load_data()
    if not papers:
        print("❌ 找不到訓練資料，系統中止。")
        return
        
    print(f"\n⏳ 正在使用 SciBERT 抽取 {len(papers)} 篇訓練論文的 773 維完整特徵...")
    print("   (這需要相當強大的 GPU 算力與時間，請耐心等候 ☕)")
    
    X_train = []
    y_train = []
    
    for i, paper in enumerate(papers):
        # 每處理 10 篇回報一次進度
        if (i + 1) % 10 == 0:
            print(f"   ... 已處理 {i + 1}/{len(papers)} 篇")
        
        # 💡 防呆機制 1：確保取得的 full_text 是真正的字串
        full_text = paper.get("text", "")
        if not isinstance(full_text, str) or len(full_text.strip()) < 50:
            # 如果文本太短或不是字串，直接跳過這篇「髒資料」
            continue
            
        # 💡 防呆機制 2：嘗試取得 abstract，若沒有或不是字串，則擷取開頭片段當作摘要錨點
        abstract = paper.get("abstract", "")
        if not isinstance(abstract, str) or not abstract.strip():
            abstract = full_text[:1000]
        
        try:
            # 呼叫完整的特徵抽取 (包含 SciBERT 語意 + 統計)，取得 773 維特徵
            extracted_text, hybrid_vector = extractor.extract_core_content(
                raw_text=full_text, 
                abstract_text=abstract, 
                top_k=30
            )
            
            # 將 1D 特徵陣列加入訓練集
            X_train.append(hybrid_vector[0])
            y_train.append(paper["label"])
            
        except Exception as e:
            # 💡 防呆機制 3：如果抽取過程中這篇論文報錯，把它印出來但不要讓整個程式崩潰
            print(f"   ⚠️ 警告：跳過第 {i+1} 篇論文，特徵抽取時發生錯誤: {e}")
            continue
        
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    if len(X_train) == 0:
        print("❌ 所有論文特徵抽取皆失敗，無法進行訓練。")
        return

    print(f"\n⚙️ 正在使用 {len(X_train)} 筆有效資料訓練 SVM 分類器...")
    # 這裡示範全部當作訓練集
    classifier.train(X_train, y_train, X_train, y_train)

    sample_pdf_path = "data/sample_paper.pdf"
    print(f"\n🔍 正在分析目標論文: {sample_pdf_path} ...")
    
    try:
        reader = PdfReader(sample_pdf_path)
        test_text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except Exception as e:
        print(f"❌ 讀取 {sample_pdf_path} 失敗: {e}")
        return

    # 對目標論文執行完整的 773 維特徵抽取
    test_abstract = test_text[:1000]
    top_sentences, test_hybrid_vector = extractor.extract_core_content(
        raw_text=test_text,
        abstract_text=test_abstract,
        top_k=30
    )
    
    # SVM 推論預測
    pred_class, accept_prob = classifier.predict(test_hybrid_vector)
    
    print("\n================ 🏆 Stage 1: SVM 快篩結果 ================")
    print(f"➔ 系統判定錄取率 (Confidence): {accept_prob * 100:.2f}%")
    
    # 模糊地帶判斷
    if 0.20 <= accept_prob <= 0.80:
        print("⚠️ 錄取率落在模糊地帶，觸發 Stage 2 (Gemini 深度審查)...")
        print("========================================================\n")
        
        print("⏳ 正在呼叫 Gemini 生成結構化深度評論...")
        # 💡 修正點：將 generate_review 改為 generate_deep_review，並同時傳入 test_text 與 top_sentences
        review_result = llm_reviewer.generate_deep_review(test_text, top_sentences)
        
        print("\n================ 🧠 Stage 2: Gemini 審查報告 ================")
        print(review_result)
        print("=============================================================")
    else:
        decision = "✅ 直接錄取" if accept_prob > 0.80 else "❌ 直接拒絕"
        print(f"➔ 決策結果: {decision} (分數落於極端值，無需動用 LLM 審查)")
        print("========================================================\n")

if __name__ == "__main__":
    main()