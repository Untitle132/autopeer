import random
import numpy as np
from sklearn.model_selection import train_test_split
from src.dataset_loader import PeerReadLoader
from src.extractor import ScientificPaperExtractor
from src.classifier import ScientificPaperClassifier
from src.llm_reviewer import LLMReviewer

def main():
    print("🎲 啟動測試集盲抽腳本 (確保模型未曾看過此論文)...")
    
    loader = PeerReadLoader()
    papers = loader.load_data()
    
    if not papers:
        print("❌ 找不到資料，請確認 PeerRead 資料集已解壓縮。")
        return
        
    # 1. 過濾出有效論文 (嚴格對齊 main_classify.py 的前處理邏輯)
    valid_papers = [
        p for p in papers 
        if isinstance(p.get("text", ""), str) and len(p.get("text", "").strip()) >= 50
    ]
    y_data = [p["label"] for p in valid_papers]
    
    # 2. 以相同的亂數種子 (42) 切分，確保這裡的 test_papers 100% 等同於昨天切出的 20% 測試集
    _, test_papers = train_test_split(
        valid_papers, test_size=0.2, random_state=42, stratify=y_data
    )
    
    # 3. 隨機盲抽一篇
    sample = random.choice(test_papers)
    real_label = "✅ Accept (錄取)" if sample["label"] == 1 else "❌ Reject (拒絕)"
    
    print("\n================ 📄 抽中測試論文 ================")
    print(f"📌 論文標題: {sample.get('title', '無標題')}")
    print(f"🏷️ 真實世界結果: {real_label}")
    print(f"📝 摘要預覽: {sample.get('abstract', sample.get('text', '')[:500])[:150]}...")
    print("=================================================\n")

    print("⚙️ 正在載入昨日快取的 SVM 模型權重...")
    classifier = ScientificPaperClassifier()
    try:
        classifier.load_model(save_dir="models")
    except Exception as e:
        print("❌ 找不到 models/svm_model.pkl 快取，請先執行 python main_classify.py 訓練一次模型。")
        return

    print("⏳ 正在對盲抽論文進行 773 維特徵抽取 (SciBERT + 統計特徵)...")
    extractor = ScientificPaperExtractor()
    llm_reviewer = LLMReviewer()
    
    try:
        top_sentences, hybrid_vector = extractor.extract_core_content(
            raw_text=sample["text"], 
            abstract_text=sample.get("abstract", sample["text"][:1000]), 
            top_k=30
        )
        
        pred_class, accept_prob = classifier.predict(hybrid_vector)
        
        print("\n================ 🏆 Stage 1: SVM 快篩結果 ================")
        print(f"➔ 系統判定錄取率 (Confidence): {accept_prob * 100:.2f}%")
        
        if 0.20 <= accept_prob <= 0.80:
            print("⚠️ 錄取率落在模糊地帶，觸發 Stage 2 (Gemini 深度審查)...")
            print("========================================================\n")
            review_result = llm_reviewer.generate_review(top_sentences)
            print("\n================ 🧠 Stage 2: Gemini 審查報告 ================")
            print(review_result)
            print("=============================================================")
        else:
            decision = "✅ 直接錄取" if accept_prob > 0.80 else "❌ 直接拒絕"
            print(f"➔ 決策結果: {decision} (分數落於極端值，無需動用 LLM 審查)")
            print("========================================================\n")
            
    except Exception as e:
        print(f"❌ 抽取或預測時發生錯誤: {e}")

if __name__ == "__main__":
    main()