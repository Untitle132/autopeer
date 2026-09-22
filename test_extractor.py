import os
import nltk
from pypdf import PdfReader
from src.extractor import ScientificPaperExtractor

def load_pdf(pdf_path):
    """ 讀取並解析 PDF 檔案 """
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text

def main():
    # 在程式一開始，確保本地有下載斷句模型資料包
    print("⏳ 正在檢查並下載 NLTK 斷句資源...")
    try:
        nltk.download('punkt_tab', quiet=True) 
        nltk.download('punkt', quiet=True)
        print("✅ NLTK 資源配置成功。")
    except Exception as e:
        print(f"⚠️ NLTK 下載失敗: {e}，請檢查網路連線。")
        
    # 確保資料夾存在
    os.makedirs("data", exist_ok=True)
    
    pdf_path = "data/sample_paper.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ 找不到測試檔案，請先將 PDF 論文放到: {pdf_path}")
        return
        
    print("Step 1: 正在解析 PDF 全文 (純本機端)...")
    raw_paper_text = load_pdf(pdf_path)
    
    print("\n--- 請輸入該論文的 Abstract (用於語意對齊錨點) ---")
    mock_abstract = input("貼上後請按 Enter: ")
    
    if not mock_abstract.strip():
        print("⚠️ 摘要不能為空！請重新執行並輸入內容。")
        return

    print("\nStep 2: 正在呼叫本地端 SciBERT 執行兩階段抽取演算法...")
    
    # 初始化新版 Extractor
    extractor = ScientificPaperExtractor()
    
    # 接收雙重回傳值：精華純文字 與 773維混合特徵向量
    compressed_text, hybrid_feature_vector = extractor.extract_core_content(
        raw_text=raw_paper_text, 
        abstract_text=mock_abstract,
        top_k=30
    )
    
    # 計算並顯示統計數據
    raw_word_count = len(raw_paper_text.split())
    compressed_word_count = len(compressed_text.split())
    compression_ratio = (1 - (compressed_word_count / raw_word_count)) * 100
    
    print("\n================== 📊 本機抽取統計結果 ==================")
    print(f"➔ 原始論文總字數: {raw_word_count} 字")
    print(f"➔ 篩選 Top-30 句後字數: {compressed_word_count} 字")
    print(f"➔ 文本壓縮率 (Data Reduction Rate): {compression_ratio:.2f}%")
    
    # 檢視準備餵給 Classifier 的混合特徵向量
    print("\n================== 🧬 快篩特徵向量狀態 (Stage 1 輸出) ==================")
    print(f"➔ 混合特徵陣列維度 (Shape): {hybrid_feature_vector.shape} (應為 1, 773)")
    print(f"➔ 統計特徵值預覽 [總字數, 文獻數, 公式數, 圖表數, Github(0或1)]:")
    print(f"   {hybrid_feature_vector[0][-5:]}")
    
    # 將抽出的 Context 儲存起來，方便肉眼審查或留給未來第二階段使用
    output_path = "data/extracted_context.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(compressed_text)
        
    print(f"\n✅ 成功！SciBERT 抽取的精華 Context 已儲存至: {output_path}")
    print("\n--- 抽取的精華片段預覽 ---")
    # 印出前 500 個字元作為預覽
    print(compressed_text[:500] + "...\n-------------------------")

if __name__ == "__main__":
    main()