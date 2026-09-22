# main.py
from pypdf import PdfReader
from src.extractor import ScientificPaperExtractor
import os
from openai import OpenAI

# 1. 讀取並解析 PDF 檔案 (以資工論文為例)
def load_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text

def main():
    # 初始化 OpenAI Client (請先設定環境變數 export OPENAI_API_KEY="your-key")
    # 如果你是跑本機的開源模型 (如 Llama 3 / Ollama)，可以更改 base_url
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # 載入測試論文
    pdf_path = "data/sample_paper.pdf"
    if not os.path.exists(pdf_path):
        print(f"請先將測試論文放到 {pdf_path}")
        return
        
    print("Step 1: 正在解析 PDF 全文...")
    raw_paper_text = load_pdf(pdf_path)
    
    # 【研究生技巧】通常我們要手動或用規則抓出 Abstract 作為 Anchor 
    # 這裡先示範手動複製摘要，或你可以寫個簡單的 string.find("Abstract") 來切字串
    mock_abstract = input("請貼上該論文的 Abstract 內容: ")
    
    print("\nStep 2: 正在執行兩階段抽取演算法 (CEM/TF-IDF)...")
    extractor = ScientificPaperExtractor(top_k=30) # 擷取最重要的 30 句話
    compressed_text = extractor.extract_core_content(raw_paper_text, reference_anchor=mock_abstract)
    
    print(f"➔ 成功將論文從 {len(raw_paper_text.split())} 字，壓縮至 {len(compressed_text.split())} 字。")
    
    print("\nStep 3: 正在呼叫 LLM 執行分面感知審稿 (Aspect-aware Review)...")
    system_prompt = (
        "You are an expert reviewer for IEEE/ACM conferences. Analyze the provided core content of the paper. "
        "Provide a structured review. You MUST prefix each paragraph with one of these tags: "
        "[Summary], [Originality], [Clarity], [Soundness]."
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini", # 或者使用你微調過後的本地模型
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Core Paper Content:\n{compressed_text}"}
        ],
        temperature=0.2 # 設低一點讓預測結果更穩定、不瞎掰
    )
    
    print("\n================== AI 審查結果預測 ==================\n")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    main()