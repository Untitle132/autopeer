import os
import numpy as np
import gradio as gr
from pypdf import PdfReader

from src.extractor import ScientificPaperExtractor
from src.classifier import ScientificPaperClassifier
from src.llm_reviewer import LLMReviewer

# 全域變數：採用延遲載入 (Lazy Loading)，直到按下按鈕才載入模型，避免啟動時卡住
extractor = None
classifier = None
reviewer = None

def init_engines():
    """初始化或喚醒三大核心引擎"""
    global extractor, classifier, reviewer
    if extractor is None:
        print("⏳ [系統初始化] 正在載入 SciBERT 特徵抽取器...")
        extractor = ScientificPaperExtractor()
    if classifier is None:
        print("⏳ [系統初始化] 正在載入 SVM 快篩分類器...")
        classifier = ScientificPaperClassifier()
        # 自動偵測並載入實驗室電腦訓練好的本地權重快取
        if os.path.exists("models"):
            try:
                if hasattr(classifier, "load_model"):
                    classifier.load_model("models")
                elif hasattr(classifier, "load"):
                    classifier.load("models")
                print("✅ 成功載入 models/ 本地模型權重！")
            except Exception as e:
                print(f"⚠️ 無法載入已存權重，將使用預設狀態 ({e})")
    if reviewer is None:
        print("⏳ [系統初始化] 正在初始化 Gemini 深度審查員...")
        reviewer = LLMReviewer()


def extract_auto(extractor_obj, text, abstract_text=""):
    """
    動態適配器：自動對接 extractor 內部的特徵萃取方法，
    確保無論回傳格式為何，皆穩定輸出 (773維向量, Top-30關鍵句)
    """
    result = None
    # 依序嘗試專案歷代使用過的方法名稱
    for method_name in ["extract_core_content", "extract_features", "_extract_statistical_features", "extract"]:
        if hasattr(extractor_obj, method_name):
            method = getattr(extractor_obj, method_name)
            try:
                result = method(text, abstract_text)
            except TypeError:
                result = method(text)
            break

    if result is None:
        raise RuntimeError("在 ScientificPaperExtractor 中找不到可用的特徵抽取函數！")

    # 動態解包防禦機制
    if isinstance(result, tuple):
        vec = result[0]
        sents = result[-1] if len(result) >= 2 else text[:1500]
    else:
        vec = result
        sents = text[:1500]

    if isinstance(sents, list):
        sents = "\n".join([str(s) for s in sents])
    elif not isinstance(sents, str):
        sents = str(text[:1500])

    return np.array(vec, dtype=np.float32).reshape(1, -1), sents


def analyze_paper(pdf_file, abstract_input, api_key_input):
    """Gradio 介面觸發的主管線：Stage 1 抽取 -> Stage 2 級聯雙閘門路由"""
    if pdf_file is None:
        return "⚠️ 尚未上傳檔案", "請先於左側上傳論文 PDF 檔案！", "", ""

    # 若使用者在網頁介面手動輸入了 API Key，動態注入環境變數
    if api_key_input and api_key_input.strip():
        os.environ["GEMINI_API_KEY"] = api_key_input.strip()

    try:
        # 1. 喚醒後台引擎
        init_engines()

        # 2. 解析 PDF 純文字
        pdf_path = pdf_file.name if hasattr(pdf_file, "name") else str(pdf_file)
        reader = PdfReader(pdf_path)
        full_text = "\n".join([page.extract_text() or "" for page in reader.pages])

        if not full_text.strip():
            return "❌ 解析失敗", "無法從該 PDF 中提取出有效文字，請確認是否為純圖片掃描檔。", "", ""

        abstract_text = abstract_input.strip() if (abstract_input and abstract_input.strip()) else full_text[:1500]

        # 3. 【Stage 1】執行 SciBERT (768維) + Regex 統計 (5維) 特徵抽取
        hybrid_vector, top_sentences = extract_auto(extractor, full_text, abstract_text)

        # 4. 【Stage 2 - Option A】本地端 SVM 毫秒級快篩預測
        try:
            pred_res = classifier.predict(hybrid_vector)
            if isinstance(pred_res, tuple):
                pred_class, accept_prob = pred_res[0], float(np.squeeze(pred_res[1]))
            else:
                accept_prob = float(np.squeeze(pred_res))
        except Exception as svm_err:
            # 防呆：若本地尚未執行過 main_classify.py 產生模型權重，自動將機率設為 50% 以平滑觸發 Stage 2
            print(f"⚠️ SVM 尚未訓練或載入失敗 ({svm_err})，自動導向模糊地帶 (50.0%)")
            accept_prob = 0.50

        prob_display = f"{accept_prob * 100:.2f}%"

        # 5. 【Stage 2 - 級聯雙閘門路由 (Cascade Dual-Gate)】
        if accept_prob > 0.80:
            gate_status = "🟢 【快篩定案：高機率錄取 (Accept)】\n信心度極高 (>80%)，直接判定通過，已省下 100% 雲端 API 成本！"
            deep_review = "✅ 本論文結構與深度語意指標優異，已由 Stage 1 本地端 SVM 分類器直接定案錄取，無須啟動大模型深度審查。"
        elif accept_prob < 0.20:
            gate_status = "🔴 【快篩定案：高機率拒絕 (Reject)】\n核心指標明顯不足 (<20%)，直接判定退稿，已省下 100% 雲端 API 成本！"
            deep_review = "❌ 本論文未達基本學術門檻，已由 Stage 1 本地端 SVM 分類器直接定案退稿，無須啟動大模型深度審查。"
        else:
            gate_status = "🟡 【觸發深度審查軌道 (Option B)】\n錄取率落入模糊地帶 (20% ~ 80%)，已自動轉交 Gemini 進行分面感知審稿！"
            
            # 確保 Reviewer 拿到最新的 API Key
            if api_key_input and api_key_input.strip():
                reviewer_instance = LLMReviewer(api_key=api_key_input.strip())
            else:
                reviewer_instance = reviewer

            # 自動適配 generate_deep_review (雙參數或單參數)
            if hasattr(reviewer_instance, "generate_deep_review"):
                try:
                    deep_review = reviewer_instance.generate_deep_review(full_text, top_sentences)
                except TypeError:
                    deep_review = reviewer_instance.generate_deep_review(top_sentences)
            else:
                deep_review = reviewer_instance.generate_review(top_sentences)

        return prob_display, gate_status, top_sentences, deep_review

    except Exception as e:
        return "❌ 執行錯誤", f"系統管線發生異常：{str(e)}", "", ""


# =========================================================
# Gradio 視覺化操作介面定義
# =========================================================
with gr.Blocks(title="AutoPeer 雙軌制自動論文審稿系統") as demo:
    gr.Markdown(
        """
        # 🎓 AutoPeer：雙軌制科學論文自動審稿與質量預測系統
        **架構流程：** Stage 1 本機端抽取 **773 維混合特徵 (SciBERT 768維 + Regex 統計 5維)** ➡️ Stage 2 **SVM 毫秒級快篩 (Option A)** 與 **Gemini 分面深度審查 (Option B)** 級聯雙閘門路由。
        """
    )

    with gr.Row():
        # 左欄：論文上傳與參數設定區
        with gr.Column(scale=1):
            pdf_input = gr.File(label="📄 步驟 1：上傳科學論文 PDF 檔案", file_types=[".pdf"])
            abstract_input = gr.Textbox(
                label="📝 步驟 2：貼上論文 Abstract（選填）",
                placeholder="若留空，系統將自動擷取 PDF 前段文字作為語意對齊錨點 (Anchor)...",
                lines=5
            )
            api_key_input = gr.Textbox(
                label="🔑 Gemini API Key（選填）",
                placeholder="若終端機已設定 $env:GEMINI_API_KEY 則可留空...",
                type="password"
            )
            submit_btn = gr.Button("🚀 啟動雙軌制審查管線", variant="primary")

        # 右欄：雙階段結果顯示區
        with gr.Column(scale=2):
            with gr.Row():
                prob_output = gr.Textbox(label="📊 Stage 1：SVM 預測錄取率 (Confidence)", interactive=False, scale=1)
                gate_output = gr.Textbox(label="🚦 級聯雙閘門路由狀態 (Cascade Gate)", interactive=False, lines=2, scale=2)

            with gr.Accordion("🔍 展開檢視：Stage 1 萃取之 Top-30 關鍵精華句 (Context)", open=False):
                extracted_output = gr.Textbox(label="SciBERT 時序恢復精華句", lines=8, interactive=False)

            review_output = gr.Markdown(label="📝 Stage 2：LLM 分面感知深度審查報告")

    # 綁定按鈕事件
    submit_btn.click(
        fn=analyze_paper,
        inputs=[pdf_input, abstract_input, api_key_input],
        outputs=[prob_output, gate_output, extracted_output, review_output]
    )

if __name__ == "__main__":
    demo.launch()