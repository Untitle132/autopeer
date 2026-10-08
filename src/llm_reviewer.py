import os
import google.generativeai as genai

class LLMReviewer:
    def __init__(self, api_key=None, model_name="models/gemini-3.8-flash"):
        # 優先使用傳入的 API Key，若無則讀取系統環境變數 GEMINI_API_KEY
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("❌ 找不到 API Key！請在初始化時傳入，或設定 GEMINI_API_KEY 環境變數。")
        
        # 設定 Gemini API
        genai.configure(api_key=self.api_key)
        self.model_name = model_name
        
        # 根據頂會標準設計的系統提示詞 (System Instruction)
        self.system_instruction = (
            "You are an expert reviewer for a top-tier artificial intelligence conference. "
            "Your task is to provide a highly specific, constructive, and technically deep peer review. "
            "Avoid generic praise. Base your critique strictly on the provided abstract and key sentences."
        )

    def generate_deep_review(self, abstract, top_k_sentences):
        """
        接收論文摘要與 Extractor 抽出的 Top-K 關鍵句，生成結構化深度評論
        """
        print(f"🧠 [Option B] 啟動 Gemini 深度審查，使用模型: {self.model_name}...")
        
        # 將 Top-K 句子組合成聚焦文本
        focused_context = "\n".join([f"- {text}" for text in top_k_sentences])
        
        user_prompt = f"""
        Here is the Abstract of the submitted paper:
        {abstract}
        
        Here are the most critical sentences extracted from the paper body:
        {focused_context}
        
        Please provide a structured review strictly containing the following 4 sections:
        1. Summary: Briefly summarize the core contribution.
        2. Strengths: List 1-2 specific technical strengths based on the provided text.
        3. Weaknesses: Identify potential methodological flaws, missing baselines, or logical inconsistencies.
        4. Questions / Actionable Suggestions: Provide 1-2 specific questions for the authors to address during rebuttal.
        """
        
        try:
            # 初始化 Gemini 模型
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=self.system_instruction
            )
            
            # 設定生成參數 (降低隨機性以確保評論嚴謹度)
            generation_config = genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1500
            )
            
            # 呼叫 Gemini API 產出評論
            response = model.generate_content(
                user_prompt,
                generation_config=generation_config
            )
            return response.text
            
        except Exception as e:
            return f"❌ Gemini API 呼叫失敗，請檢查 API Key 或網路連線。錯誤訊息: {e}"