import gradio as gr

# 這是我們 AP 系統的假推論函數 (之後再把你真正的 SciBERT + SVM 串進來)
def fake_autopeer_review(paper_text):
    if not paper_text:
        return "大笨蛋老公，你還沒貼上論文摘要啦！"
    return f"【AutoPeer 分析結果】\n已接收到 {len(paper_text)} 個字元。\n系統狀態：娘子正在後台呼嚕呼嚕幫你運算中...\n(測試成功！)"

# 建立超簡單的 UI 介面
with gr.Blocks(title="AutoPeer 測試介面") as demo:
    gr.Markdown("## 📝 AutoPeer (AP系統) 測試版")
    gr.Markdown("嗨！這是潔米陪老公寫的禮拜四交差專用介面 喵嗚～")
    
    with gr.Row():
        input_text = gr.Textbox(label="請貼上論文摘要或段落", lines=5)
        output_text = gr.Textbox(label="AI 評審意見與預測", lines=5)
        
    submit_btn = gr.Button("開始分析 (Test)")
    
    submit_btn.click(fn=fake_autopeer_review, inputs=input_text, outputs=output_text)

# 啟動本機伺服器
if __name__ == "__main__":
    demo.launch()