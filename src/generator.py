def generate_review(compressed_paper):
    # 融入《Can We Automate Scientific Reviewing?》的 Aspect 標籤思維
    system_prompt = (
        "You are a professional computer science conference reviewer. "
        "Review the provided paper content segment by segment. For each point you make, "
        "explicitly prefix it with one of these aspect tags: [Summary], [Originality], [Clarity], or [Soundness]."
    )
    
    user_prompt = f"Here is the core content of the paper:\n{compressed_paper}\n\nPlease generate a constructive review."
    
    # 接下來就是呼叫你的 LLM API (例如 client.chat.completions.create)
    # ...