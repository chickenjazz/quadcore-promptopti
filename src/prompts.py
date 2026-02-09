INSTRUCTION_TEMPLATE = """You are an expert in English prompt optimization.

Rewrite the **Original Prompt** into an **Optimized Prompt** that improves response quality by following these rules:

1. **Clarity**: Set clear, unambiguous expectations for the responder.
2. **Precision**: Use more specific and purposeful language than the original.
3. **Concise Chain of Thought**: Include brief structural or reasoning hints that guide the responder’s thinking, without being verbose.
4. **Information Preservation**: Do not remove or alter the original intent or information.

**Original Prompt**:
{raw_prompt}

Return **only** the rewritten optimized prompt in English.
Do not include labels, explanations, or any additional notes or text.

**Optimized Prompt**:
"""


