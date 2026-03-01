
import re

BANNED_WORDS = ["hate", "violence", "terror"]

def check_toxicity(text: str) -> bool:
    for word in BANNED_WORDS:
        if word in text.lower():
            return False
    return True

def detect_prompt_injection(text: str) -> bool:
    patterns = ["ignore previous instructions", "override system"]
    for p in patterns:
        if re.search(p, text.lower()):
            return False
    return True

def apply_guardrails(text: str) -> str:
    if not check_toxicity(text):
        return "Content blocked due to unsafe language."

    if not detect_prompt_injection(text):
        return "Prompt injection attempt detected."

    return text