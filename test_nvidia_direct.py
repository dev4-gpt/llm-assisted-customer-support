import os
from app.core.config import get_settings
from app.services.llm_client import LLMClient

def test():
    os.environ["LLM_PROFILE"] = "nvidia"
    settings = get_settings()
    print("Profile:", settings.llm_profile)
    print("Model:", settings.nvidia_model)
    client = LLMClient(settings)
    
    prompt = """
    Evaluate the response.
    Return ONLY a JSON object with this exact schema:
    { "score": 1.0, "checks": {"empathetic_tone": true, "actionable_next_step": true, "policy_safety": true, "resolved_or_escalated": true}, "coaching_feedback": "Looks good.", "flagged_phrases": [] }
    """
    
    print("Calling LLM...")
    res = client.complete_json(prompt, schema_hint="QualityResult")
    print("Result:", res)

if __name__ == "__main__":
    test()