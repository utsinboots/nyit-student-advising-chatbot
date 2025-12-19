from app.config import settings
from app.services.llm_service import LLMService

print("Testing LLM Service...")
print(f"API Key configured: {bool(settings.OPENAI_API_KEY)}")
print(f"Model: {settings.CHAT_MODEL}")

llm = LLMService()

try:
    response = llm.generate_response(
        query="Does NYIT offer computer science?",
        context="",
        model=settings.CHAT_MODEL
    )
    print("\n✓ LLM Response:")
    print(response['answer'])
    print(f"\nTokens: {response['tokens_used']}")
    print(f"Cost: ${response['cost']:.4f}")
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()