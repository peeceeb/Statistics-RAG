
from backend.services.chat_service import generate_chat_response


response = generate_chat_response(
    "Explain regression analysis."
)

print("\nANSWER:")
print(response["answer"])

print("\nSOURCES:")
for source in response["sources"]:
    print(source)
