"""
Interactive Chat Client for NYIT Chatbot
"""

import requests

API_URL = "http://localhost:8000/chat"


def chat_loop():
    print("\n" + "-" * 35)
    print("Welcome! NYIT Academic Chatbot!")
    print("Type 'quit/exit' to exit")
    print("-" * 35)
    print("\nChatbot: How can I assist you today?\n")

    while True:
        try:
            query = input("You: ").strip()

            if not query:
                continue

            if query.lower() in ("quit", "exit"):
                print("Thank you for using NYIT Academic Chatbot!")
                break

            response = requests.post(
                API_URL,
                json={
                    "query": query,
                    "include_debug": False   # no debug traces
                },
                timeout=30
            )

            if response.status_code != 200:
                print(f"Error: HTTP {response.status_code}")
                print(response.text)
                continue

            data = response.json()

            # --- Answer ---
            print("\nChatbot: " + data.get("answer", "No answer returned"))

            # --- Metadata ---
            print("\n--- Metadata ---")
            print(f"Route Used : {data.get('route_used', 'unknown')}")
            print(f"Confidence : {float(data.get('confidence', 0)):.2f}")
            print(f"Latency   : {float(data.get('latency_ms', 0)):.0f} ms\n")

        except KeyboardInterrupt:
            print("\nThank you for using NYIT Academic Chatbot!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    chat_loop()
