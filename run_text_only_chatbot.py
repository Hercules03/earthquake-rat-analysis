"""
Script to run the Text-Only Earthquake RAT Chatbot
"""

from text_only_earthquake_chatbot import TextOnlyEarthquakeChatbot

def run_text_only_chatbot():
    """
    Run the text-only earthquake chatbot with sample questions.
    """
    print("Starting Text-Only Earthquake RAT Chatbot...")
    
    # Create the chatbot (uses default paths)
    chatbot = TextOnlyEarthquakeChatbot(
        # Using text files only to avoid PDF processing issues
        knowledge_base_path="./text_knowledge_base",
        # Model names based on your Ollama installation
        model_name="llama3.2:3b-instruct-q4_K_M",  # Based on your log output
        reasoning_model_name="deepseek-r1:7b",     # Based on your log output
        reflection=2,
        vector_store_path="./text_vector_store"    # Using a different store for this test
    )
    
    # Build the chatbot
    print("Building the chatbot...")
    chatbot.build()
    print("Chatbot is ready!")
    
    # Sample questions about earthquakes in Japan
    sample_questions = [
        "What causes earthquakes in Japan?",
        "What are the current approaches to earthquake prediction?",
        "How do tsunami warning systems work in Japan?"
    ]
    
    # Ask sample questions
    print("\n----- Sample Questions -----")
    for i, question in enumerate(sample_questions, 1):
        print(f"\nQuestion {i}: {question}")
        response = chatbot.ask(question)
        print(f"\nResponse: {response}")
        print("-" * 50)
    
    # Start interactive mode
    chatbot.interactive_session()

if __name__ == "__main__":
    run_text_only_chatbot()
