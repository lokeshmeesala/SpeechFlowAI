import requests

class LLMClient:
    def __init__(self, model="mistral", host="http://localhost:11434"):
        self.model = model
        self.host = host

    def generate(self, prompt, max_tokens=500):
        """
        Sends a request to the Ollama API to generate a response based on the prompt.
        """
        url = f"{self.host}/api/generate"
        payload = {"model": self.model, "prompt": prompt, "max_tokens": max_tokens, "stream": False}
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama: {e}")
            return None
        

if __name__ == "__main__":
    client = LLMClient(model="llama2")
    result = client.generate("You are a speech therapist agent that can analyses users speech and suggest exercises. First give the user a complicated paragraph to read, it should have 'h' 'm' words. Keep it under 60 words.")
    print("Ollama Response:", result)