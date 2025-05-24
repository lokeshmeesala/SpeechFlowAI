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
    result = client.generate("""You are a speech therapist agent that helps assess speech fluency. Your task is to generate a paragraph that challenges the user’s fluency, articulation, and rhythm. The paragraph must meet the following conditions:

- Length: Between 30 and 40 words (strictly no more than 40)
- Content: Include multisyllabic words, natural pausing points, and flowing connected speech
- Tone: Use realistic and natural language, as seen in clinical assessments like the Rainbow or Grandfather Passage
- Output: Do NOT include any introductions, explanations, or concluding statements — ONLY the paragraph content
- Clarity: No lists or markdown formatting — return plain text only

Examples:

1. The highway was hidden behind the hills, humming with heavy morning traffic. Michael hesitated before merging, holding the wheel with both hands as his heart hammered in his chest.

2. She silently slid the silver spoon into the steaming stew, savoring the scent of simmering spices while listening to soft static from the speakers in the corner.

3. Olivia organized an orchestra of octopuses for an outrageous ocean opera. The melody murmured through the massive marina, mesmerizing the murmuring masses on the mossy pier.

Now generate a new paragraph like this.""")
    print("Ollama Response:", result)