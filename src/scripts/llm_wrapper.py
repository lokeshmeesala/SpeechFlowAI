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
        

# Example Usage
if __name__ == "__main__":
    client = LLMClient(model="llama2")
    result = client.generate("You are a speech therapist agent that can analyses users speech and suggest exercises. First give the user a complicated paragraph to read, it should have 'h' 'm' words. Your response should be in a json format with 2 keys 'Text', 'Avg_Time(s)'. Avg_Time(s) is the average seconds needed to read the the paragraph. Ex: {'Text': 'The human brain is an incredibly intricate and mysterious organ, capable of processing an astounding amount of information in a relatively short period of time. However, it is not without its limitations, and can often struggle with certain sounds, such as the 'h' and 'm' sounds, which are commonly found in many languages. These sounds can be particularly challenging for some individuals, as they require a specific placement of the tongue and lips in order to produce them correctly. As a result, speech therapists often recommend targeted exercises to help improve the production of these sounds. One such exercise involves practicing the 'h' sound by itself, followed by practicing it in combination with other sounds, such as 'm.' By doing so, individuals can strengthen their articulatory muscles and improve their overall pronunciation.', 'Avg_Time(s)': 40}")
    print("Ollama Response:", result)
