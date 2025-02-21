from google import genai
import json
import os
from dotenv import load_dotenv
import re

load_dotenv()

def remove_markdown_fences(text):
    # Remove the starting ```json or ``` and the ending ```
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text

class LLMAnalysis:
    def __init__(self, model_name: str = "gemini-2.0-flash-lite-preview-02-05"):
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        if self.api_key is None:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        self.model_name = model_name
        self.client = None

    def _initialize_client(self):
        """Initializes the genai client."""
        if self.api_key is None:
            raise ValueError("API key must be set before calling analyze_lyrics.")
        self.client = genai.Client(api_key=self.api_key)

    def generate_prompt(self, cleaned_lyrics: str) -> str:
        prompt = (
            "Analyze the following lyrics line by line and list the key concept(s) or themes present "
            "in each line and the subject being addressed.\n\n"
            "Data Format: Organize the output into a JSON object. For example:\n\n"
            "{\n"
            "  \"line1\": {\n"
            "    \"text\": \"first lyric\",\n"
            "    \"concepts\": [\"concept 1\", \"concept 2\"],\n"
            "    \"subject\": [\"subject 1\", \"subject 2\"]\n"
            "  },\n"
            "  \"line2\": {\n"
            "    \"text\": \"second lyric\",\n"
            "    \"concepts\": [\"concept 1\", \"concept 2\", \"concept 3\"],\n"
            "    \"subject\": [\"subject 1\"]\n"
            "  },\n"
            "  \"line3\": {\n"
            "    \"text\": \"third lyric\",\n"
            "    \"concepts\": [\"concept 1\", \"concept 2\", \"concept 3\"],\n"
            "    \"subject\": [\"subject 1\", \"subject 2\"]\n"
            "  }\n"
            "  // ... additional lines\n"
            "}\n\n"
            "Song Lyrics:\n"
            f"{cleaned_lyrics}"
        )
        return prompt

    def analyze_lyrics(self, cleaned_lyrics: str):
        try:
            self._initialize_client()
            prompt = self.generate_prompt(cleaned_lyrics)
            # Use the streaming method
            response_stream = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=[prompt]
            )

            json_buffer = ""
            for chunk in response_stream:
                clean_chunk = remove_markdown_fences(chunk.text)
                print(clean_chunk, end="", flush=True)
                json_buffer += chunk.text

            
            clean_json = remove_markdown_fences(json_buffer)
            try:
                analysis_result = json.loads(clean_json)
                return analysis_result

            except json.JSONDecodeError as e:
                print(f"\nError decoding JSON after streaming: {e}. Full Response: {json_buffer}")
                return {}
        except Exception as e:
            print(f"\nAn error occurred during analysis: {e}")
            return {}
