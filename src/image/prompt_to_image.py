from together import Together
import os
import base64
import io
from PIL import Image
import re
import time
import os
from dotenv import load_dotenv

class PromptToImageGenerator:
    def __init__(self, model="black-forest-labs/FLUX.1-dev-lora", width=1024, height=768, steps=4):
        """
        Initialize the image generator with the specified model and parameters.
        
        Args:
            model (str): The model to use for image generation
            width (int): The width of the generated image
            height (int): The height of the generated image
            steps (int): Number of inference steps
        """
        self.api_key = os.environ.get("TOGETHER_API_KEY")
        self.client = Together(api_key=self.api_key)
        self.model = model
        self.width = width
        self.height = height
        self.steps = steps
        
    def create_safe_filename(self, text):
        """Convert text to a safe filename by removing invalid characters."""
        # Replace invalid filename characters with underscores
        safe_name = re.sub(r'[\\/*?:"<>|]', "_", text)
        # Limit filename length to avoid potential issues
        if len(safe_name) > 100:
            safe_name = safe_name[:100]
        return safe_name
        
    def generate_image(self, line, prompt, song_name):
        """
        Generate an image based on the provided prompt and save it.
        
        Args:
            line (str): The text content of the line (used for naming)
            prompt (str): The prompt to generate the image
            song_name (str): The name of the current song (used for folder name)
        
        Returns:
            str: Path to the saved image
        """
        try:
            start_time = time.time()
            print(f"\nGenerating image for: {line}")
            
            # Create directory for the song if it doesn't exist
            song_folder = self.create_safe_filename(song_name)
            os.makedirs(song_folder, exist_ok=True)
            
            # Create a safe filename from the line text
            filename = self.create_safe_filename(line)
            filepath = os.path.join(song_folder, f"{filename}.png")
            
            # Generate the image
            response = self.client.images.generate(
                prompt=prompt,
                model=self.model,
                width=self.width,
                height=self.height,
                steps=self.steps,
                n=1,
                response_format="b64_json",
                image_loras=[{"path":"glif/anime-blockprint-style","scale":1}]
            )
            
            # Decode and save the image
            image_data = base64.b64decode(response.data[0].b64_json)
            image = Image.open(io.BytesIO(image_data))
            image.save(filepath)
            
            elapsed_time = time.time() - start_time
            print(f"Image generated in {elapsed_time:.2f} seconds and saved to {filepath}")
            return filepath
        
        except Exception as e:
            print(f"Error generating image: {e}")
            return None
