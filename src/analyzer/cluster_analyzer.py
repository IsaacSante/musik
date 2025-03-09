from google import genai
import os
import random
import re
import asyncio
import threading

class ClusterImagePromptGenerator:
    """
    Generates text-to-image prompts for clusters of song lyrics.
    Works with the LyricEmbeddingPipeline to create appropriate
    visual prompts based on lyric themes and subjects.
    """
    
    def __init__(self, model_name="gemini-2.0-flash-lite-preview-02-05"):
        """Initialize the image prompt generator."""
        self.model_name = model_name
        self.cluster_prompts = {}  # Maps cluster_id to list of prompts
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        self.client = None
        self.lock = threading.Lock()  # Add a lock for thread safety
        print("Initialized ClusterImagePromptGenerator")
        
    def _initialize_client(self):
        """Initialize the LLM client if not already done."""
        if self.client is None:
            # Create an event loop for this thread if one doesn't exist
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                # No event loop exists in this thread, create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            # Now initialize the client
            self.client = genai.Client(api_key=self.api_key)
    
    def generate_prompts_for_cluster(self, cluster_id, cluster_data):
        """Generate image prompts for a specific cluster."""
        try:
            self._initialize_client()
            
            # Extract cluster information
            lyrics = cluster_data.get('lyrics', [])
            concepts = cluster_data.get('concepts', [])
            subjects = cluster_data.get('subjects', [])
            
            # Don't process empty clusters
            if not lyrics or len(lyrics) < 2:
                return []
                
            # Create the prompt for the LLM
            llm_prompt = self._create_prompt_generation_prompt(lyrics, concepts, subjects)
            
            # Get suggestions from the LLM
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[llm_prompt]
            )
            
            # Parse the response to extract the prompts
            image_prompts = self._parse_prompt_response(response.text)
            
            # Store the prompts for this cluster with thread safety
            with self.lock:
                self.cluster_prompts[cluster_id] = image_prompts
            
            return image_prompts
            
        except Exception as e:
            print(f"Error generating prompts for cluster {cluster_id}: {e}")
            import traceback
            traceback.print_exc()  # Print stack trace for debugging
            return []
    
    def _create_prompt_generation_prompt(self, lyrics, concepts, subjects):
        """Create a prompt for the LLM to generate image prompts."""
        # Count frequency of concepts and subjects
        concept_counts = {}
        for concept in concepts:
            concept_counts[concept] = concept_counts.get(concept, 0) + 1
            
        subject_counts = {}
        for subject in subjects:
            subject_counts[subject] = subject_counts.get(subject, 0) + 1
            
        # Get top concepts and subjects
        top_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_subjects = sorted(subject_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Format as strings
        concept_str = ", ".join([c for c, _ in top_concepts])
        subject_str = ", ".join([s for s, _ in top_subjects])
        
        # Select a few representative lyrics (up to 3)
        sample_lyrics = lyrics[:min(3, len(lyrics))]
        lyric_str = "\n".join([f"- \"{lyric}\"" for lyric in sample_lyrics])
        
        # Create the prompt
        prompt = f"""
Generate 2 different text-to-image prompts that visually represent the theme of these song lyrics.

LYRICS SAMPLE:
{lyric_str}

MAIN THEMES:
{concept_str}

MAIN SUBJECTS:
{subject_str}

OUTPUT INSTRUCTIONS:
- Create exactly 2 different image prompts, each 50-75 words
- Each prompt should be highly visual and detailed
- Include mood, lighting, colors, composition, and visual elements
- Transform abstract themes into concrete visual symbols
- Avoid directly quoting lyrics; create visual metaphors instead
- Format output with "PROMPT 1:" and "PROMPT 2:" headers
- Make each prompt distinct and evocative
- Focus on creating emotionally resonant visual scenes
"""
        
        return prompt
    
    def _parse_prompt_response(self, response_text):
        """Parse the LLM response to extract image prompts."""
        prompts = []
        
        # Look for "PROMPT X:" headers followed by text
        prompt_matches = re.finditer(r"PROMPT\s+(\d+):(.*?)(?=PROMPT\s+\d+:|$)", 
                                    response_text, 
                                    re.DOTALL)
        
        for match in prompt_matches:
            prompt_text = match.group(2).strip()
            if prompt_text:
                prompts.append(prompt_text)
        
        # If regex failed, try a simple split approach
        if not prompts and "PROMPT" in response_text:
            parts = response_text.split("PROMPT")
            for part in parts[1:]:
                cleaned_part = re.sub(r"^\s*\d+\s*:", "", part).strip()
                if cleaned_part:
                    prompts.append(cleaned_part)
        
        # If we still don't have prompts, use the whole response
        if not prompts and response_text.strip():
            prompts = [response_text.strip()]
            
        return prompts
    
    def get_prompt_for_lyric(self, lyric_text, cluster_id=None):
        """Get an appropriate image prompt for a given lyric."""
        try:
            # If we know the cluster and have prompts for it, use one
            with self.lock:
                if cluster_id is not None and cluster_id in self.cluster_prompts:
                    prompts = self.cluster_prompts[cluster_id]
                    if prompts:
                        return random.choice(prompts)
            
            # Otherwise, generate a new prompt
            self._initialize_client()
            
            # Create a simple prompt for a single lyric
            llm_prompt = f"""
Generate a detailed text-to-image prompt that visually represents this song lyric:

LYRIC: "{lyric_text}"

Create a 50-75 word visual prompt that:
- Captures the emotional essence and theme of the lyric
- Includes specific visual elements, mood, lighting, colors, and composition
- Transforms the lyric's meaning into concrete visual imagery
- Avoids directly quoting the lyric text
- Is suitable for an AI image generation system

PROMPT:
"""
            
            # Get a suggestion from the LLM
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[llm_prompt]
            )
            
            # Clean up the response
            prompt = response.text.strip()
            return prompt
                
        except Exception as e:
            print(f"Error generating prompt for lyric: {e}")
            import traceback
            traceback.print_exc()  # Print stack trace for debugging
            return f"A visual representation of the lyric: {lyric_text}"  # Fallback
