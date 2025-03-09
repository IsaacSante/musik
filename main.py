import threading
import time
import asyncio
from src.info.song_info import SongInfo
from src.analyzer.llm_analysis import LLMAnalysis
from src.analyzer.lyric_to_prompt import LyricToImagePrompter
from src.image.prompt_to_image import PromptToImageGenerator

async def make_prompts_and_images(lyrics: str, song_name: str, image_generator):
    """
    Asynchronously generates prompts from lyrics and images in real-time.
    """
    prompt_generator = LyricToImagePrompter()
    analysis_result = await asyncio.to_thread(
        prompt_generator.generate_image_prompts, 
        lyrics,
        image_generator=image_generator,
        song_name=song_name
    )
    return analysis_result


async def analyze_lyrics_async(lyrics: str):
    """
    Asynchronously analyzes the lyrics using LLMAnalysis.
    """
    llm_analysis = LLMAnalysis()
    analysis_result = await asyncio.to_thread(llm_analysis.analyze_lyrics, lyrics)  # Run sync function in separate thread
    return analysis_result


def main():
    song_info = SongInfo(headless=True)
    song_info.load_site()
    stop_event = threading.Event()
    
    # Initialize the image generator
    image_generator = PromptToImageGenerator()

    async def monitor_song_title():
        current_song = None
        
        while not stop_event.is_set():
            updated_title = song_info.update_song_title()
            if updated_title is not None:
                if updated_title != current_song:
                    current_song = updated_title
                    print("Current song:", current_song)
                    print("-" * 40)
                    time.sleep(1)
                    lyrics = song_info.get_fullscreen_lyrics()
                    print("-" * 40)

                     # Asynchronously analyze the lyrics
                    analysis_result = await analyze_lyrics_async(lyrics)
                    if analysis_result:
                        print("Lyrics Analysis:")
                        print(analysis_result)
                    else:
                        print("Lyrics analysis failed or returned empty result.")
                    
                    # # Generate prompts and images in real-time
                    # prompt_results = await make_prompts_and_images(lyrics, current_song, image_generator)
                    
                    # if not prompt_results:
                    #     print("Lyrics analysis failed or returned empty result.")
            time.sleep(2)

    async def run_monitor():
        await monitor_song_title()

    monitor_thread = threading.Thread(target=lambda: asyncio.run(run_monitor()))
    monitor_thread.start()

    input("Press Enter to exit and close the browser...")

    stop_event.set()
    monitor_thread.join()

    song_info.close()

if __name__ == "__main__":
    main()
