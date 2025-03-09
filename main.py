import threading
import time
import asyncio
from src.info.song_info import SongInfo
from src.analyzer.llm_analysis import LLMAnalysis

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
