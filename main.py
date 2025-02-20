# main.py
import threading
import time
import asyncio
from src.info.song_info import SongInfo
from src.analyzer.lyrics_analyzer import LyricsAnalyzer

async def analyze_lyrics_async(lyrics, lyrics_analyzer):
    """
    Runs lyrics analysis asynchronously.
    """
    loop = asyncio.get_event_loop()  
    await loop.run_in_executor(None, lyrics_analyzer.analyze_lyrics, lyrics) 

def main():
    song_info = SongInfo(headless=False)
    lyrics_analyzer = LyricsAnalyzer()

    song_info.load_site()
    stop_event = threading.Event()

    def monitor_song_title():
        while not stop_event.is_set():
            updated_title = song_info.update_song_title()
            if updated_title is not None:
                print("Current song:", updated_title)
                print("-" * 40)
                time.sleep(1)
                lyrics = song_info.get_fullscreen_lyrics()
                print(lyrics)
                print("-" * 40)
                asyncio.run(analyze_lyrics_async(lyrics, lyrics_analyzer))

            time.sleep(2)

    monitor_thread = threading.Thread(target=monitor_song_title)
    monitor_thread.start()

    input("Press Enter to exit and close the browser...")

    stop_event.set()
    monitor_thread.join()

    song_info.close()


if __name__ == "__main__":
    main()
