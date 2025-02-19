import os
import getpass
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException

class SongInfo:
    def __init__(self, headless=False):
        """
        Initializes the SongInfo instance.
        :param headless: Whether to run Chrome in headless mode.
        """
        self.headless = headless
        self.driver = None
        self.current_song_title = None
        self.song_title_history = []  # Maintains a Python-side history of song titles.

    def _initialize_driver(self):
        """
        Sets up the Chrome WebDriver with desired options, including a dynamically created user data directory.
        """
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        
        # Dynamically create a Chrome profile folder based on the OS username.
        username = getpass.getuser()
        base_profile_dir = os.path.join(os.path.expanduser("~"), ".selenium_profiles")
        os.makedirs(base_profile_dir, exist_ok=True)
        user_profile_dir = os.path.join(base_profile_dir, f"chrome_profile_{username}")
        os.makedirs(user_profile_dir, exist_ok=True)

        # Tell Chrome to use this profile directory.
        chrome_options.add_argument(f"--user-data-dir={user_profile_dir}")

        self.driver = webdriver.Chrome(options=chrome_options)

    def load_site(self, url="https://open.spotify.com/lyrics"):
        """
        Loads the given URL using Selenium and returns the page source.
        
        :param url: The URL to load.
        :return: The page source as a string.
        """
        if self.driver is None:
            self._initialize_driver()
        self.driver.get(url)
        return self.driver.page_source

    def get_song_title(self):
        attempts = 0
        while attempts < 3:
            try:
                element = self.driver.find_element(By.CSS_SELECTOR, '[data-testid="context-item-link"]')
                return element.text.strip()
            except StaleElementReferenceException:
                attempts += 1
                time.sleep(0.1)  # short wait before retrying
            except NoSuchElementException:
                # If the element is not found at all, we return None immediately.
                return None
        return None


    def update_song_title(self):
        new_title = self.get_song_title()
        if new_title is None or new_title == self.current_song_title:
            # Either the site hasn't loaded, or the title hasn't changed.
            return None
        # New title detected.
        self.current_song_title = new_title
        self.song_title_history.append(new_title)
        return new_title

    def get_fullscreen_lyrics(self):
        """
        Finds all divs with data-testid="fullscreen-lyric",
        extracts the text content from their inner div,
        and returns all text as one block with each piece on a new line.
        """
        # Locate all lyric containers
        lyric_elements = self.driver.find_elements(By.CSS_SELECTOR, 'div[data-testid="fullscreen-lyric"]')
        lyrics = []
        for elem in lyric_elements:
            try:
                # Get the inner div that holds the actual lyric text
                inner_div = elem.find_element(By.XPATH, './div')
                text = inner_div.text.strip()
                if text:
                    lyrics.append(text)
            except Exception:
                continue
        return "\n".join(lyrics)


    def close(self):
        """
        Closes the Selenium WebDriver.
        """
        if self.driver:
            self.driver.quit()
            self.driver = None
