# Muzik

## Project Overview:

* LLM Analysis: Uses Google’s generative AI to analyze song lyrics.
* Lyrics Clustering: Employs transformer-based embeddings and clustering techniques for lyric analysis and visualization.
* Song Information: Uses Selenium to scrape song titles and lyrics from a web page.

## Setup Instructions

### Environment Variables

Create a `.env` file in the root directory of the repository and add your Google API key. This key is required for the LLM analysis.

GOOGLE_API_KEY=your_actual_api_key


### NLTK Setup

The repository uses NLTK for text processing. On the first run, you might need to download the necessary resources:

```python
import nltk

nltk.download('punkt')
nltk.download('stopwords')

You can run these commands in a Python shell or include them in a setup script.

Selenium & ChromeDriver
ChromeDriver Requirement: Ensure that you have ChromeDriver installed on your system. The version of ChromeDriver should match your installed Chrome browser.

PATH Configuration: Add the ChromeDriver executable to your system's PATH, or specify its location directly in the code if needed.

Running the Application
Once the dependencies are installed and the environment is configured, you can start the application by running:

python main.py

This will launch the main program that monitors song titles and performs lyric analysis.

Additional Notes
Headless Mode: The SongInfo class supports a headless mode. To run Chrome in headless mode, instantiate the class with headless=True:

song_info = SongInfo(headless=True)

Troubleshooting
If you encounter errors related to missing NLTK data, ensure that both punkt and stopwords have been downloaded.
For Selenium issues, verify that ChromeDriver is correctly installed and accessible.
