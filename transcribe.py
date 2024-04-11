from youtube_transcript_api import YouTubeTranscriptApi , TranscriptsDisabled
from pytube import YouTube
from deepgram import Deepgram
import os, re

def download_audio_from_video(video_url, output_path='./Audio'):
    try:
        # Create a YouTube object
        yt = YouTube(video_url)

        # Select the stream with the audio you want (e.g., highest quality audio stream)
        audio_stream = yt.streams.filter(only_audio=True).first()

        # Download the audio and specify the output path
        audio_file = audio_stream.download(output_path=output_path)
        print(audio_file)
        return audio_file  # Return the filename of the downloaded audio
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def transcribeUsingDeepGram(video_url):
    try:
        DEEPGRAM_API_KEY = '7c5ba97aca47441583c0e3d513bf9d3eeed342ef'
        PATH_TO_FILE = download_audio_from_video(video_url)
        if PATH_TO_FILE is None:
            raise "Failed to Download Audio"
        deepgram = Deepgram(DEEPGRAM_API_KEY)
        # Open the audio file
        with open(PATH_TO_FILE, 'rb') as audio:
            # Set the correct mimetype for your file
            source = {'buffer': audio, 'mimetype': 'mp4'}
            response = deepgram.transcription.sync_prerecorded(source, {'punctuate': True})
            transcript = response['results']['channels'][0]['alternatives'][0]['transcript']
        os.remove(PATH_TO_FILE)
        return transcript
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
def convert_link_to_video_id(link):
    '''Converts a YouTube video link to a video ID.
    Args:
        Link: The link to the YouTube video.
    Returns:
        The video ID, or None if the link is not valid.
    '''
    video_id_regex = r'v=([^&]+)'
    video_id_match = re.search(video_id_regex, link)

    # If the video ID match is found, then return the video ID.
    if video_id_match is not None:
        return video_id_match.group(1)
    else:
        return None

def download_subtitles_from_video(video_link, language='en'):
    '''Downloads the subtitles for a YouTube video from a video link.
    Args:
        video_link: The link to the YouTube video.
        language: The language of the subtitles to download.
    Returns:
        The combined text of subtitles or an error message.
    '''
    try:
        # Converting to Video_id
        video_id = convert_link_to_video_id(video_link)
        if video_id is None:
            raise ValueError("The video link is not valid.")
        
        # Attempt to retrieve subtitles
        srt = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Process subtitles
        combined_text = ""
        for item in srt:
            text = item['text']
            combined_text += text + " "
        
        # Remove the trailing space
        combined_text = combined_text.strip()
        return combined_text
    except TranscriptsDisabled as e:
        return transcribeUsingDeepGram(video_link)
    except ValueError as e:
        return "The video link is not valid."
    except Exception as e:
        return f"An error occurred: {str(e)}"

