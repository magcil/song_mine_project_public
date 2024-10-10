import os
import sys
import json
import time
import math
import shutil
import argparse
import spotipy
from spotipy import Spotify
from typing import List,Tuple
from dotenv import load_dotenv
from exiftool import ExifToolHelper
from spotipy.oauth2 import SpotifyClientCredentials




class SongNotFoundError(Exception):
    """ Class for SongNotFoundError.
        This error is raised when a song is not found in the Spotify database

    Args:
        Exception : The exception to be raised
    """
    def __init__(self,track:str):
        self.track =track
        self.message = f'{track} was not found in the Spotify database'
        super().__init__(self.message)



def track_metadata(track_id: str, sp: Spotify) -> dict:
    """
    Retrieve the metadata for a given track id from Spotify

    Parameters:
        - track_id (str): the id of the track to retrieve metadata for
        - sp (Spotify): a Spotify object to perform the API call

    Returns:
        - dict: a dictionary containing the track's metadata
    """
    try:
        track_info = sp.track(track_id)
        track_genre = track_info['album']['artists'][0].get('genres', "")

        if isinstance(track_genre, list):
            track_genre = ', '.join(track_genre)

    except (spotipy.exceptions.SpotifyException, KeyError) as e:
        print(f'Error: {e}')
        return None

    try:
        metadata = {
            'track_id': track_info['id'],
            'track_name': track_info['name'],
            'track_length': track_info['duration_ms'],
            'track_preview_url': track_info['preview_url'],
            'track_number': track_info['track_number'],
            'track_href': track_info['href'],
            'track_external_urls': track_info['external_urls']['spotify'],
            'track_external_id_isrc': track_info['external_ids']['isrc'],
            'album_id': track_info['album']['id'],
            'album_name': track_info['album']['name'],
            'album_href': track_info['album']['href'],
            'album_ext_url': track_info['album']['external_urls']['spotify'],
            'album_release_date': track_info['album']['release_date'],
            'album_release_date_precision': track_info['album']['release_date_precision'],
            'album_total_tracks': track_info['album']['total_tracks'],
            'album_type': track_info[f'album']['type'],
            'album_image_large': track_info['album']['images'][0]['url'],
            'album_image_medium': track_info['album']['images'][1]['url'],
            'album_image_small': track_info['album']['images'][2]['url'],
            'artist_id': track_info['album']['artists'][0]['id'],
            'artist_name': track_info['album']['artists'][0]['name'],
            'artist_href': track_info['album']['artists'][0]['href'],
            'artists_ext_url': track_info['album']['artists'][0]['external_urls']['spotify'],
            'track_genre': track_genre,
        }
        return metadata
    except (IndexError,KeyError) as e:
        print(f'Error: {e}')
        return None



def get_track_id(response: dict) -> str:
    """ Given an existing song on Spotify, get the Spotify track id

    Returns:
        str: The track's id
    """
    try:
        return response['tracks']['items'][0]['id']
    except (IndexError, KeyError, TypeError):
        print(f'Error: {response}')
        return None


def search_song_existance(response: dict) -> bool:
    """ Searches for a song a song's existance in the spotify database

    Returns:
        bool: True if the song exists, False otherwise
    """
    return  len(response['tracks']['items']) > 0



def connect_to_spotify(client_id: str, client_secret: str) -> spotipy.Spotify:
    """ Connect to Spotify using the client_id and client_secret

    Args:
        client_id (str): The client_id
        client_secret (str): The client_secret

    Returns:
        spotipy.Spotify: A spotipy.Spotify object
    """
    client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    if sp is None:
        raise ValueError('Error: Could not connect to Spotify.')
    return sp



def get_spotify_client_keys() -> Tuple[str, str]:

    """ Retrieve the Spotify client id and secret from the .env file located in the root directory of the project.

    Returns:
        Tuple[str, str]: A tuple containing the Spotify client id and secret.

    Raises:
        ValueError: If the Spotify client credentials are missing or the .env file does not exist.
    """

    env_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    load_dotenv(dotenv_path=env_file_path)

    client_id = os.environ.get('client_id')
    client_secret = os.environ.get('client_secret')

    if client_id is None or client_secret is None:
        raise ValueError('Error: Spotify client credentials are missing or the .env file does not exist.')
    return client_id, client_secret



def show_statistics(songs: List[str], song_artist_title: List[Tuple[str, str]]):
        """ Show statistics about the songs

        Args:
            ongs (List[str]): The list of songs
            song_artist_title (List[Tuple[str, str]]): The list of songs with artist and title
        """
        print(f'Number of songs: {len(songs)}')
        print(f'Number of songs with artist and title: {len(song_artist_title)}')
        print(f'Difference: {len(songs) - len(song_artist_title)}')
        print('Percentage of songs with artist: {:.2f}%'.format(len(song_artist_title) / len(songs) * 100))
        print('Percentage of songs with title: {:.2f}%'.format(len(song_artist_title) / len(songs) * 100))



def parse_song_properties(songs: List[str]) -> List[Tuple[str, str]]:
    """ Parse the audio file properties

    Returns:
        List of tuples: A list of tuples containing the artist and title of each song
    """

    song_info = []

    try:
        with ExifToolHelper() as et:
            for song in songs:
                metadata = et.get_metadata(song)

                if not metadata:
                    #print(f"{song} has no metadata in its properties.")
                    continue
                artist, title = metadata[0].get('RIFF:Artist', None), metadata[0].get('RIFF:Title', None)

                if not artist or not title:
                    #print(f"{song} has no artist or title in its properties.")
                    continue
                song_info.append((artist, title))
    except IndexError as e:
        print(f'Error: {e}')

    return song_info



def get_songs(root_dir: str, excluded_dirs: List[str]=[], extentions: List[str]=[]) -> List[str]:
    """ Get the songs from the root directory

    Args:
        root_dir (str): The root directory
        excluded_dirs (List[str], optional): A lsit containing sub-dirs to exclude. Defaults to [].

    Returns:
        List[str]: A list containing the songs found in the given directory
    """
    songs = []
    for dir_name, subdir_list, file_list in os.walk(root_dir):
        for subdir in (f.name for f in os.scandir(dir_name) if f.is_dir()):
            if subdir in excluded_dirs:
                print(f'Excluding sub-directory: {subdir}')
                subdir_list.remove(subdir)
            print('\t- subdirectory: %s' % subdir)

        for fname in file_list:
            if fname.endswith(tuple(extentions)):
                file_path = os.path.abspath(os.path.join(dir_name, fname))
                songs.append(file_path)
            else:
                file_path = os.path.abspath(os.path.join(dir_name, fname))
                songs.append(file_path)
    print(f'\nNumber of songs found: {len(songs)}')
    return songs



def split_audio_files(audio_files: List[str], directories: List[str], files_per_directory: str):
    """ Split the audio files into the given directories

    Args:
        audio_files (list): A list containing the audio files
        directories (list): A list containing the directories
        files_per_directory (str): The approx. number of files per directory
    """
    num_dirs = len(directories)
    files_per_dir = int(files_per_directory)

    for i, file in enumerate(audio_files):
        dir_index = i // files_per_dir

        if dir_index >= num_dirs:
            dir_index = num_dirs - 1
        dest = os.path.join(directories[dir_index], os.path.basename(file))
        #print(f'Copying {i} of {len(audio_files)} {os.path.basename(file)} to {dest}...')
        shutil.copy2(file, dest)



def decide_folders(number_of_files, base_number=8000) -> int:
    """ Decide how many folders are needed

    Args:
        number_of_files (int): The number of files to be split
        base_number (int, optional): The base number. Defaults to 8000.

    Returns:
        int: The number of folders needed
    """
    folders_needed = number_of_files // base_number + 1
    if number_of_files % base_number == 0:
        folders_needed -= 1
    return folders_needed