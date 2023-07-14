"""
.. module:: Katna.ffmpeg_helper
    :platform: OS X
    :synopsis: This module has functions related to key frame extraction 
"""
import os
import cv2
import numpy as np
from scipy.signal import argrelextrema

import subprocess
from imageio_ffmpeg import get_ffmpeg_exe


def _convert_to_seconds(time):
    """Will convert any time into seconds.
    If the type of `time` is not valid,
    it's returned as is.
    Here are the accepted formats::
    >>> convert_to_seconds(15.4)   # seconds
    15.4
    >>> convert_to_seconds((1, 21.5))   # (min,sec)
    81.5
    >>> convert_to_seconds((1, 1, 2))   # (hr, min, sec)
    3662
    >>> convert_to_seconds('01:01:33.045')
    3693.045
    >>> convert_to_seconds('01:01:33,5')    # coma works too
    3693.5
    >>> convert_to_seconds('1:33,5')    # only minutes and secs
    99.5
    >>> convert_to_seconds('33.5')      # only secs
    33.5

    :param time: time_string
    :type time: string
    :return: time in seconds
    :rtype: float
    """

    factors = (1, 60, 3600)

    if isinstance(time, str):
        time = [float(part.replace(",", ".")) for part in time.split(":")]

    if not isinstance(time, (tuple, list)):
        return time

    return sum(mult * part for mult, part in zip(factors, reversed(time)))


def _check_if_valid_video(file_path):
    """Function to check if given video file is a valid video compatible with
    ffmpeg/opencv

    :param file_path: video filename
    :type file_path: str
    :return: Return True if valid video file else False
    :rtype: bool
    """
    try:
        # Check if file extension of video is in list of
        # supported/valid videos according to ffmpeg
        file_extension = os.path.splitext(file_path)[1]
        if file_extension not in video_extensions:
            return False

        vid = cv2.VideoCapture(str(file_path))
        if vid.isOpened():
            # Making sure we can read at least two frames from video
            ret, frame = vid.read()
            ret, frame = vid.read()
            # Making sure video frame is not empty
            if frame is not None:
                return True
            else:
                return False
        else:
            return False
    except cv2.error as e:
        #print("cv2.error:", e)
        return False
    except Exception as e:
        #print("Exception:", e)
        return False


def get_video_info(file_path):
    """
    Function to get the video frame size in bytes.
    :param file_path:
    :type file_path:
    :return:
    :rtype:
    """
    try:
        # Check if file extension of video is in list of
        # supported/valid videos according to ffmpeg
        file_extension = os.path.splitext(file_path)[1]
        if file_extension not in video_extensions:
            return False

        vid = cv2.VideoCapture(str(file_path))
        if vid.isOpened():
            # Making sure we can read at least two frames from video
            ret, frame = vid.read()
            ret, frame = vid.read()

            frame_size_in_bytes = frame.size

            fps = vid.get(cv2.CAP_PROP_FPS)
            frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)

            # Making sure video frame is not empty
            if frame is not None:
                return frame_size_in_bytes, fps, frame_count
            else:
                raise Exception(" Could not read frame from Video.")
        else:
            raise Exception(" Could not read frame from Video.")
    except cv2.error as e:
        #print("cv2.error:", e)
        raise Exception(" Could not read frame from Video.", e)
    except Exception as e:
        raise Exception(" Could not read frame from Video.", e)


def _set_ffmpeg_binary_path():
    """Function for getting path to ffmpeg binary on your system to be
    used by ffmpy
    # Derived from ffmpeg detection code borrowed from moviepy
    # https://github.com/Zulko/moviepy/moviepy/config.py
    # The MIT License (MIT)
    # Copyright (c) 2015 Zulko
    #
    :raises IOError: [description]
    """
    FFMPEG_BINARY = os.getenv("FFMPEG_BINARY", "ffmpeg-imageio")

    if FFMPEG_BINARY == "ffmpeg-imageio":
        FFMPEG_BINARY = get_ffmpeg_exe()
    else:
        success, err = _try_cmd([FFMPEG_BINARY])
        if not success:
            raise IOError(
                f"{err} - The path specified for the ffmpeg binary might be wrong"
            )
    os.environ["FFMPEG_BINARY"] = FFMPEG_BINARY


def _try_cmd(cmd):
    """
    # Derived from ffmpeg detection code borrowed from moviepy
    # https://github.com/Zulko/moviepy/moviepy/config.py
    # The MIT License (MIT)
    # Copyright (c) 2015 Zulko
    #
    :param cmd: command to execute
    :type cmd: string
    :return: True/False with error
    :rtype: Bool, Error
    """
    try:
        popen_params = _cross_platform_popen_params(
            {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "stdin": subprocess.DEVNULL,
            }
        )
        proc = subprocess.Popen(cmd, **popen_params)
        proc.communicate()
    except Exception as err:
        return False, err
    else:
        return True, None


def _cross_platform_popen_params(popen_params):
    """
    # Derived from ffmpeg detection code borrowed from moviepy
    # https://github.com/Zulko/moviepy/moviepy/config.py
    # The MIT License (MIT)
    # Copyright (c) 2015 Zulko
    #
    Wrap with this function a dictionary of ``subprocess.Popen`` kwargs and
    will be ready to work without unexpected behaviours in any platform.
    Currently, the implementation will add to them:
    - ``creationflags=0x08000000``: no extra unwanted window opens on Windows
    when the child process is created. Only added on Windows.

    :param popen_params: original popen_parameters
    :type popen_params: dict
    :return: modified popen_parameters
    :rtype: dict
    """
    if os.name == "nt":
        popen_params["creationflags"] = 0x08000000
    return popen_params


video_extensions = [
        ".str",
        ".aa",
        ".aac",
        ".ac3",
        ".acm",
        ".adf",
        ".adp",
        ".dtk",
        ".ads",
        ".ss2",
        ".adx",
        ".aea",
        ".afc",
        ".aix",
        ".al",
        ".ape",
        ".apl",
        ".mac",
        ".aptx",
        ".aptxhd",
        ".aqt",
        ".ast",
        ".avi",
        ".avr",
        ".bfstm",
        ".bcstm",
        ".bit",
        ".bmv",
        ".brstm",
        ".cdg",
        ".cdxl",
        ".xl",
        ".c2",
        ".302",
        ".daud",
        ".str",
        ".dss",
        ".dts",
        ".dtshd",
        ".dv",
        ".dif",
        ".cdata",
        ".eac3",
        ".paf",
        ".fap",
        ".flm",
        ".flac",
        ".flv",
        ".fsb",
        ".g722",
        ".722",
        ".tco",
        ".rco",
        ".g723_1",
        ".g729",
        ".genh",
        ".gsm",
        ".h261",
        ".h26l",
        ".h264",
        ".264",
        ".avc",
        ".hevc",
        ".h265",
        ".265",
        ".idf",
        ".cgi",
        ".sf",
        ".ircam",
        ".ivr",
        ".flv",
        ".lvf",
        ".m4v",
        ".mkv",
        ".mk3d",
        ".mka",
        ".mks",
        ".mjpg",
        ".mjpeg",
        ".mpo",
        ".j2k",
        ".mlp",
        ".mov",
        ".mp4",
        ".m4a",
        ".3gp",
        ".3g2",
        ".mj2",
        ".mp2",
        ".mp3",
        ".m2a",
        ".mpa",
        ".mpc",
        ".mjpg",
        ".txt",
        ".mpl2",
        ".sub",
        ".msf",
        ".mtaf",
        ".ul",
        ".musx",
        ".mvi",
        ".mxg",
        ".v",
        ".nist",
        ".sph",
        ".nsp",
        ".nut",
        ".ogg",
        ".oma",
        ".omg",
        ".aa3",
        ".pjs",
        ".pvf",
        ".yuv",
        ".cif",
        ".qcif",
        ".rgb",
        ".rt",
        ".rsd",
        ".rsd",
        ".rso",
        ".sw",
        ".sb",
        ".smi",
        ".sami",
        ".sbc",
        ".msbc",
        ".sbg",
        ".scc",
        ".sdr2",
        ".sds",
        ".sdx",
        ".shn",
        ".vb",
        ".son",
        ".sln",
        ".mjpg",
        ".stl",
        ".sub",
        ".sub",
        ".sup",
        ".svag",
        ".tak",
        ".thd",
        ".tta",
        ".ans",
        ".art",
        ".asc",
        ".diz",
        ".ice",
        ".nfo",
        ".txt",
        ".vt",
        ".ty",
        ".ty+",
        ".uw",
        ".ub",
        ".v210",
        ".yuv10",
        ".vag",
        ".vc1",
        ".viv",
        ".idx",
        ".vpk",
        ".txt",
        ".vqf",
        ".vql",
        ".vqe",
        ".vtt",
        ".wsd",
        ".xmv",
        ".xvag",
        ".yop",
        ".y4m",
    ]