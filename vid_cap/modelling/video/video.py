import os
from pydoc import Helper
import ffmpy
import random
from imageio_ffmpeg import get_ffmpeg_exe
from multiprocessing import Pool, cpu_count
import functools
import operator
import os.path
import os
import numpy as np
import vid_cap.modelling.video.helper_functions as helper
from vid_cap.modelling.video.frame_extractor import FrameExtractor
from vid_cap.modelling.video.image_selector import ImageSelector

class Video(object):
    """Class for all video frames operations

    :param object: base class inheritance
    :type object: class:`Object`
    """

    def __init__(self):
        # Find out location of ffmpeg binary on system
        helper._set_ffmpeg_binary_path()
        self.temp_folder = os.path.abspath(os.path.join("clipped"))
        self._min_video_duration = 5.0
       
        # Calculating optimum number of processes for multiprocessing
        self.n_processes = cpu_count() // 2 - 1
        if self.n_processes < 1:
            self.n_processes = None

        self.mediapipe_autoflip = None
        
        if not os.path.isdir(self.temp_folder):
            os.mkdir(self.temp_folder)
        
    def extract_video_keyframes(self, no_of_frames, file_path) -> list:
        """Returns a list of best key images/frames from a single video.

        :param no_of_frames: Number of key frames to be extracted
        :type no_of_frames: int, required
        :param file_path: video file location
        :type file_path: str, required
        :param writer: Writer object to process keyframe data
        :type writer: Writer, required
        :return: List of numpy.2darray Image objects
        :rtype: list
        """

        top_frames = self._extract_keyframes_from_video(no_of_frames, file_path)

        # TODO: if len(top_frames) != no_of_frames, repeat some random frames in top_frames in order to reach same len, place the repeted frames as the following index from its origin without smashing the frame in that position
        if len(top_frames) != no_of_frames:
            selected_frames = set()
            while len(top_frames) < no_of_frames:
                random_frame = random.choice(top_frames)
                indices = np.where((top_frames == random_frame).all(axis=1))[0]
                for index in indices:
                    frame_str = str(random_frame)
                    if frame_str not in selected_frames:
                        top_frames.insert(index + 1, random_frame)
                        selected_frames.add(frame_str)

        return top_frames

    def _extract_keyframes_from_video(self, no_of_frames, file_path):
            """Core method to extract keyframe for a video

            :param no_of_frames: [description]
            :type no_of_frames: [type]
            :param file_path: [description]
            :type file_path: [type]
            """
            # Creating the multiprocessing pool
            self.pool_extractor = Pool(processes=self.n_processes)
            # Split the input video into chunks. Each split(video) will be stored
            # in a temp
            if not helper._check_if_valid_video(file_path):
                raise Exception("Invalid or corrupted video: " + str(file_path))

            # split videos in chunks in smaller chunks for parallel processing.
            chunked_videos = self._split(file_path)
            frame_extractor = FrameExtractor()

            # Passing all the clipped videos for  the frame extraction using map function of the
            # multiprocessing pool
            with self.pool_extractor:
                extracted_candidate_frames = self.pool_extractor.map(
                    frame_extractor.extract_candidate_frames, chunked_videos
                )
            # Converting the nested list of extracted frames into 1D list
            extracted_candidate_frames = functools.reduce(operator.iconcat, extracted_candidate_frames, [])

            self._remove_clips(chunked_videos)
            image_selector = ImageSelector(self.n_processes)

            top_frames = image_selector.select_best_frames(
                extracted_candidate_frames, no_of_frames
            )

            del extracted_candidate_frames

            return top_frames
    
    def _split(self, file_path):
        """Split videos using ffmpeg library first by copying audio and
        video codecs from input files, it leads to faster splitting, But if
        resulting splitted videos are unreadable try again splitting by using
        ffmpeg default codecs. If splitteed videos are still unreadable throw an
        exception.

        :param file_path: path of video file
        :type file_path: str, required
        :return: List of path of splitted video clips
        :rtype: list
        """
        chunked_videos = self._split_with_ffmpeg(file_path)
        corruption_in_chunked_videos = False
        for chunked_video in chunked_videos:
            if not helper._check_if_valid_video(chunked_video):
                corruption_in_chunked_videos = True

        if corruption_in_chunked_videos:
            chunked_videos = self._split_with_ffmpeg(file_path, override_video_codec=True)
            for chunked_video in chunked_videos:
                if not helper._check_if_valid_video(chunked_video):
                    raise Exception(
                        "Error in splitting videos in multiple chunks, corrupted video chunk: "
                        + chunked_video
                    )

        return chunked_videos
    
    def _split_with_ffmpeg(self, file_path, override_video_codec=False, break_point_duration_in_sec=None):
        """Function to split the videos and persist the chunks

        :param file_path: path of video file
        :type file_path: str, required
        :param override_video_codec: If true overrides input video codec to ffmpeg default codec else copy input video codec, defaults to False
        :type override_video_codec: bool, optional
        :param break_point_duration_in_sec: duration in sec for break point
        :type break_point_duration_in_sec: int, optional
        :return: List of path of splitted video clips
        :rtype: list
        """
        clipped_files = []
        duration = self._get_video_duration_with_cv(file_path)
        # setting the start point to zero
        # Setting the breaking point for the clip to be 25 or if video is big
        # then relative to core available in the machine
        # If video size is large it makes sense to split videos into chunks
        # proportional to number of cpu cores. So each cpu core will get on
        # video to process.
        # if video duration is divided by cpu_count() then result should be
        # 15 sec is thumb rule for threshold value it could be set to 25 or
        # any other value. Logic ensures for large enough videos we don't end
        # up dividing video in too many clips.
        # TODO: Try max 5 minutes video

        if break_point_duration_in_sec is None:
            clip_start, break_point = (
                0,
                duration // cpu_count() if duration // cpu_count() > 15 else 25,
            )
        else:
            clip_start, break_point = (
                0,
                break_point_duration_in_sec,
            )

        # Loop over the video duration to get the clip stating point and end point to split the video
        while clip_start < duration:

            clip_end = clip_start + break_point

            # Setting the end position of the particular clip equals to the end time of original clip,
            # if end position or end position added with the **min_video_duration** is greater than
            # the end time of original video
            if clip_end > duration or (clip_end + self._min_video_duration) > duration:
                clip_end = duration

            clipped_files.append(
                self._write_videofile(file_path, clip_start, clip_end, override_video_codec)
            )

            clip_start = clip_end
        return clipped_files
    
    def _write_videofile(self, video_file_path, start, end, override_video_codec=False):
        """Function to clip the video for given start and end points and save the video

        :param video_file_path: path of video file
        :type video_file_path: str, required
        :param start: start time for clipping
        :type start: float, required
        :param end: end time for clipping
        :type end: float, required
        :param override_video_codec: If true overrides input video codec to ffmpeg default codec else copy input video codec, defaults to False
        :type override_video_codec: bool, optional
        :return: path of splitted video clip
        :rtype: str
        """

        name = os.path.split(video_file_path)[1]

        # creating a unique name for the clip video
        # Naming Format: <video name>_<start position>_<end position>.mp4
        _clipped_file_path = os.path.join(
            self.temp_folder,
            "{0}_{1}_{2}.mp4".format(
                name.split(".")[0], int(1000 * start), int(1000 * end)
            ),
        )

        self._ffmpeg_extract_subclip(
            video_file_path,
            start,
            end,
            targetname=_clipped_file_path,
            override_video_codec=override_video_codec,
        )
        return _clipped_file_path
    
    def _ffmpeg_extract_subclip(
        self, filename, t1, t2, targetname=None, override_video_codec=False
    ):
        """chops a new video clip from video file ``filename`` between
            the times ``t1`` and ``t2``, Uses ffmpy wrapper on top of ffmpeg
            library
        :param filename: path of video file
        :type filename: str, required
        :param t1: time from where video to clip
        :type t1: float, required
        :param t2: time to which video to clip
        :type t2: float, required
        :param override_video_codec: If true overrides input video codec to ffmpeg default codec else copy input video codec, defaults to False
        :type override_video_codec: bool, optional
        :param targetname: path where clipped file to be stored
        :type targetname: str, optional
        :return: None
        """
        name, ext = os.path.splitext(filename)

        if not targetname:
            T1, T2 = [int(1000 * t) for t in [t1, t2]]
            targetname = name + "{0}SUB{1}_{2}.{3}".format(name, T1, T2, ext)

        #timeParamter = "-ss " + "%0.2f" % t1 + " -t " + "%0.2f" % (t2 - t1)

        ssParameter = "-ss " + "%0.2f" % t1
        timeParamter = " -t " + "%0.2f" % (t2 - t1)
        hideBannerParameter = " -y -hide_banner -loglevel panic  "
        if override_video_codec:
            codecParameter = " -vcodec libx264 -max_muxing_queue_size 9999"
        else:
            codecParameter = " -vcodec copy -avoid_negative_ts 1 -max_muxing_queue_size 9999"

        # Uses ffmpeg binary for video clipping using ffmpy wrapper
        FFMPEG_BINARY = os.getenv("FFMPEG_BINARY")
        ff = ffmpy.FFmpeg(
            executable=FFMPEG_BINARY,
            inputs={filename: ssParameter + hideBannerParameter},
            outputs={targetname: timeParamter + codecParameter},
        )
        # Uncomment next line for watching ffmpeg command line being executed
        # print("ff.cmd", ff.cmd)
        ff.run()

    def _get_video_duration_with_cv(self, file_path):
        """
        Computes video duration by getting frames count and fps info (using opencv)
        :param file_path:
        :type file_path:
        :return:
        :rtype:
        """
        video_info = helper.get_video_info(file_path)
        video_frame_size = video_info[0]
        video_fps = video_info[1]
        video_frames = video_info[2]
        video_duration = round((video_frames / video_fps), 2)
        return video_duration
    
    def _remove_clips(self, video_clips):
        """Remove video clips from the temp directory given list of video clips

        :param video_clips: [description]
        :type video_clips: [type]
        """
        for clip in video_clips:
            try:
                os.remove(clip)
            except OSError:
                print("Error in removing clip: " + clip)
            # print(clip, " removed!")