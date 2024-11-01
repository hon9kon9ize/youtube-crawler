import os
import os.path
from pytubefix import YouTube
from moviepy.editor import VideoFileClip
from tqdm.auto import tqdm
from pydub import AudioSegment, silence
from resemble_enhance.enhancer.inference import denoise
import torchaudio
import torch
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"


# denoise and enhance audio
def enhance_audio(audio_file: str, output_file=None):
    dwav, sr = torchaudio.load(audio_file)
    dwav = dwav.mean(dim=0)
    wav, new_sr = denoise(dwav, sr, device)

    if output_file:
        torchaudio.save(output_file, wav.unsqueeze(0), new_sr)
    else:
        return wav, new_sr


def segment_audio(
    audio_file: str,
    min_silence_len: int = 500,
    silence_thresh: int = 16,
    len_max: int = 5,
    len_min: int = 2,
):
    sound = AudioSegment.from_wav(audio_file)
    dBFS = sound.dBFS
    chunks = silence.split_on_silence(
        sound,
        min_silence_len=min_silence_len,
        silence_thresh=dBFS - silence_thresh,
    )
    target_max_length = len_max * 1000
    target_min_length = len_min * 1000
    chunks = [
        chunk
        for chunk in chunks
        if len(chunk) >= target_min_length and len(chunk) <= target_max_length
    ]

    return chunks


def main():
    args = argparse.ArgumentParser()
    args.add_argument("list_file", type=str, help="Path to the download list")
    args = args.parse_args()

    # read list file
    yt_links = []
    with open(args.list_file, "r") as f:
        yt_links = f.readlines()
    yt_links = [link.strip() for link in yt_links]

    print("Number of videos to download:", len(yt_links))

    # download videos
    os.makedirs("./videos", exist_ok=True)

    for i, yt_link in tqdm(enumerate(yt_links), total=len(yt_links)):
        yt = YouTube(yt_link)
        video_file = f"{i}.mp4"

        if os.path.exists(f"./videos/{video_file}"):
            continue

        try:
            stream = yt.streams.get_highest_resolution()
            stream.download("./videos", filename=video_file)
        except Exception as e:
            print(f"Failed to download video {yt_link}: {e}")
            continue

    # convert to audio
    # !mkdir -p ./audios
    os.makedirs("./audios", exist_ok=True)

    # convert to audio
    for i in range(len(yt_links)):
        audio_file = f"./audios/{i}.wav"
        video_file = f"./videos/{i}.mp4"

        if os.path.exists(audio_file) or not os.path.exists(video_file):
            continue

        try:
            video = VideoFileClip(video_file)
            audio = video.audio
            audio.write_audiofile(audio_file)
        except:
            pass

    # remove videos to save space
    os.system("rm -rf ./videos")

    for i in tqdm(range(len(yt_links))):
        audio_file = f"./audios/{i}.wav"

        if not os.path.exists(audio_file):
            continue

        try:
            enhance_audio(f"./audios/{i}.wav", f"./audios/{i}_enhanced.wav")
        except:
            pass

    # segment audio
    os.makedirs("./audio_chunks", exist_ok=True)

    for i in tqdm(range(len(yt_links))):
        audio_file = f"./audios/{i}_enhanced.wav"

        if not os.path.exists(audio_file):
            continue

        chunks = segment_audio(audio_file)

        for j, chunk in enumerate(chunks):
            chunk.export(f"./audio_chunks/{i}_{j}.wav", format="wav")


if __name__ == "__main__":
    main()
