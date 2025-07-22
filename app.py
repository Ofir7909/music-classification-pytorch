import gradio as gr
import models
import torch
import dataset
import io
import numpy as np
import re

from pytubefix import YouTube
from pydub import AudioSegment

MODEL_PATH = "models/cnn-v2-mixed-dropout.pth"

css = """
.container {max-width: 1200px; margin: auto;}
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = models.CNNv2()
net.load_state_dict(torch.load(MODEL_PATH))
net = net.to(device)


def download_from_youtube(url: str) -> io.BytesIO:
    try:
        stream = (
            YouTube(url)
            .streams.filter(only_audio=True, file_extension="mp4")
            .order_by("abr")
            .desc()
            .first()
        )

        f = io.BytesIO()
        stream.stream_to_buffer(f)
        f.seek(0)

        audio = AudioSegment.from_file(f)
        audio.export(f, format="wav")
        return f
    except Exception as e:
        raise gr.Error("Failed to download the audio.\n" + str(e))


def predict_genre_from_youtube(url: str, n=30) -> dict[str, float]:
    file = download_from_youtube(url)

    mel_spec = dataset.MusicGenres.make_mel(file)
    mel_chunks = dataset.MusicGenres.split_to_chunks(mel_spec)
    np.random.shuffle(mel_chunks)
    mel_chunks = mel_chunks if len(mel_chunks) <= n else mel_chunks[:n]
    mel_chunks = torch.from_numpy(np.array(mel_chunks))
    mel_chunks = mel_chunks.view(-1, 1, 128, 128).to(device)

    with torch.no_grad():
        net_out = net(mel_chunks)
        predictions = torch.sum(net_out, dim=0)
        predictions = torch.softmax(predictions, dim=0)

    genres = dataset.MusicGenres.GENRES
    predictions = dict(zip(genres, predictions.tolist()))
    return predictions


def parse_youtube_link(url: str) -> str:
    try:
        match = re.match(
            r"(?:https://)?(?:www\.)?(?:youtube\.com|youtu\.be)/(?:watch\?v=|v/)?([A-Za-z0-9_-]{11})(?:&.*)?",
            url,
        )
        if not match:
            raise gr.Error("Failed to parse the url")
        else:
            video_id = match.group(1)
    except IndexError:
        raise gr.Error("Failed to parse the url")
    return video_id


def on_predict_btn_clicked(url: str) -> tuple[str, dict[str, float]]:
    video_id = parse_youtube_link(url)
    video_player_html = f'<iframe width="560" height="315" src="https://www.youtube.com/embed/{video_id}" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>'
    return video_player_html, predict_genre_from_youtube(url)


with gr.Blocks(css=css) as app:
    with gr.Column(elem_classes="container"):
        url = gr.Text(
            label="Url", placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        submit_btn = gr.Button("Predict", variant="primary")
        with gr.Row():
            video_player_html = gr.HTML()
            labels = gr.Label(label="Genre", num_top_classes=3)

    submit_btn.click(
        fn=on_predict_btn_clicked, inputs=url, outputs=[video_player_html, labels]
    )

app.launch(share=True)
