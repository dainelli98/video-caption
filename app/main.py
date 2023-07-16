import asyncio
import logging
import subprocess
from flask import Flask, render_template, request

from pathlib import Path

from vid_cap import DATA_DIR
from vid_cap.modelling import preprocessing as pre
from vid_cap.modelling import _utils as utils

# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
app = Flask(__name__)

async def predict_review_sentiment(video_path):
    print(video_path)   

    args = [
        "inference",
        "--use-gpu",
        "--n-heads",
        "4",
        "--n-layers",
        "2",
        "--video-path",
        video_path,
        "--inference-model",
        DATA_DIR / Path("output/best/model"),
        "--inference-model-vocab",
        DATA_DIR / Path("output/best/vocab.pkl"),
    ]

    process = await asyncio.create_subprocess_exec(
        "python", "-m", "vid_cap.__main__", *args,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    
    stdout, _ = await process.communicate()
    output = str(stdout.strip())
    start_index = output.find("['")
    end_index = output.find("']")
    output = output[start_index + 2:end_index]
    return (output)


@app.route("/predict", methods=["POST"])
async def predict():
    """The input parameter is `review`"""
    review = request.form["review"]
    print(f"Prediction for review:\n {review}")

    result = await predict_review_sentiment(review)
    return render_template("result.html", result=result)


@app.route("/", methods=["GET"])
def hello():
    """ Return an HTML. """
    return render_template("hello.html")


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == "__main__":
    # Used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host="localhost", port=8080, debug=True)