import os
import shlex
import subprocess
import time
from pathlib import Path

from flask import Flask, redirect, render_template_string, request, send_from_directory, url_for
from werkzeug.utils import secure_filename


APP = Flask(__name__)
ALLOWED_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv"}


def env_path(name, default=None):
    value = os.environ.get(name, default)
    return Path(value).expanduser().resolve() if value else None


def get_config():
    root = env_path("LINGBOT_MAP_ROOT")
    checkpoint = env_path("LINGBOT_MAP_CKPT")
    python_bin = env_path("LINGBOT_MAP_PYTHON", "python")
    run_root = env_path("LINGBOT_MAP_RUN_ROOT", "lingbot_map_runs")
    return root, checkpoint, python_bin, run_root


def validate_config():
    root, checkpoint, python_bin, run_root = get_config()
    errors = []
    if root is None or not (root / "demo_render" / "batch_demo.py").is_file():
        errors.append("LINGBOT_MAP_ROOT must point to a LingBot-Map checkout.")
    if checkpoint is None or not checkpoint.is_file():
        errors.append("LINGBOT_MAP_CKPT must point to a checkpoint file.")
    if python_bin is None:
        errors.append("LINGBOT_MAP_PYTHON is empty.")
    run_root.mkdir(parents=True, exist_ok=True)
    return errors, root, checkpoint, python_bin, run_root


def run_lingbot(video_path, output_dir, root, checkpoint, python_bin, target_frames):
    cmd = [
        str(python_bin),
        "demo_render/batch_demo.py",
        "--video_path",
        str(video_path),
        "--target_frames",
        str(target_frames),
        "--output_folder",
        str(output_dir),
        "--model_path",
        str(checkpoint),
        "--config",
        "demo_render/config/outdoor_large.yaml",
        "--num_scale_frames",
        "4",
        "--use_sdpa",
        "--video_width",
        "640",
        "--video_height",
        "360",
        "--video_fps",
        "12",
        "--camera_vis",
        "default",
        "--frame_tag",
        "--save_predictions",
        "--video_suffix",
        "_pointcloud",
    ]
    log_path = output_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(shlex.quote(part) for part in cmd) + "\n\n")
        process = subprocess.run(
            cmd,
            cwd=root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return process.returncode, log_path


@APP.route("/", methods=["GET", "POST"])
def index():
    errors, root, checkpoint, python_bin, run_root = validate_config()
    if request.method == "POST" and not errors:
        uploaded = request.files.get("video")
        target_frames = int(request.form.get("target_frames", "120"))
        target_frames = max(16, min(target_frames, 512))
        if uploaded is None or uploaded.filename == "":
            errors.append("Please choose a video file.")
        else:
            suffix = Path(uploaded.filename).suffix.lower()
            if suffix not in ALLOWED_SUFFIXES:
                errors.append("Unsupported video type.")
            else:
                run_id = time.strftime("%Y%m%d-%H%M%S")
                run_dir = run_root / run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                safe_name = secure_filename(uploaded.filename) or f"input{suffix}"
                video_path = run_dir / safe_name
                uploaded.save(video_path)
                return_code, _ = run_lingbot(
                    video_path, run_dir, root, checkpoint, python_bin, target_frames
                )
                return redirect(url_for("result", run_id=run_id, return_code=return_code))

    return render_template_string(INDEX_HTML, errors=errors)


@APP.route("/runs/<run_id>")
def result(run_id):
    _, _, _, _, run_root = validate_config()
    run_dir = (run_root / secure_filename(run_id)).resolve()
    if not run_dir.is_dir() or run_root not in run_dir.parents:
        return "Run not found", 404
    videos = sorted(path.name for path in run_dir.glob("*.mp4"))
    logs = (run_dir / "run.log").read_text(encoding="utf-8", errors="replace")[-8000:]
    return render_template_string(RESULT_HTML, run_id=run_id, videos=videos, logs=logs)


@APP.route("/runs/<run_id>/<path:filename>")
def run_file(run_id, filename):
    _, _, _, _, run_root = validate_config()
    run_dir = (run_root / secure_filename(run_id)).resolve()
    if not run_dir.is_dir() or run_root not in run_dir.parents:
        return "Run not found", 404
    return send_from_directory(run_dir, filename)


INDEX_HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LingBot-Map Web Demo</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 920px; margin: 40px auto; padding: 0 20px; }
    form { display: grid; gap: 16px; }
    input, button { font: inherit; padding: 10px; }
    button { cursor: pointer; }
    .error { color: #b00020; }
  </style>
</head>
<body>
  <h1>LingBot-Map Web Demo</h1>
  {% if errors %}
    <div class="error">
      {% for error in errors %}<p>{{ error }}</p>{% endfor %}
    </div>
  {% endif %}
  <form method="post" enctype="multipart/form-data">
    <label>Непрерывное видео <input type="file" name="video" accept="video/*" required></label>
    <label>Число кадров выборки <input type="number" name="target_frames" min="16" max="512" value="120"></label>
    <button type="submit">Начать реконструкцию</button>
  </form>
</body>
</html>
"""


RESULT_HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LingBot-Map Result</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 20px; }
    video { width: 100%; margin: 12px 0 28px; background: #111; }
    pre { white-space: pre-wrap; padding: 16px; background: #f5f5f5; overflow: auto; }
  </style>
</head>
<body>
  <h1>Run {{ run_id }}</h1>
  {% for video in videos %}
    <h2>{{ video }}</h2>
    <video controls muted preload="metadata" src="{{ url_for('run_file', run_id=run_id, filename=video) }}"></video>
  {% endfor %}
  <h2>Log</h2>
  <pre>{{ logs }}</pre>
  <p><a href="{{ url_for('index') }}">Назад</a></p>
</body>
</html>
"""


if __name__ == "__main__":
    APP.run(host="127.0.0.1", port=int(os.environ.get("PORT", "7860")), debug=False)
