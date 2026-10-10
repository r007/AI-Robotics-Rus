"""Reproducible LightNav navigation lessons using the upstream MicroDuck backend."""

from __future__ import annotations

import base64
import copy
import io
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np


def ensure_model_server(model_env, model_path, output, port=8050, timeout=480):
    """Start a local GPU model service if the selected port is not listening."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    def listening():
        with socket.socket() as sock:
            sock.settimeout(0.5)
            return sock.connect_ex(("127.0.0.1", port)) == 0

    if listening():
        return {"url": f"ws://127.0.0.1:{port}", "state": "existing listener"}
    model_env, model_path = Path(model_env), Path(model_path)
    for name in ("config.json", "eval_config.json", "processor_config.json"):
        if not (model_path / name).is_file():
            raise FileNotFoundError(model_path / name)
    if not (model_path / "action_tokenizer").is_dir():
        raise FileNotFoundError(model_path / "action_tokenizer")
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", VLN_KV_CACHE_GIB="2")
    log = output / "model_service.log"
    args = [
        str(model_env / "bin/lightnav-serve"), "--task", "vln", "--backend", "vllm_local",
        "--model_path", str(model_path), "--host", "127.0.0.1", "--port", str(port),
        "--gpu_memory_utilization", os.getenv("LIGHTNAV_GPU_MEMORY_UTILIZATION", "0.22"),
        "--max_batch_size", "1", "--ready_file", str(output / "model.ready"),
    ]
    with log.open("ab") as stream:
        process = subprocess.Popen(
            args, env=env, stdout=stream, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Model service exited: {log.read_text(errors='replace')[-2500:]}")
        if listening():
            return {"url": f"ws://127.0.0.1:{port}", "state": "started", "pid": process.pid}
        time.sleep(2)
    # This process belongs to this call; a timeout must not leave a loading job behind.
    os.killpg(process.pid, 15)
    process.wait(timeout=30)
    raise TimeoutError(f"Model service did not become ready; inspect {log}")


def _box(world, name, position, size, color, material=None):
    attrs = {"name": name, "type": "box", "pos": " ".join(map(str, position)),
             "size": " ".join(map(str, size)), "rgba": color,
             "contype": "1", "conaffinity": "1", "friction": "0.8 0.01 0.001"}
    if material:
        attrs["material"] = material
    ET.SubElement(world, "geom", attrs)


def build_scenes(output):
    """Keep the official apartment and build two new furnished room layouts."""
    from vln_mujoco.model import load_scene_xml, scene_path

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    official = load_scene_xml()
    bodies = official.findall("./worldbody/body")

    def furniture(prefix, x, y, name):
        source = next(body for body in bodies if body.get("name", "").startswith(prefix))
        body = copy.deepcopy(source)
        position = list(map(float, body.get("pos").split()))
        position[:2] = [x, y]
        body.set("pos", " ".join(map(str, position)))
        body.set("name", name)
        return body

    def room(name, bounds, objects):
        root = copy.deepcopy(official)
        for section in ("worldbody", "contact", "equality", "sensor", "actuator", "keyframe"):
            item = root.find(section)
            if item is not None:
                root.remove(item)
        world = ET.SubElement(root, "worldbody")
        ET.SubElement(world, "light", pos="7.5 13.8 3.0", dir="0 0 -1", directional="false")
        asset = root.find("asset")
        ET.SubElement(asset, "texture", name="lesson_floor_texture", type="2d", builtin="checker",
                      width="256", height="256", rgb1="0.68 0.72 0.74", rgb2="0.82 0.85 0.86")
        ET.SubElement(asset, "material", name="lesson_floor_material", texture="lesson_floor_texture",
                      texrepeat="8 8", texuniform="true")
        x0, x1, y0, y1 = bounds
        _box(world, "lesson_floor", ((x0+x1)/2, (y0+y1)/2, -0.05),
             ((x1-x0)/2, (y1-y0)/2, 0.05), "1 1 1 1", "lesson_floor_material")
        _box(world, "wall_back", (x0, (y0+y1)/2, 1.2), (0.06, (y1-y0)/2, 1.2), "0.9 0.93 0.94 1")
        _box(world, "wall_front", (x1, (y0+y1)/2, 1.2), (0.06, (y1-y0)/2, 1.2), "0.9 0.93 0.94 1")
        for side, y in (("left", y1), ("right", y0)):
            _box(world, f"wall_{side}", ((x0+x1)/2, y, 1.2), ((x1-x0)/2, 0.06, 1.2), "0.84 0.9 0.92 1")
        for prefix, x, y, label in objects:
            world.append(furniture(prefix, x, y, label))
        root.set("model", name)
        return root

    scenes = {
        "official_apartment": {"root": official, "instruction": "Walk to the table on your left and stop.",
            "target_xy": [8.67515, 15.7876], "description": "Official ProcTHOR val_2 apartment"},
        "living_room": {"root": room("learning_living_room", (5.6, 10.3, 11.8, 16.1), [
            ("armchair_", 8.3, 14.0, "goal_armchair"), ("sofa_", 7.0, 15.45, "side_sofa"),
            ("houseplant_", 9.5, 12.45, "corner_plant")]),
            "instruction": "Walk to the armchair in front of you and stop.",
            "target_xy": [8.3, 14.0], "description": "New open living-room layout"},
        "corridor": {"root": room("learning_corridor", (5.6, 10.5, 12.75, 14.85), [
            ("houseplant_", 9.2, 13.8, "goal_plant"), ("stool_", 7.6, 14.4, "side_stool")]),
            "instruction": "Go straight down the hallway to the potted plant and stop.",
            "target_xy": [9.2, 13.8], "description": "New corridor with side furniture"},
    }
    manifest = {"upstream_scene": str(scene_path()), "asset_license": "CC BY 4.0",
                "modifications": "New room walls, floor and furniture placement; original meshes retained.",
                "scenes": {}}
    for key, scene in scenes.items():
        path = output / f"{key}.xml"
        ET.ElementTree(scene["root"]).write(path, encoding="utf-8", xml_declaration=True)
        scene["path"] = path
        manifest["scenes"][key] = {k: v for k, v in scene.items() if k not in {"root", "path"}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return scenes


def make_robot(scene, robot_model, walking_policy):
    from vln_mujoco.robots import microduck

    # Assets were validated by upstream load_scene_xml before layouts were changed.
    # The scoped adapter lets its fixed-scene constructor consume the new XML root.
    with patch.object(microduck, "load_scene_xml", lambda *_: copy.deepcopy(scene["root"])):
        return microduck.MicroDuckBackend(Path(robot_model), Path(walking_policy))


def decode_response(message, sequence):
    payload = json.loads(message)
    data = payload.get("data", {})
    if payload.get("action") != "next" or data.get("rc") != 0 or data.get("seq") != sequence:
        raise RuntimeError(f"Invalid inference response: {payload}")
    actions = data.get("actions")
    if isinstance(actions, dict):
        actions = actions.get("actions")
    trajectory = np.asarray(actions, dtype=np.float64)
    if trajectory.shape != (10, 3) or not np.isfinite(trajectory).all():
        raise RuntimeError(f"Expected a finite 10x3 trajectory, got {trajectory.shape}")
    if type(data.get("stop")) is not bool:
        raise RuntimeError("Missing Boolean model stop decision")
    return data, trajectory


def camera_image(robot, renderer, third_person=False):
    import mujoco

    if third_person:
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        robot.third_person_camera(camera)
        camera.distance = 1.3
        camera.azimuth = 155
        camera.elevation = -25
    else:
        camera = robot.camera_name
    renderer.update_scene(robot.data, camera=camera)
    return renderer.render().copy()


def _annotated_frame(robot, renderer, label, elapsed, prediction):
    from PIL import Image, ImageDraw

    first, third = camera_image(robot, renderer), camera_image(robot, renderer, True)
    canvas = Image.fromarray(np.concatenate((first, third), axis=1))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 24), fill=(20, 24, 26))
    draw.text((8, 6), f"{label} | LightNav GPU | simulation {elapsed:.1f}s", fill="white")
    if prediction:
        point = (prediction.get("pointing") or {}).get("opos_px")
        if isinstance(point, (list, tuple)) and len(point) == 2:
            x, y = map(float, point)
            draw.ellipse((x-5, y-5, x+5, y+5), outline="#ff542b", width=2)
    return np.asarray(canvas)


def run_episode(scene_id, scene, robot_model, walking_policy, server_url, output,
                seconds=24, prediction_period=0.25, fps=10):
    """Record real GPU predictions and contact-based locomotion in simulation time.

    Physics pauses during network inference. Targets are used for measurement only;
    controls come from LightNav's RGB trajectory and the upstream MPC tracker.
    """
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image
    import websocket
    from vln_mujoco.mpc import (MPCController, build_pose_aligned_reference,
                               project_body_to_world, Q_WEIGHTS, R_WEIGHTS)

    if seconds <= 0 or prediction_period <= 0 or fps <= 0:
        raise ValueError("Episode duration, prediction period and fps must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    robot = make_robot(scene, robot_model, walking_policy)
    renderer = mujoco.Renderer(robot.model, height=270, width=480)
    controller = MPCController(horizon=5, dt_s=0.1, w_max=1.2, a_max_v=2.0, a_max_w=5.0,
                               q_weights=Q_WEIGHTS, r_weights=R_WEIGHTS)
    video_path = output / f"{scene_id}_latest.mp4"
    writer = imageio.get_writer(video_path, fps=fps, codec="libx264", quality=7,
                               macro_block_size=2, ffmpeg_params=["-movflags", "+faststart"])
    connection = None
    predictions, trace, previews = [], [], []
    termination = "time_limit"
    command = (0.0, 0.0)
    world_path = None
    latest = None
    next_prediction = next_control = next_frame = 0.0
    start_wall = time.monotonic()
    initial = np.asarray(robot.state().pose[:2])
    start_time = float(robot.data.time)
    sequence = 0
    try:
        # Settle with the exact same pretrained low-level walking policy.
        for _ in range(100):
            robot.step((0.0, 0.0))
        start_time = float(robot.data.time)
        connection = websocket.create_connection(server_url, timeout=90, http_no_proxy=["127.0.0.1", "localhost"])
        connection.send(json.dumps({"action": "login", "data": {"clientId": f"lesson-{scene_id}"}}))
        login = json.loads(connection.recv())
        if login.get("data", {}).get("rc") != 0:
            raise RuntimeError(f"Model login failed: {login}")
        while float(robot.data.time) - start_time < seconds:
            elapsed = float(robot.data.time) - start_time
            state = robot.state()
            pose = np.array([state.pose[0], state.pose[1], state.pose[3]])
            if elapsed + 1e-8 >= next_prediction:
                rgb = camera_image(robot, renderer)
                jpeg = io.BytesIO()
                Image.fromarray(rgb).save(jpeg, format="JPEG", quality=90)
                sequence += 1
                sent = time.monotonic()
                connection.send(json.dumps({"action": "next", "data": {
                    "seq": sequence, "image": base64.b64encode(jpeg.getvalue()).decode("ascii"),
                    "instruction": scene["instruction"]}}))
                latest, body_path = decode_response(connection.recv(), sequence)
                latency = 1000 * (time.monotonic() - sent)
                world_path = project_body_to_world(body_path, pose)
                predictions.append({"sim_time": round(elapsed, 3), "latency_ms": latency,
                                    "pose": pose.tolist(), **latest})
                next_prediction += prediction_period
                if latest["stop"]:
                    command = (0.0, 0.0)
                    termination = "model_stop"
                    writer.append_data(_annotated_frame(robot, renderer, scene_id, elapsed, latest))
                    break
            if elapsed + 1e-8 >= next_control and world_path is not None:
                reference = build_pose_aligned_reference(world_path, pose, horizon=5, weights=Q_WEIGHTS)
                command, _ = controller.solve(pose, reference, command, v_max=0.30)
                next_control += 0.1
            if elapsed + 1e-8 >= next_frame:
                frame = _annotated_frame(robot, renderer, scene_id, elapsed, latest)
                writer.append_data(frame)
                if not previews:
                    previews.append(frame)
                trace.append({"sim_time": elapsed, "x": state.pose[0], "y": state.pose[1],
                              "z": state.pose[2], "v": command[0], "yaw_rate": command[1]})
                next_frame += 1 / fps
            if state.pose[2] < 0.065 or not np.isfinite(robot.data.qpos).all():
                termination = "fall"
                break
            robot.step(command)
        # Record a genuine stationary tail after a stop; no reset or target teleport.
        for step in range(200):
            robot.step((0.0, 0.0))
            if step % 20 == 0:
                frame = _annotated_frame(robot, renderer, scene_id, robot.data.time-start_time, latest)
                writer.append_data(frame)
        previews.append(_annotated_frame(robot, renderer, scene_id, robot.data.time-start_time, latest))
    finally:
        if connection is not None:
            connection.close()
        writer.close()
        renderer.close()
    if not predictions:
        raise RuntimeError("No real model prediction was recorded")
    final = robot.state().pose
    target = np.asarray(scene["target_xy"])
    report = {"scene": scene_id, "instruction": scene["instruction"],
              "navigation_backend": "LightNav-0 / vllm_local / GPU",
              "walking_backend": "pretrained alpha_walking / ONNX Runtime CPU",
              "time_mode": "simulation pauses during inference", "training": False,
              "prediction_count": len(predictions), "trajectory_shape": [10, 3],
              "termination": termination, "sim_seconds": float(robot.data.time)-start_time,
              "wall_seconds": time.monotonic()-start_wall,
              "median_latency_ms": float(np.median([p["latency_ms"] for p in predictions])),
              "initial_target_distance_m": float(np.linalg.norm(initial-target)),
              "final_target_distance_m": float(np.linalg.norm(np.array(final[:2])-target)),
              "final_trunk_height_m": final[2], "video": str(video_path)}
    (output / f"{scene_id}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / f"{scene_id}_predictions.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    (output / f"{scene_id}_trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
    return report, previews
