import asyncio
import math
import os
import sys
import time
from pathlib import Path

import omni.kit.app
import omni.kit.asset_converter
import omni.usd
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
from pxr import Gf, Sdf, UsdGeom, UsdLux


async def _next_updates(n: int = 1):
    app = omni.kit.app.get_app()
    for _ in range(n):
        await app.next_update_async()


async def _convert_to_usd(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    context = omni.kit.asset_converter.AssetConverterContext()
    context.keep_all_materials = True
    context.merge_all_meshes = False
    context.embed_textures = True
    task = omni.kit.asset_converter.get_instance().create_converter_task(str(src), str(dst), None, context)
    return await task.wait_until_finished()


async def main(src_glb: str, out_png: str):
    src = Path(src_glb).resolve()
    out = Path(out_png).resolve()
    usd_path = out.with_suffix(".usd")

    ok = await _convert_to_usd(src, usd_path)
    if not ok or not usd_path.exists():
        raise RuntimeError(f"GLB to USD conversion failed: {src}")

    ctx = omni.usd.get_context()
    await ctx.open_stage_async(str(usd_path))
    await _next_updates(20)
    stage = ctx.get_stage()

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
    mins = [math.inf, math.inf, math.inf]
    maxs = [-math.inf, -math.inf, -math.inf]
    mesh_count = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh_count += 1
        box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        lo = box.GetMin()
        hi = box.GetMax()
        if any(not math.isfinite(v) for v in (*lo, *hi)):
            continue
        if any(hi[i] <= lo[i] for i in range(3)):
            continue
        for i in range(3):
            mins[i] = min(mins[i], lo[i])
            maxs[i] = max(maxs[i], hi[i])
    if mesh_count == 0 or any(not math.isfinite(v) for v in (*mins, *maxs)):
        raise RuntimeError(f"No renderable mesh bounds found in stage: {usd_path}")

    lo = Gf.Vec3d(*mins)
    hi = Gf.Vec3d(*maxs)
    center = Gf.Vec3d((lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, (lo[2] + hi[2]) * 0.5)
    extent = max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 1.0)
    print(f"MESH_COUNT={mesh_count}")
    print(f"BBOX_MIN={lo}")
    print(f"BBOX_MAX={hi}")

    light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/KeyLight"))
    light.CreateIntensityAttr(4500.0)
    light.CreateAngleAttr(0.35)

    camera = UsdGeom.Camera.Define(stage, Sdf.Path("/World/Camera"))
    eye = center + Gf.Vec3d(extent * 1.8, -extent * 2.4, extent * 1.3)
    view = Gf.Matrix4d().SetLookAt(eye, center, Gf.Vec3d(0, 0, 1))
    camera.AddTransformOp().Set(view.GetInverse())
    camera.CreateFocalLengthAttr(35.0)
    camera.CreateFocusDistanceAttr((eye - center).GetLength())

    viewport = get_active_viewport()
    viewport.camera_path = Sdf.Path("/World/Camera")
    await _next_updates(60)
    await viewport.wait_for_rendered_frames()

    out.parent.mkdir(parents=True, exist_ok=True)
    capture = capture_viewport_to_file(viewport, file_path=str(out))
    await capture.wait_for_result(0)
    await _next_updates(60)

    deadline = time.time() + 30
    while not out.exists() and time.time() < deadline:
        await _next_updates(5)
        await asyncio.sleep(0.25)
    if not out.exists():
        raise RuntimeError(f"Capture did not create output: {out}")
    print(f"USD={usd_path}")
    print(f"PNG={out}")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        input_glb, output_png = sys.argv[1], sys.argv[2]
    else:
        input_glb = os.environ.get("PHYSX_OMNI_RENDER_INPUT")
        output_png = os.environ.get("PHYSX_OMNI_RENDER_OUTPUT")
    if not input_glb or not output_png:
        raise SystemExit(
            "usage: render_glb_in_isaac.py <input.glb> <output.png>, "
            "or set PHYSX_OMNI_RENDER_INPUT/PHYSX_OMNI_RENDER_OUTPUT"
        )

    async def _run_and_quit():
        try:
            await main(input_glb, output_png)
        finally:
            omni.kit.app.get_app().post_quit()

    from omni.kit.async_engine import run_coroutine

    run_coroutine(_run_and_quit())
