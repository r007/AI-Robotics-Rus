from pathlib import Path

import vtk


ROOT = Path(__file__).resolve().parent
STL_PATH = ROOT / "outputs" / "robot_coin.stl"
ASSET_DIR = ROOT / "assets"
PNG_PATH = ASSET_DIR / "robot_coin_preview.png"


def main() -> None:
    if not STL_PATH.exists():
        raise FileNotFoundError(f"Missing STL file: {STL_PATH}")

    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(STL_PATH))

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1.0, 0.64, 0.08)
    actor.GetProperty().SetAmbient(0.34)
    actor.GetProperty().SetDiffuse(0.72)
    actor.GetProperty().SetSpecular(0.18)
    actor.GetProperty().SetSpecularPower(28)
    actor.GetProperty().EdgeVisibilityOn()
    actor.GetProperty().SetEdgeColor(0.18, 0.12, 0.02)
    actor.GetProperty().SetLineWidth(1.0)

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)
    renderer.AddActor(actor)
    renderer.ResetCamera()

    camera = renderer.GetActiveCamera()
    camera.ParallelProjectionOn()
    camera.SetPosition(80, -95, 72)
    camera.SetFocalPoint(0, 0, 2.2)
    camera.SetViewUp(0, 0, 1)
    camera.SetParallelScale(48)
    renderer.ResetCameraClippingRange()

    light = vtk.vtkLight()
    light.SetLightTypeToSceneLight()
    light.SetPosition(120, -140, 160)
    light.SetFocalPoint(0, 0, 0)
    light.SetIntensity(0.9)
    renderer.AddLight(light)

    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.AddRenderer(renderer)
    render_window.SetSize(1200, 900)
    render_window.Render()

    window_to_image = vtk.vtkWindowToImageFilter()
    window_to_image.SetInput(render_window)
    window_to_image.SetScale(1)
    window_to_image.SetInputBufferTypeToRGB()
    window_to_image.ReadFrontBufferOff()
    window_to_image.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(PNG_PATH))
    writer.SetInputConnection(window_to_image.GetOutputPort())
    writer.Write()

    print(f"Preview image written to: {PNG_PATH}")


if __name__ == "__main__":
    main()
