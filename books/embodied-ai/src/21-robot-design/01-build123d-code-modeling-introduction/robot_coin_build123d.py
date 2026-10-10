from pathlib import Path

from build123d import (
    Box,
    BuildPart,
    Cylinder,
    Locations,
    Mode,
    Sphere,
    export_step,
    export_stl,
)


OUTPUT_DIR = Path(__file__).with_name("outputs")


def build_robot_coin():
    """Build a small robot badge/coin model with simple parametric solids."""

    with BuildPart() as robot_coin:
        # Base coin: a round 68 mm badge with a through-hole for a key ring.
        Cylinder(radius=34, height=4)
        with Locations((0, 25, 0)):
            Cylinder(radius=3.2, height=8, mode=Mode.SUBTRACT)

        # Robot face and body are raised features on top of the coin.
        with Locations((0, 0, 2.7)):
            Box(34, 22, 5)

        with Locations((-9, 3, 5.8), (9, 3, 5.8)):
            Cylinder(radius=2.2, height=1.4)

        with Locations((-13, -10, 4.2), (13, -10, 4.2)):
            Box(7, 12, 4)

        with Locations((-18, -5, 4.1), (18, -5, 4.1)):
            Box(6, 18, 3.2)

        # Antenna: horizontal cylinder plus a small sphere.
        with Locations((0, 17, 5.2)):
            Cylinder(radius=1.2, height=8, rotation=(90, 0, 0))

        with Locations((0, 22, 5.2)):
            Sphere(radius=2.2)

    return robot_coin.part


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = build_robot_coin()
    step_path = OUTPUT_DIR / "robot_coin.step"
    stl_path = OUTPUT_DIR / "robot_coin.stl"

    export_step(model, step_path)
    export_stl(model, stl_path)

    print("Build123d robot coin generated.")
    print(f"Bounding box: {model.bounding_box()}")
    print(f"STEP: {step_path}")
    print(f"STL:  {stl_path}")


if __name__ == "__main__":
    main()
