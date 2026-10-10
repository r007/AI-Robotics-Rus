from math import cos, pi, sin

from build123d import Box, BuildPart, Cylinder, Locations, Mode, Sphere


def gen_step():
    """Generate an original simplified 6DOF desktop robot arm teaching model.

    Coordinate convention:
    - XY is the desktop plane.
    - +Z points upward.
    - The model is a visual/mechanical teaching assembly, not a certified
      manufacturable robot arm.
    """

    base_radius = 42.0
    base_thickness = 8.0
    bolt_circle_radius = 30.0
    bolt_hole_radius = 2.8

    with BuildPart() as arm:
        # Base plate with four mounting holes.
        Cylinder(radius=base_radius, height=base_thickness)
        for i in range(4):
            angle = i * pi / 2.0 + pi / 4.0
            x = bolt_circle_radius * cos(angle)
            y = bolt_circle_radius * sin(angle)
            with Locations((x, y, 0)):
                Cylinder(radius=bolt_hole_radius, height=base_thickness + 4, mode=Mode.SUBTRACT)

        # Joint 1: vertical yaw stack.
        with Locations((0, 0, 13)):
            Cylinder(radius=24, height=18)
        with Locations((0, 0, 26)):
            Cylinder(radius=16, height=10)

        # Shoulder side plates and Joint 2 axis.
        with Locations((-18, 0, 47), (18, 0, 47)):
            Box(8, 34, 38)
        with Locations((0, 0, 55)):
            Cylinder(radius=13, height=52, rotation=(90, 0, 0))

        # Upper arm link.
        with Locations((52, 0, 68)):
            Box(80, 18, 14)
        with Locations((52, 0, 68)):
            Box(70, 10, 24)

        # Joint 3: elbow pitch axis.
        with Locations((96, 0, 68)):
            Cylinder(radius=13, height=48, rotation=(90, 0, 0))
        with Locations((96, 0, 68)):
            Sphere(radius=15)

        # Forearm link with a raised cable-guide ridge.
        with Locations((138, 0, 86)):
            Box(78, 16, 12)
        with Locations((138, 0, 96)):
            Box(64, 6, 8)

        # Wrist cluster: Joints 4, 5, and 6 use three visible orthogonal axes.
        with Locations((182, 0, 86)):
            Cylinder(radius=10, height=34, rotation=(0, 90, 0))
        with Locations((198, 0, 86)):
            Cylinder(radius=8, height=30, rotation=(90, 0, 0))
        with Locations((214, 0, 86)):
            Cylinder(radius=7, height=24)

        # Tool flange and a simple two-finger gripper placeholder.
        with Locations((226, 0, 86)):
            Cylinder(radius=13, height=7, rotation=(0, 90, 0))
        with Locations((238, -7, 86), (238, 7, 86)):
            Box(24, 4, 8)
        with Locations((250, -7, 86), (250, 7, 86)):
            Box(6, 4, 18)

    model = arm.part
    model.label = "every_embodied_6dof_teaching_arm"
    return model
