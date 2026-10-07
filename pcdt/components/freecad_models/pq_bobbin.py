# -*- coding: utf-8 -*-
"""
Generate a round PQ-style bobbin and export it as STEP using FreeCAD.

All dimensions are in mm.
The bobbin consists of:
- round center-leg hole
- cylindrical winding barrel
- circular top and bottom flanges
- rounded inner hole edges
- rounded outer top and bottom flange edges
- optional wire exit slots on the positive Y side

Environment variables
---------------------
WINDOW_H_MM
WINDOW_W_MM
CORE_INNER_DIAMETER_MM
FLANGE_THICKNESS_INNER_MM
FLANGE_THICKNESS_TOP_MM
FLANGE_THICKNESS_BOT_MM
CLEARANCE
INNER_EDGE_RADIUS
OUTER_EDGE_RADIUS
ENABLE_WIRE_SLOTS
WIRE_SLOT_WIDTH
WIRE_SLOT_DEPTH
WIRE_SLOT_HEIGHT
WIRE_SLOTS_POSITION
OUTPUT_STEP_FILE
SAVE_FCSTD_FILE

Example:
WINDOW_H_MM=17.2 \
WINDOW_W_MM=7.0 \
CORE_INNER_DIAMETER_MM=13.45 \
FLANGE_THICKNESS_INNER_MM=1.0 \
FLANGE_THICKNESS_TOP_MM=2.5 \
FLANGE_THICKNESS_BOT_MM=1.5 \
OUTPUT_STEP_FILE="./pq_bobbin.step" \
FreeCADCmd pq_bobbin.py
"""

import os
import sys
import logging

import FreeCAD as App
import Part
import Import


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def read_float_environment_variable(variable_name: str, default_value: float) -> float:
    """
    Read a float from an environment variable.

    :param variable_name: variable name
    :type variable_name: str
    :param default_value: default value
    :type default_value: str
    """
    value = os.environ.get(variable_name)

    if value is None or value.strip() == "":
        return float(default_value)

    try:
        return float(value)
    except ValueError as error:
        raise ValueError(
            f"Environment variable {variable_name} must be numeric, "
            f"but received {value!r}."
        ) from error


def read_bool_environment_variable(variable_name: str, default_value: bool) -> bool:
    """
    Read a boolean from an environment variable.

    :param variable_name: variable name
    :type variable_name: str
    :param default_value: default value
    :type default_value: str
    """
    value = os.environ.get(variable_name)

    if value is None or value.strip() == "":
        return bool(default_value)

    value = value.strip().lower()

    if value in ("1", "true", "yes", "on"):
        return True

    if value in ("0", "false", "no", "off"):
        return False

    raise ValueError(
        f"Environment variable {variable_name} must be boolean, "
        f"but received {value!r}."
    )


def read_string_environment_variable(variable_name: str, default_value: str) -> str:
    """
    Read a string from an environment variable.

    :param variable_name: variable name
    :type variable_name: str
    :param default_value: default value
    :type default_value: str
    """
    value = os.environ.get(variable_name)

    if value is None or value.strip() == "":
        return default_value

    return value.strip()


def vector_from_rz(radius_mm: float, z_mm: float) -> App.Vector:
    """
    Create a point in the X/Z profile plane.

    X is radial distance from the Z axis, Y is always zero.

    :param radius_mm: radius in mm
    :type radius_mm: float
    :param z_mm: z coordinate in mm
    :type z_mm: float
    :return: FreeCAD vector in the X/Z profile plane.
    :rtype: App.Vector
    """
    return App.Vector(radius_mm, 0.0, z_mm)


def make_arc_edge(
    start_point: App.Vector,
    middle_point: App.Vector,
    end_point: App.Vector
) -> object:
    """
    Create a circular arc through three points.

    :param start_point: start point
    :type start_point: App.Vector
    :param middle_point: middle point
    :type middle_point: App.Vector
    :param end_point: end point
    :type end_point: App.Vector
    """
    return Part.Arc(
        start_point,
        middle_point,
        end_point
    ).toShape()


# ---------------------------------------------------------------------------
# Geometry creation
# ---------------------------------------------------------------------------

def create_round_bobbin(
    window_h_mm: float,
    window_w_mm: float,
    core_inner_diameter_mm: float,
    flange_thickness_inner_mm: float,
    flange_thickness_top_mm: float,
    flange_thickness_bot_mm: float,
    clearance: float,
    inner_edge_radius_mm: float,
    outer_edge_radius_mm: float,
    enable_wire_slots: bool,
    wire_slot_width_mm: float,
    wire_slot_depth_mm: float,
    wire_slot_height_mm: float,
    wire_slots_position: str
) -> object:
    """
    Create the bobbin geometry.

    flange_thickness_inner_mm controls the radial wall thickness of the
    central tube:

        barrel_outer_radius =
            center_hole_radius + flange_thickness_inner_mm

    flange_thickness_top_mm and flange_thickness_bot_mm control the
    respective axial flange thicknesses.


    :param window_h_mm: window height in mm
    :type window_h_mm: float
    :param window_w_mm: window height in mm
    :type window_w_mm: float
    :param core_inner_diameter_mm: core inner diameter in mm
    :type core_inner_diameter_mm: float
    :param flange_thickness_inner_mm: inner flange thickness in mm
    :type flange_thickness_inner_mm: float
    :param flange_thickness_top_mm: top flange thickness in mm
    :type flange_thickness_top_mm: float
    :param flange_thickness_bot_mm: bottom flange thickness in mm
    :type flange_thickness_bot_mm: float
    :param clearance: clearance in mm
    :type clearance: float
    :param inner_edge_radius_mm: inner edge radius in mm
    :type inner_edge_radius_mm: float
    :param outer_edge_radius_mm: outer edge radius in mm
    :type outer_edge_radius_mm: float
    :param enable_wire_slots: True to enable wire slots
    :type enable_wire_slots: bool
    :param wire_slot_width_mm: wire slot width in mm
    :type wire_slot_width_mm: float
    :param wire_slot_depth_mm: wire slot depth in mm
    :type wire_slot_depth_mm: float
    :param wire_slot_height_mm: wire slot height in mm
    :type wire_slot_height_mm: float
    :param wire_slots_position: wire slot position
    :type wire_slots_position: str
    """
    # -----------------------------------------------------------------------
    # Input validation
    # -----------------------------------------------------------------------

    if window_h_mm <= 0:
        raise ValueError("window_h_mm must be greater than 0.")

    if window_w_mm <= 0:
        raise ValueError("window_w_mm must be greater than 0.")

    if core_inner_diameter_mm <= 0:
        raise ValueError(
            "core_inner_diameter_mm must be greater than 0."
        )

    if flange_thickness_inner_mm <= 0:
        raise ValueError(
            "flange_thickness_inner_mm must be greater than 0."
        )

    if flange_thickness_top_mm <= 0:
        raise ValueError(
            "flange_thickness_top_mm must be greater than 0."
        )

    if flange_thickness_bot_mm <= 0:
        raise ValueError(
            "flange_thickness_bot_mm must be greater than 0."
        )

    if clearance < 0:
        raise ValueError("clearance must not be negative.")

    if inner_edge_radius_mm < 0:
        raise ValueError(
            "inner_edge_radius_mm must not be negative."
        )

    if outer_edge_radius_mm < 0:
        raise ValueError(
            "outer_edge_radius_mm must not be negative."
        )

    wire_slots_position = wire_slots_position.strip().lower()

    if wire_slots_position not in ("both", "top", "bottom", "none"):
        raise ValueError(
            "wire_slots_position must be one of: "
            "'both', 'top', 'bottom', 'none'."
        )

    # -----------------------------------------------------------------------
    # Derived dimensions
    # -----------------------------------------------------------------------

    bobbin_height_mm = window_h_mm - 2.0 * clearance

    center_hole_diameter_mm = (
        core_inner_diameter_mm + 2.0 * clearance
    )
    center_hole_radius_mm = center_hole_diameter_mm / 2.0

    # This is the actual material thickness of the central cylindrical tube.
    barrel_outer_radius_mm = (center_hole_radius_mm + flange_thickness_inner_mm)

    winding_height_mm = (bobbin_height_mm - flange_thickness_top_mm - flange_thickness_bot_mm)

    flange_outer_diameter_mm = (core_inner_diameter_mm + 2.0 * window_w_mm - 2.0 * clearance)
    flange_outer_radius_mm = flange_outer_diameter_mm / 2.0

    z_min_mm = -bobbin_height_mm / 2.0
    z_max_mm = bobbin_height_mm / 2.0

    z_barrel_bottom_mm = (
        z_min_mm + flange_thickness_bot_mm
    )
    z_barrel_top_mm = (
        z_max_mm - flange_thickness_top_mm
    )

    # -----------------------------------------------------------------------
    # Geometry validation
    # -----------------------------------------------------------------------

    if bobbin_height_mm <= 0:
        raise ValueError(
            "Invalid bobbin height. WINDOW_H_MM must be greater than "
            "2 * CLEARANCE."
        )

    if winding_height_mm <= 0:
        raise ValueError(
            "Invalid winding height. The sum of "
            "FLANGE_THICKNESS_TOP_MM and FLANGE_THICKNESS_BOT_MM "
            "must be smaller than the usable bobbin height."
        )

    if barrel_outer_radius_mm <= center_hole_radius_mm:
        raise ValueError(
            "Invalid geometry: central tube outer radius must be larger "
            "than the center-hole radius."
        )

    if flange_outer_radius_mm <= barrel_outer_radius_mm:
        raise ValueError(
            "Invalid geometry: flange outer radius must be larger than "
            "the winding-barrel outer radius."
        )

    radial_tube_wall_mm = (
        barrel_outer_radius_mm - center_hole_radius_mm
    )

    radial_flange_overhang_mm = (
        flange_outer_radius_mm - barrel_outer_radius_mm
    )

    # Each flange has independent rounding radii, since its thickness may
    # be different from the other flange.
    inner_roundover_radius_bot_mm = min(
        inner_edge_radius_mm,
        flange_thickness_bot_mm * 0.95,
        radial_tube_wall_mm,
        winding_height_mm / 2.0
    )

    inner_roundover_radius_top_mm = min(
        inner_edge_radius_mm,
        flange_thickness_top_mm * 0.95,
        radial_tube_wall_mm,
        winding_height_mm / 2.0
    )

    outer_roundover_radius_bot_mm = min(
        outer_edge_radius_mm,
        flange_thickness_bot_mm * 0.95,
        radial_flange_overhang_mm
    )

    outer_roundover_radius_top_mm = min(
        outer_edge_radius_mm,
        flange_thickness_top_mm * 0.95,
        radial_flange_overhang_mm
    )

    if inner_roundover_radius_bot_mm <= 0:
        raise ValueError(
            "Computed bottom inner roundover radius is invalid."
        )

    if inner_roundover_radius_top_mm <= 0:
        raise ValueError(
            "Computed top inner roundover radius is invalid."
        )

    if outer_roundover_radius_bot_mm <= 0:
        raise ValueError(
            "Computed bottom outer roundover radius is invalid."
        )

    if outer_roundover_radius_top_mm <= 0:
        raise ValueError(
            "Computed top outer roundover radius is invalid."
        )

    # -----------------------------------------------------------------------
    # Build radial profile
    # -----------------------------------------------------------------------

    # Bottom horizontal surface: bore side -> flange exterior.
    p0 = vector_from_rz(
        center_hole_radius_mm + inner_roundover_radius_bot_mm,
        z_min_mm
    )

    p1 = vector_from_rz(
        flange_outer_radius_mm - outer_roundover_radius_bot_mm,
        z_min_mm
    )

    # Bottom flange outside rounding.
    p2 = vector_from_rz(
        flange_outer_radius_mm,
        z_min_mm + outer_roundover_radius_bot_mm
    )

    p_bottom_outer_mid = vector_from_rz(
        flange_outer_radius_mm - outer_roundover_radius_bot_mm + outer_roundover_radius_bot_mm * 0.7071067811865476,
        z_min_mm + outer_roundover_radius_bot_mm - outer_roundover_radius_bot_mm * 0.7071067811865476
    )

    # Outer edge of lower flange.
    p3 = vector_from_rz(
        flange_outer_radius_mm,
        z_barrel_bottom_mm
    )

    # Transition from lower flange to central tube.
    p4 = vector_from_rz(
        barrel_outer_radius_mm,
        z_barrel_bottom_mm
    )

    # Outer surface of the cylindrical central tube / winding barrel.
    p5 = vector_from_rz(
        barrel_outer_radius_mm,
        z_barrel_top_mm
    )

    # Transition from central tube to upper flange.
    p6 = vector_from_rz(
        flange_outer_radius_mm,
        z_barrel_top_mm
    )

    # Outer upper-flange edge below roundover.
    p7 = vector_from_rz(
        flange_outer_radius_mm,
        z_max_mm - outer_roundover_radius_top_mm
    )

    # Upper flange outside rounding.
    p8 = vector_from_rz(
        flange_outer_radius_mm - outer_roundover_radius_top_mm,
        z_max_mm
    )

    p_top_outer_mid = vector_from_rz(
        flange_outer_radius_mm - outer_roundover_radius_top_mm + outer_roundover_radius_top_mm * 0.7071067811865476,
        z_max_mm - outer_roundover_radius_top_mm + outer_roundover_radius_top_mm * 0.7071067811865476
    )

    # Top horizontal surface: flange exterior -> bore side.
    p9 = vector_from_rz(
        center_hole_radius_mm + inner_roundover_radius_top_mm,
        z_max_mm
    )

    # Upper edge roundover of the center hole.
    p10 = vector_from_rz(
        center_hole_radius_mm,
        z_max_mm - inner_roundover_radius_top_mm
    )

    p_top_inner_mid = vector_from_rz(
        center_hole_radius_mm + inner_roundover_radius_top_mm - inner_roundover_radius_top_mm * 0.7071067811865476,
        z_max_mm - inner_roundover_radius_top_mm + inner_roundover_radius_top_mm * 0.7071067811865476
    )

    # Inside wall of the bore.
    p11 = vector_from_rz(
        center_hole_radius_mm,
        z_min_mm + inner_roundover_radius_bot_mm
    )

    # Lower edge roundover of the center hole.
    p_bottom_inner_mid = vector_from_rz(
        center_hole_radius_mm + inner_roundover_radius_bot_mm - inner_roundover_radius_bot_mm * 0.7071067811865476,
        z_min_mm + inner_roundover_radius_bot_mm - inner_roundover_radius_bot_mm * 0.7071067811865476
    )

    profile_edges = [
        Part.makeLine(p0, p1),
        make_arc_edge(p1, p_bottom_outer_mid, p2),
        Part.makeLine(p2, p3),
        Part.makeLine(p3, p4),
        Part.makeLine(p4, p5),
        Part.makeLine(p5, p6),
        Part.makeLine(p6, p7),
        make_arc_edge(p7, p_top_outer_mid, p8),
        Part.makeLine(p8, p9),
        make_arc_edge(p9, p_top_inner_mid, p10),
        Part.makeLine(p10, p11),
        make_arc_edge(p11, p_bottom_inner_mid, p0)
    ]

    profile_wire = Part.Wire(profile_edges)

    if not profile_wire.isClosed():
        raise RuntimeError(
            "Internal error: bobbin profile wire is not closed."
        )

    profile_face = Part.Face(profile_wire)

    bobbin_shape = profile_face.revolve(
        App.Vector(0.0, 0.0, 0.0),
        App.Vector(0.0, 0.0, 1.0),
        360.0
    )

    # -----------------------------------------------------------------------
    # Optional wire slots
    # -----------------------------------------------------------------------

    if enable_wire_slots and wire_slots_position != "none":
        if wire_slot_width_mm <= 0:
            raise ValueError(
                "wire_slot_width_mm must be greater than 0."
            )

        if wire_slot_depth_mm <= 0:
            raise ValueError(
                "wire_slot_depth_mm must be greater than 0."
            )

        if wire_slot_height_mm <= 0:
            raise ValueError(
                "wire_slot_height_mm must be greater than 0."
            )

        slot_x_min_mm = -wire_slot_width_mm / 2.0

        # The cutter starts 0.2 mm outside the outer flange perimeter.
        slot_y_min_mm = (flange_outer_radius_mm - wire_slot_depth_mm - 0.2)
        slot_y_length_mm = wire_slot_depth_mm + 0.4

        def make_wire_slot(slot_z_center_mm: float) -> object:
            """
            Create a rectangular solid for cutting a wire slot.

            :param slot_z_center_mm: z-distance to center in mm
            :type slot_z_center_mm: float
            """
            slot_z_min_mm = (slot_z_center_mm - wire_slot_height_mm / 2.0)

            return Part.makeBox(
                wire_slot_width_mm,
                slot_y_length_mm,
                wire_slot_height_mm,
                App.Vector(
                    slot_x_min_mm,
                    slot_y_min_mm,
                    slot_z_min_mm
                )
            )

        slot_cutters = []

        if wire_slots_position in ("bottom", "both"):
            bottom_slot_z_center_mm = (
                z_min_mm + flange_thickness_bot_mm / 2.0
            )
            slot_cutters.append(
                make_wire_slot(bottom_slot_z_center_mm)
            )

        if wire_slots_position in ("top", "both"):
            top_slot_z_center_mm = (
                z_max_mm - flange_thickness_top_mm / 2.0
            )
            slot_cutters.append(
                make_wire_slot(top_slot_z_center_mm)
            )

        for slot_cutter in slot_cutters:
            bobbin_shape = bobbin_shape.cut(slot_cutter)

    bobbin_shape = bobbin_shape.removeSplitter()

    if bobbin_shape.isNull():
        raise RuntimeError("Generated bobbin shape is empty.")

    if not bobbin_shape.isValid():
        raise RuntimeError("Generated bobbin shape is invalid.")

    return bobbin_shape


# ---------------------------------------------------------------------------
# STEP export
# ---------------------------------------------------------------------------

def export_round_bobbin_step(
    window_h_mm: float,
    window_w_mm: float,
    core_inner_diameter_mm: float,
    flange_thickness_inner_mm: float,
    flange_thickness_top_mm: float,
    flange_thickness_bot_mm: float,
    clearance: float,
    inner_edge_radius_mm: float,
    outer_edge_radius_mm: float,
    enable_wire_slots: bool,
    wire_slot_width_mm: float,
    wire_slot_depth_mm: float,
    wire_slot_height_mm: float,
    wire_slots_position: str,
    output_step_file: str,
    save_freecad_file: bool = False
) -> str:
    """
    Create the bobbin and export it to a STEP file.

    :param window_h_mm: window height in mm
    :type window_h_mm: float
    :param window_w_mm: window height in mm
    :type window_w_mm: float
    :param core_inner_diameter_mm: core inner diameter in mm
    :type core_inner_diameter_mm: float
    :param flange_thickness_inner_mm: inner flange thickness in mm
    :type flange_thickness_inner_mm: float
    :param flange_thickness_top_mm: top flange thickness in mm
    :type flange_thickness_top_mm: float
    :param flange_thickness_bot_mm: bottom flange thickness in mm
    :type flange_thickness_bot_mm: float
    :param clearance: clearance in mm
    :type clearance: float
    :param inner_edge_radius_mm: inner edge radius in mm
    :type inner_edge_radius_mm: float
    :param outer_edge_radius_mm: outer edge radius in mm
    :type outer_edge_radius_mm: float
    :param enable_wire_slots: True to enable wire slots
    :type enable_wire_slots: bool
    :param wire_slot_width_mm: wire slot width in mm
    :type wire_slot_width_mm: float
    :param wire_slot_depth_mm: wire slot depth in mm
    :type wire_slot_depth_mm: float
    :param wire_slot_height_mm: wire slot height in mm
    :type wire_slot_height_mm: float
    :param wire_slots_position: wire slot position
    :type wire_slots_position: str
    :param output_step_file: output step file name
    :type output_step_file: str
    :param save_freecad_file: True to save freecad file
    :type save_freecad_file: bool
    :return: Filepath of output step file
    :rtype: str
    """
    if not output_step_file:
        raise ValueError("output_step_file must not be empty.")

    output_step_file = os.path.abspath(output_step_file)
    output_directory = os.path.dirname(output_step_file)

    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    document = App.newDocument("pq_bobbin")

    try:
        final_shape = create_round_bobbin(
            window_h_mm=window_h_mm,
            window_w_mm=window_w_mm,
            core_inner_diameter_mm=core_inner_diameter_mm,
            flange_thickness_inner_mm=flange_thickness_inner_mm,
            flange_thickness_top_mm=flange_thickness_top_mm,
            flange_thickness_bot_mm=flange_thickness_bot_mm,
            clearance=clearance,
            inner_edge_radius_mm=inner_edge_radius_mm,
            outer_edge_radius_mm=outer_edge_radius_mm,
            enable_wire_slots=enable_wire_slots,
            wire_slot_width_mm=wire_slot_width_mm,
            wire_slot_depth_mm=wire_slot_depth_mm,
            wire_slot_height_mm=wire_slot_height_mm,
            wire_slots_position=wire_slots_position
        )

        bobbin_object = document.addObject(
            "Part::Feature",
            "PQ_Round_Bobbin"
        )
        bobbin_object.Label = "PQ Round Bobbin"
        bobbin_object.Shape = final_shape

        document.recompute()

        Import.export([bobbin_object], output_step_file)
        # Export for 3D-visualization
        filename_no_extension, _ = os.path.splitext(output_step_file)
        output_glb_file = filename_no_extension + ".glb"
        # Add arrays to the shape
        bobbin_object.Shape.tessellate(0.1)
        # Export glb-file
        Import.export([bobbin_object], output_glb_file)

        if not os.path.isfile(output_step_file):
            raise RuntimeError(
                f"FreeCAD did not create the STEP file: {output_step_file}"
            )

        if os.path.getsize(output_step_file) == 0:
            raise RuntimeError(
                f"FreeCAD created an empty STEP file: {output_step_file}"
            )

        if save_freecad_file:
            output_freecad_file = (
                os.path.splitext(output_step_file)[0] + ".FCStd"
            )
            document.saveAs(output_freecad_file)
            logger.info(
                "FreeCAD document created: %s",
                output_freecad_file
            )

        logger.info("STEP file created: %s", output_step_file)

        return output_step_file

    finally:
        App.closeDocument(document.Name)


# ---------------------------------------------------------------------------
# Read parameters
# ---------------------------------------------------------------------------

window_h_mm = read_float_environment_variable(
    "WINDOW_H_MM",
    17.2
)

window_w_mm = read_float_environment_variable(
    "WINDOW_W_MM",
    7.0
)

core_inner_diameter_mm = read_float_environment_variable(
    "CORE_INNER_DIAMETER_MM",
    13.45
)

# Wall thickness of the central cylindrical tube.
flange_thickness_inner_mm = read_float_environment_variable(
    "FLANGE_THICKNESS_INNER_MM",
    2.0
)

# Axial material thickness of the upper flange.
flange_thickness_top_mm = read_float_environment_variable(
    "FLANGE_THICKNESS_TOP_MM",
    2.0
)

# Axial material thickness of the lower flange.
flange_thickness_bot_mm = read_float_environment_variable(
    "FLANGE_THICKNESS_BOT_MM",
    2.0
)

clearance = read_float_environment_variable(
    "CLEARANCE",
    0.3
)

inner_edge_radius_mm = read_float_environment_variable(
    "INNER_EDGE_RADIUS",
    0.6
)

outer_edge_radius_mm = read_float_environment_variable(
    "OUTER_EDGE_RADIUS",
    0.6
)

enable_wire_slots = read_bool_environment_variable(
    "ENABLE_WIRE_SLOTS",
    True
)

wire_slot_width_mm = read_float_environment_variable(
    "WIRE_SLOT_WIDTH",
    4.0
)

wire_slot_depth_mm = read_float_environment_variable(
    "WIRE_SLOT_DEPTH",
    window_w_mm - flange_thickness_inner_mm
)

wire_slot_height_mm = read_float_environment_variable(
    "WIRE_SLOT_HEIGHT",
    max(
        flange_thickness_top_mm,
        flange_thickness_bot_mm
    ) + 0.5
)

wire_slots_position = read_string_environment_variable(
    "WIRE_SLOTS_POSITION",
    "both"
)

output_step_file = os.environ.get(
    "OUTPUT_STEP_FILE",
    "./pq_bobbin.step"
)

save_freecad_file = read_bool_environment_variable(
    "SAVE_FCSTD_FILE",
    False
)


# ---------------------------------------------------------------------------
# Generate and export
# ---------------------------------------------------------------------------

try:
    logger.info("Generating PQ round bobbin with parameters:")
    logger.info("  window_h_mm: %s", window_h_mm)
    logger.info("  window_w_mm: %s", window_w_mm)
    logger.info(
        "  core_inner_diameter_mm: %s",
        core_inner_diameter_mm
    )
    logger.info(
        "  flange_thickness_inner_mm: %s",
        flange_thickness_inner_mm
    )
    logger.info(
        "  flange_thickness_top_mm: %s",
        flange_thickness_top_mm
    )
    logger.info(
        "  flange_thickness_bot_mm: %s",
        flange_thickness_bot_mm
    )
    logger.info("  clearance: %s", clearance)
    logger.info("  inner_edge_radius_mm: %s", inner_edge_radius_mm)
    logger.info("  outer_edge_radius_mm: %s", outer_edge_radius_mm)
    logger.info("  enable_wire_slots: %s", enable_wire_slots)
    logger.info("  wire_slots_position: %s", wire_slots_position)
    logger.info("  wire_slot_width_mm: %s", wire_slot_width_mm)
    logger.info("  wire_slot_depth_mm: %s", wire_slot_depth_mm)
    logger.info("  wire_slot_height_mm: %s", wire_slot_height_mm)
    logger.info("  output_step_file: %s", output_step_file)

    export_round_bobbin_step(
        window_h_mm=window_h_mm,
        window_w_mm=window_w_mm,
        core_inner_diameter_mm=core_inner_diameter_mm,
        flange_thickness_inner_mm=flange_thickness_inner_mm,
        flange_thickness_top_mm=flange_thickness_top_mm,
        flange_thickness_bot_mm=flange_thickness_bot_mm,
        clearance=clearance,
        inner_edge_radius_mm=inner_edge_radius_mm,
        outer_edge_radius_mm=outer_edge_radius_mm,
        enable_wire_slots=enable_wire_slots,
        wire_slot_width_mm=wire_slot_width_mm,
        wire_slot_depth_mm=wire_slot_depth_mm,
        wire_slot_height_mm=wire_slot_height_mm,
        wire_slots_position=wire_slots_position,
        output_step_file=output_step_file,
        save_freecad_file=save_freecad_file
    )

except Exception as error:
    logger.exception("Failed to create PQ round bobbin: %s", error)
    print(
        f"ERROR while creating PQ bobbin: {error}",
        file=sys.stderr
    )
    sys.exit(1)
