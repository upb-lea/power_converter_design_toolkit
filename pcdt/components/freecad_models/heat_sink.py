# -*- coding: utf-8 -*-
"""
Generate a parametric extruded heat sink and export it as STEP using FreeCAD.

All dimensions are in mm.

The heat sink consists of:
- A top base plate
- Two outer fins with half the regular fin thickness
- Internal fins with full regular fin thickness
- Equally sized cooling channels

The extrusion direction is the Y direction (heat-sink length).

Environment variables
---------------------
HEIGHT_C_MM
    Cooling-fin height.

HEIGHT_D_MM
    Top base-plate thickness.

LENGTH_L_MM
    Heat-sink length / extrusion direction.

NUMBER_COOLING_CHANNELS_N
    Number of cooling channels.

THICKNESS_FIN_T_MM
    Thickness of a regular internal fin.
    The outer fins are half this thickness.

WIDTH_B_MM
    Total heat-sink width.

OUTPUT_STEP_FILE
    Destination STEP-file path.

SAVE_FCSTD_FILE
    Optional. Set to 1, true, yes, or on to also save an editable FCStd file.

Example command-line call:
HEIGHT_C_MM=25 \
HEIGHT_D_MM=3 \
LENGTH_L_MM=100 \
NUMBER_COOLING_CHANNELS_N=8 \
THICKNESS_FIN_T_MM=2 \
WIDTH_B_MM=80 \
OUTPUT_STEP_FILE="./heat_sink.step" \
FreeCADCmd heat_sink.py
"""

# Python libraries
import os
import logging

# FreeCAD libraries
import FreeCAD as App
import Part
import Import


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def read_float_environment_variable(
    variable_name: str,
    default_value: float
) -> float:
    """
    Read a floating-point value from an environment variable.

    The default value is returned when the environment variable is not set
    or is an empty string.

    :param variable_name: variable name
    :type variable_name: str
    :param default_value: default value
    :type default_value: str
    """
    value = os.environ.get(variable_name)

    if value is None or value == "":
        return default_value

    try:
        return float(value)
    except ValueError as error:
        raise ValueError(
            f"Environment variable {variable_name} must be numeric, "
            f"but received {value!r}."
        ) from error


def read_int_environment_variable(
    variable_name: str,
    default_value: int
) -> int:
    """
    Read an integer value from an environment variable.

    The default value is returned when the environment variable is not set
    or is an empty string.

    :param variable_name: variable name
    :type variable_name: str
    :param default_value: default value
    :type default_value: int
    """
    value = os.environ.get(variable_name)

    if value is None or value == "":
        return default_value

    try:
        parsed_value = int(value)
    except ValueError as error:
        raise ValueError(
            f"Environment variable {variable_name} must be an integer, "
            f"but received {value!r}."
        ) from error

    return parsed_value


# ---------------------------------------------------------------------------
# Geometry generation
# ---------------------------------------------------------------------------

def create_heat_sink(
    height_c_mm: float,
    height_d_mm: float,
    length_l_mm: float,
    number_cooling_channels_n: int,
    thickness_fin_t_mm: float,
    width_b_mm: float
) -> Part.Shape:
    """
    Create a parametric extruded heat-sink shape.

    Coordinate system:
    - X: heat-sink width
    - Y: heat-sink length / extrusion direction
    - Z: vertical direction

    The bottom of the fins is at Z = 0.
    The base plate begins at Z = height_c_mm.

    :param height_c_mm: Cooling-fin height in mm.
    :type height_c_mm: float
    :param height_d_mm: Top base-plate thickness in mm.
    :type height_d_mm: float
    :param length_l_mm: Heat-sink length in mm.
    :type length_l_mm: float
    :param number_cooling_channels_n: Number of cooling channels.
    :type number_cooling_channels_n: int
    :param thickness_fin_t_mm: Thickness of a regular internal fin in mm.
    :type thickness_fin_t_mm: float
    :param width_b_mm: Total heat-sink width in mm.
    :type width_b_mm: float
    :return: Final fused heat-sink shape.
    """
    # -----------------------------------------------------------------------
    # Input validation
    # -----------------------------------------------------------------------

    if height_c_mm <= 0:
        raise ValueError("height_c_mm must be greater than 0.")

    if height_d_mm <= 0:
        raise ValueError("height_d_mm must be greater than 0.")

    if length_l_mm <= 0:
        raise ValueError("length_l_mm must be greater than 0.")

    if number_cooling_channels_n < 1:
        raise ValueError(
            "number_cooling_channels_n must be at least 1."
        )

    if thickness_fin_t_mm <= 0:
        raise ValueError(
            "thickness_fin_t_mm must be greater than 0."
        )

    if width_b_mm <= 0:
        raise ValueError("width_b_mm must be greater than 0.")

    # -----------------------------------------------------------------------
    # Derived dimensions
    # -----------------------------------------------------------------------

    # For n cooling channels:
    # - two outer fins each use half the nominal fin thickness;
    # - n - 1 internal fins use the full nominal fin thickness.
    #
    # The total fin-material width is therefore n * thickness_fin_t_mm.
    total_fin_material_width_mm = (
        number_cooling_channels_n * thickness_fin_t_mm
    )

    channel_width_mm = (
        width_b_mm - total_fin_material_width_mm
    ) / number_cooling_channels_n

    if channel_width_mm <= 0:
        raise ValueError(
            "width_b_mm is too small: cooling channels need positive width. "
            f"Calculated channel width: {channel_width_mm} mm."
        )

    outer_fin_thickness_mm = thickness_fin_t_mm / 2.0

    # Small overlap ensures reliable Boolean fusing.
    overlap_mm = 0.01

    # -----------------------------------------------------------------------
    # Top base plate
    # -----------------------------------------------------------------------

    base_plate_shape = Part.makeBox(
        width_b_mm,
        length_l_mm,
        height_d_mm,
        App.Vector(0, 0, height_c_mm)
    )

    # -----------------------------------------------------------------------
    # Left outer fin
    # -----------------------------------------------------------------------

    left_outer_fin_shape = Part.makeBox(
        outer_fin_thickness_mm,
        length_l_mm,
        height_c_mm + overlap_mm,
        App.Vector(0, 0, 0)
    )

    # -----------------------------------------------------------------------
    # Internal fins
    # -----------------------------------------------------------------------

    internal_fin_shapes = []

    # There are n - 1 full-thickness internal fins for n cooling channels.
    for fin_index in range(1, number_cooling_channels_n):
        x_fin_mm = (
            outer_fin_thickness_mm + channel_width_mm + (fin_index - 1) * (thickness_fin_t_mm + channel_width_mm)
        )

        internal_fin_shape = Part.makeBox(
            thickness_fin_t_mm,
            length_l_mm,
            height_c_mm + overlap_mm,
            App.Vector(x_fin_mm, 0, 0)
        )

        internal_fin_shapes.append(internal_fin_shape)

    # -----------------------------------------------------------------------
    # Right outer fin
    # -----------------------------------------------------------------------

    right_outer_fin_shape = Part.makeBox(
        outer_fin_thickness_mm,
        length_l_mm,
        height_c_mm + overlap_mm,
        App.Vector(
            width_b_mm - outer_fin_thickness_mm,
            0,
            0
        )
    )

    # -----------------------------------------------------------------------
    # Combine all solid regions
    # -----------------------------------------------------------------------

    final_shape = base_plate_shape.fuse(left_outer_fin_shape)

    for internal_fin_shape in internal_fin_shapes:
        final_shape = final_shape.fuse(internal_fin_shape)

    final_shape = final_shape.fuse(right_outer_fin_shape)

    return final_shape.removeSplitter()


# ---------------------------------------------------------------------------
# STEP export
# ---------------------------------------------------------------------------

def export_heat_sink_step(
    height_c_mm: float,
    height_d_mm: float,
    length_l_mm: float,
    number_cooling_channels_n: int,
    thickness_fin_t_mm: float,
    width_b_mm: float,
    output_step_file: str,
    save_freecad_file: bool = False
) -> str:
    """
    Create a heat sink and export it as a STEP file.

    :param height_c_mm: Cooling-fin height in mm.
    :param height_d_mm: Top base-plate thickness in mm.
    :param length_l_mm: Heat-sink length in mm.
    :param number_cooling_channels_n: Number of cooling channels.
    :param thickness_fin_t_mm: Thickness of regular internal fins in mm.
    :param width_b_mm: Total heat-sink width in mm.
    :param output_step_file: Destination STEP-file path.
    :param save_freecad_file: True to save an editable FCStd file as well.
    :return: Absolute path of the generated STEP file.
    """
    if not output_step_file:
        raise ValueError("output_step_file must not be empty.")

    output_step_file = os.path.abspath(output_step_file)
    output_directory = os.path.dirname(output_step_file)

    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    document = App.newDocument("Parametric_Heat_Sink")

    try:
        final_shape = create_heat_sink(
            height_c_mm=height_c_mm,
            height_d_mm=height_d_mm,
            length_l_mm=length_l_mm,
            number_cooling_channels_n=number_cooling_channels_n,
            thickness_fin_t_mm=thickness_fin_t_mm,
            width_b_mm=width_b_mm,
        )

        heat_sink_object = document.addObject(
            "Part::Feature",
            "Heat_Sink"
        )
        heat_sink_object.Label = "Parametric Heat Sink"
        heat_sink_object.Shape = final_shape

        document.recompute()

        # Export the fused single-solid heat sink.
        Import.export([heat_sink_object], output_step_file)
        # Export for 3D-visualization
        filename_no_extension, _ = os.path.splitext(output_step_file)
        output_glb_file = filename_no_extension + ".glb"
        # Add arrays to the shape
        heat_sink_object.Shape.tessellate(0.1)
        # Export glb-file
        Import.export([heat_sink_object], output_glb_file)

        if save_freecad_file:
            output_freecad_file = (
                os.path.splitext(output_step_file)[0] + ".FCStd"
            )
            document.saveAs(output_freecad_file)
            logger.info(
                f"FreeCAD document created: {output_freecad_file}"
            )

        logger.info(f"STEP file created: {output_step_file}")

        return output_step_file

    finally:
        App.closeDocument(document.Name)


# ---------------------------------------------------------------------------
# Read input values from environment variables
# ---------------------------------------------------------------------------

height_c_mm = read_float_environment_variable(
    "HEIGHT_C_MM",
    25.0
)

height_d_mm = read_float_environment_variable(
    "HEIGHT_D_MM",
    3.0
)

length_l_mm = read_float_environment_variable(
    "LENGTH_L_MM",
    100.0
)

number_cooling_channels_n = read_int_environment_variable(
    "NUMBER_COOLING_CHANNELS_N",
    8
)

thickness_fin_t_mm = read_float_environment_variable(
    "THICKNESS_FIN_T_MM",
    2.0
)

width_b_mm = read_float_environment_variable(
    "WIDTH_B_MM",
    80.0
)

output_step_file = os.environ.get(
    "OUTPUT_STEP_FILE",
    "./parametric_heat_sink.step"
)

save_fcstd_file = os.environ.get(
    "SAVE_FCSTD_FILE",
    "0"
).strip().lower() not in ("0", "false", "no", "off")


# ---------------------------------------------------------------------------
# Generate and export the heat sink
# ---------------------------------------------------------------------------

logger.info("Generating parametric heat sink with parameters:")
logger.info(f"  height_c_mm: {height_c_mm}")
logger.info(f"  height_d_mm: {height_d_mm}")
logger.info(f"  length_l_mm: {length_l_mm}")
logger.info(
    f"  number_cooling_channels_n: {number_cooling_channels_n}"
)
logger.info(f"  thickness_fin_t_mm: {thickness_fin_t_mm}")
logger.info(f"  width_b_mm: {width_b_mm}")
logger.info(f"  output_step_file: {output_step_file}")

export_heat_sink_step(
    height_c_mm=height_c_mm,
    height_d_mm=height_d_mm,
    length_l_mm=length_l_mm,
    number_cooling_channels_n=number_cooling_channels_n,
    thickness_fin_t_mm=thickness_fin_t_mm,
    width_b_mm=width_b_mm,
    output_step_file=output_step_file,
    save_freecad_file=save_fcstd_file,
)
