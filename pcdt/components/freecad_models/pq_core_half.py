# -*- coding: utf-8 -*-
"""
Generate a lower half of a parametric PQ core and export it as STEP using FreeCAD.

All dimensions are in mm.

The script reads its parameters from environment variables so it can be
called through a wrapper using subprocess and FreeCADCmd.

Environment variables
---------------------
CORE_H_MM
CORE_INNER_DIAMETER_MM
WINDOW_H_MM
WINDOW_W_MM
CORE_DIMENSION_X_MM
CORE_DIMENSION_Y_MM
L_AIR_GAP_MM
OUTPUT_STEP_FILE

Example command-line call:

CORE_INNER_DIAMETER_MM=16.0 \
L_AIR_GAP_MM=0.8 \
OUTPUT_STEP_FILE="./pq_core_half.step" \
FreeCADCmd pq_core_half.py
"""
# python libraries
import os
import logging

# 3rd party libraries
import numpy as np

# freecad libraries
import FreeCAD as App
import Part
import Import

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def read_float_environment_variable(variable_name: str) -> float:
    """
    Read a floating-point value from an environment variable.

    The default value is returned when the environment variable is not set
    or is an empty string.

    :param variable_name: variable name
    :type variable_name: str
    """
    value = os.environ.get(variable_name)

    if value is None or value == "":
        raise ValueError(f"Environment variable {variable_name} does not exist.")

    try:
        return float(value)
    except ValueError as error:
        raise ValueError(
            f"Environment variable {variable_name} must be numeric, "
            f"but received {value!r}."
        ) from error


def create_pq_core_half(core_h_mm: float, core_inner_diameter_mm: float, window_h_mm: float,
                        window_w_mm: float, core_dimension_x_mm: float, core_dimension_y_mm: float,
                        l_air_gap_mm: float) -> Part.makeCylinder:
    """
    Create the shape of a lower PQ core half.

    The model consists of:  
    - A lower yoke
    - An outer ring / outer legs
    - A round center leg

    The center leg is directly made shorter by l_air_gap_mm / 2.
    Thus, no air-gap cylinder needs to be subtracted, which avoids Boolean
    artifacts on the air-gap face.

    :param core_h_mm: core height in mm
    :type core_h_mm: float
    :param core_inner_diameter_mm: core inner diameter in mm
    :type core_inner_diameter_mm: float
    :param window_h_mm: window height in mm
    :type window_h_mm: float
    :param window_w_mm: window width in mm
    :type window_w_mm: float
    :param core_dimension_x_mm: x-core dimension in mm
    :type core_dimension_x_mm: float
    :param core_dimension_y_mm: y-core dimension in mm
    :type core_dimension_y_mm: float
    :param l_air_gap_mm: air gap in mm
    :type l_air_gap_mm: float
    """
    # Small overlap avoids coincident Boolean faces.
    overlap_mm = 0.1

    # -----------------------------------------------------------------------
    # Input validation
    # -----------------------------------------------------------------------
    if core_h_mm <= 0:
        raise ValueError("core_h_mm must be greater than 0.")

    if core_inner_diameter_mm <= 0:
        raise ValueError(
            "core_inner_diameter_mm must be greater than 0."
        )

    if window_h_mm <= 0 or window_h_mm >= core_h_mm:
        raise ValueError(
            "window_h_mm must be greater than 0 and smaller than core_h_mm."
        )

    if window_w_mm <= 0:
        raise ValueError("window_w_mm must be greater than 0.")

    if core_dimension_x_mm <= 0 or core_dimension_y_mm <= 0:
        raise ValueError(
            "core_dimension_x_mm and core_dimension_y_mm "
            "must be greater than 0."
        )

    if l_air_gap_mm < 0:
        raise ValueError("l_air_gap_mm must not be negative.")

    # -----------------------------------------------------------------------
    # Derived dimensions
    # -----------------------------------------------------------------------

    # Radius of the round center leg.
    center_leg_radius_mm = core_inner_diameter_mm / 2.0

    # Inner radius of the outer ring / outer legs.
    # The radial gap between this radius and the center leg is window_w_mm.
    outer_leg_inner_radius_mm = (center_leg_radius_mm + window_w_mm)

    # Thickness of the bottom yoke.
    yoke_thickness_mm = (core_h_mm - window_h_mm) / 2.0

    # Nominal height of one complete core half.
    half_core_h_mm = core_h_mm / 2.0

    # Each of two identical halves contributes half of the total air gap.
    half_air_gap_mm = l_air_gap_mm / 2.0

    # The center leg has its final height directly.
    center_leg_h_mm = half_core_h_mm - half_air_gap_mm

    if yoke_thickness_mm <= 0:
        raise ValueError(
            "Invalid geometry: yoke thickness must be greater than 0."
        )

    if center_leg_h_mm <= 0:
        raise ValueError(
            "l_air_gap_mm is too large: the center leg would have no height."
        )

    if center_leg_h_mm < yoke_thickness_mm:
        raise ValueError(
            "l_air_gap_mm is too large: the center leg is shorter than "
            "the lower yoke."
        )

    # This radius covers the whole final rectangular X/Y clipping area.
    outer_blank_radius_mm = np.sqrt((core_dimension_x_mm / 2.0) ** 2 + (core_dimension_y_mm / 2.0) ** 2)

    # -----------------------------------------------------------------------
    # Common X/Y clipping solid (total core dimensions)
    # -----------------------------------------------------------------------
    outer_xy_clipping_box = Part.makeBox(
        core_dimension_x_mm,
        core_dimension_y_mm,
        half_core_h_mm,
        App.Vector(
            -core_dimension_x_mm / 2.0,
            -core_dimension_y_mm / 2.0,
            0
        )
    )

    # -----------------------------------------------------------------------
    # Lower yoke
    # -----------------------------------------------------------------------
    # Solid round blank, clipped to the required outer X/Y dimensions.
    lower_yoke_cylinder = Part.makeCylinder(
        outer_blank_radius_mm,
        yoke_thickness_mm,
        App.Vector(0, 0, 0)
    )

    lower_yoke_shape = lower_yoke_cylinder.common(outer_xy_clipping_box)

    # -----------------------------------------------------------------------
    # Outer ring / outer legs
    # -----------------------------------------------------------------------
    # This annular region begins at the top of the lower yoke and extends
    # to the nominal mating plane at Z = half_core_h_mm.
    outer_leg_h_mm = half_core_h_mm - yoke_thickness_mm + overlap_mm

    outer_leg_outer_cylinder = Part.makeCylinder(
        outer_blank_radius_mm,
        outer_leg_h_mm,
        App.Vector(0, 0, yoke_thickness_mm - overlap_mm)
    )

    outer_leg_inner_cylinder = Part.makeCylinder(
        outer_leg_inner_radius_mm,
        outer_leg_h_mm + 2.0 * overlap_mm,
        App.Vector(0, 0, yoke_thickness_mm - overlap_mm)
    )

    outer_ring_shape = outer_leg_outer_cylinder.cut(
        outer_leg_inner_cylinder).common(outer_xy_clipping_box)

    # -----------------------------------------------------------------------
    # Center leg
    # -----------------------------------------------------------------------
    # The final center-leg height already includes the air-gap reduction.
    center_leg_shape = Part.makeCylinder(
        center_leg_radius_mm,
        center_leg_h_mm,
        App.Vector(0, 0, 0)
    )

    # -----------------------------------------------------------------------
    # Combine all solid regions
    # -----------------------------------------------------------------------
    final_shape = lower_yoke_shape.fuse(outer_ring_shape).fuse(center_leg_shape)

    return final_shape.removeSplitter()


def export_pq_core_half_step(
    core_h_mm: float,
    core_inner_diameter_mm: float,
    window_h_mm: float,
    window_w_mm: float,
    core_dimension_x_mm: float,
    core_dimension_y_mm: float,
    l_air_gap_mm: float,
    output_step_file: str,
    save_freecad_file: bool = False
) -> str:
    """
    Create a single PQ core half and export it to a STEP file.

    :param core_h_mm: core height in mm
    :type core_h_mm: float
    :param core_inner_diameter_mm: core inner diameter in mm
    :type core_inner_diameter_mm: float
    :param window_h_mm: window height in mm
    :type window_h_mm: float
    :param window_w_mm: window width in mm
    :type window_w_mm: float
    :param core_dimension_x_mm: x-core dimension in mm
    :type core_dimension_x_mm: float
    :param core_dimension_y_mm: y-core dimension in mm
    :type core_dimension_y_mm: float
    :param l_air_gap_mm: air gap in mm
    :type l_air_gap_mm: float
    :param output_step_file: directory of output step file
    :type output_step_file: str
    :param save_freecad_file: True to save the freecad file
    :type save_freecad_file: bool
    """
    if not output_step_file:
        raise ValueError("output_step_file must not be empty.")

    output_step_file = os.path.abspath(output_step_file)
    output_directory = os.path.dirname(output_step_file)

    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    # Use a unique document name when the function is called repeatedly.
    document = App.newDocument("PQ_Core_Half")

    try:
        final_shape = create_pq_core_half(
            core_h_mm=core_h_mm,
            core_inner_diameter_mm=core_inner_diameter_mm,
            window_h_mm=window_h_mm,
            window_w_mm=window_w_mm,
            core_dimension_x_mm=core_dimension_x_mm,
            core_dimension_y_mm=core_dimension_y_mm,
            l_air_gap_mm=l_air_gap_mm,
        )

        core_object = document.addObject(
            "Part::Feature",
            "PQ_Core_Half"
        )

        core_object.Label = "PQ Core Half"
        core_object.Shape = final_shape

        document.recompute()

        # Export the single solid as a STEP file.
        Import.export([core_object], output_step_file)
        # Export for 3D-visualization
        filename_no_extension, _ = os.path.splitext(output_step_file)
        output_glb_file = filename_no_extension + ".glb"
        # Add arrays to the shape
        core_object.Shape.tessellate(0.1)
        # Export glb-file
        Import.export([core_object], output_glb_file)

        # Optionally save an editable FreeCAD document.
        if save_freecad_file:
            output_freecad_file = (os.path.splitext(output_step_file)[0] + ".FCStd")
            document.saveAs(output_freecad_file)
            logger.info(f"FreeCAD document created: {output_freecad_file}")

        logger.info(f"STEP file created: {output_step_file}")
        return output_step_file

    finally:
        # Important when this is called repeatedly in a long-running process.
        App.closeDocument(document.Name)


# ---------------------------------------------------------------------------
# Read input values from environment variables
# ---------------------------------------------------------------------------

core_h_mm = read_float_environment_variable("CORE_H_MM")

core_inner_diameter_mm = read_float_environment_variable("CORE_INNER_DIAMETER_MM")

window_h_mm = read_float_environment_variable("WINDOW_H_MM")

window_w_mm = read_float_environment_variable("WINDOW_W_MM")

core_dimension_x_mm = read_float_environment_variable("CORE_DIMENSION_X_MM")

core_dimension_y_mm = read_float_environment_variable("CORE_DIMENSION_Y_MM")

l_air_gap_mm = read_float_environment_variable("L_AIR_GAP_MM")

output_step_file = os.environ.get("OUTPUT_STEP_FILE", "./pq_core_half.step")

save_fcstd_file = os.environ.get("SAVE_FCSTD_FILE", "0").strip().lower() not in ("0", "false", "no", "off")


# ---------------------------------------------------------------------------
# Generate and export the core
# ---------------------------------------------------------------------------

logger.info("Generating PQ core lower half with parameters:")
logger.info(f"  core_h_mm: {core_h_mm}")
logger.info(f"  core_inner_diameter_mm: {core_inner_diameter_mm}")
logger.info(f"  window_h_mm: {window_h_mm}")
logger.info(f"  window_w_mm: {window_w_mm}")
logger.info(f"  core_dimension_x_mm: {core_dimension_x_mm}")
logger.info(f"  core_dimension_y_mm: {core_dimension_y_mm}")
logger.info(f"  l_air_gap_mm: {l_air_gap_mm}")
logger.info(f"  output_step_file: {output_step_file}")

export_pq_core_half_step(
    core_h_mm=core_h_mm,
    core_inner_diameter_mm=core_inner_diameter_mm,
    window_h_mm=window_h_mm,
    window_w_mm=window_w_mm,
    core_dimension_x_mm=core_dimension_x_mm,
    core_dimension_y_mm=core_dimension_y_mm,
    l_air_gap_mm=l_air_gap_mm,
    output_step_file=output_step_file,
    save_freecad_file=save_fcstd_file,
)
