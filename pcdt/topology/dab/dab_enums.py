"""Enumerations for the DAB topology."""
# python libraries
import enum

class DabEnums(enum.Enum):
    """DAB enumerations for the filter.

    Naming convention: x_y_z_sort_filter

    Scheme:
     - x_y_z objectives in x, y, z or
     - x_y objectives in x, y
     - sort_filter sort against the filter value
    """

    eta_icost_sort_frequency = "eta_icost_sort_frequency"
    eta_icost_zvs_sort_frequency = "eta_icost_zvs_sort_frequency"
    zvs_icost_sort_frequency = "zvs_icost_sort_frequency"
    f_eta_zvs_sort_frequency = "f_eta_zvs_sort_frequency"
    zvs_icost_sort_irms = "zvs_icost_sort_irms"
