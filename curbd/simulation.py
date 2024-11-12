import numpy as np
from pydantic import BaseModel


class Region(BaseModel):
    #: The number of neurons in the region
    num_neurons: int
    #: The chaos parameter of the region
    chaos: float


def simulate_regions(*regions) -> dict:
    """
    Simulate the activity of multiple, interacting neural populations.

    :param regions: A list of dictionaries, each containing the parameters of a region.

    :return: A dictionary containing the activity of each region.
    """


    activity = {}
    for region in regions:
        activity[region['name']] = np.random.rand(region['number_units'], region['number_timepoints'])
    return activity