import numpy as np
from pydantic import BaseModel
from enum import IntEnum


class NeuronalType(IntEnum):
    MIXED = 1
    EXCITATORY = 2
    INHIBITORY = 3


class Region(BaseModel):
    #: The number of neurons in the region
    num_neurons: int = 100
    #: The chaos parameter of the region
    chaos: float = 1.5
    #: Whether the neurons are constrained to be only excitatory or inhibitory
    neuronal_type: NeuronalType = NeuronalType.MIXED


class Brain(BaseModel):
    #: The regions to simulate
    regions: list[Region]
    #: The inter-region connection weight
    connection_weight: float = 0.01
    #: The fraction of inter-region connections
    connection_fraction: float = 0.01
    #: The external input weight
    external_input_weight: float = 1.0
    #: The width of the sequential and/or fixed-point bumps
    bump_width: float = 200.0
    #: The time constant
    tau: float = 0.1
    #: The simulation time step
    time_step: float = 0.01


class Noise(BaseModel):
    #: The weight of white noise input
    noise_weight: float = 0.01
    #: The time constant
    noise_tau: float = 0.1


class RNN(BaseModel):
    #: The time constant
    tau: float = 0.1
    #: The learning rate
    learning_rate: float = 1.0
    #: The number of training iterations
    num_iterations: int = 500
    #: Date time step (s)
    data_time_step: float = 0.01
    #: Model integration time step (s)
    model_time_step: float = 0.001





def simulate_regions(brain: Brain) -> dict:
    """
    Simulate the activity of multiple, interacting neural populations.

    :param regions: A list of dictionaries, each containing the parameters of a region.

    :return: A dictionary containing the activity of each region.
    """


    activity = {}
    for region in regions:
        activity[region['name']] = np.random.rand(region['number_units'], region['number_timepoints'])
    return activity