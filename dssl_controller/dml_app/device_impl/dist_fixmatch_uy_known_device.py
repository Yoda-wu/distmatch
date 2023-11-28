from config import Configuration
from device_impl.base_device import Base_Device


class Distributed_FixMatch_Uy_Known_Device(Base_Device):
    def __init__(self, config: Configuration) -> None:
        super().__init__(config)