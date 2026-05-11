from abc import ABC


class BaseGateway(ABC):
    """
    Base class for all gateways. This class should be inherited by all gateways and should implement the send and receive methods.
    """

    def __init__(self):
        """Initialize the gateway."""

    def send(self, message):
        """Send a message to the gateway."""

    def receive(self):
        """Receive a message from the gateway."""
