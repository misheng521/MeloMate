from typing import Awaitable, Callable


WebSocketSend = Callable[[str], Awaitable[None]]
