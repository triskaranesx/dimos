# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Encoder mixins for PubSub implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
import pickle
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

TopicT = TypeVar("TopicT")
MsgT = TypeVar("MsgT")
EncodingT = TypeVar("EncodingT")


class PubSubEncoderMixin(Generic[TopicT, MsgT, EncodingT], ABC):
    """Mixin that encodes messages before publishing and decodes them after receiving.

    This will override publish and subscribe methods to add encoding/decoding.

    Usage: Just specify encoder and decoder as a subclass:

    class MyPubSubWithJSON(PubSubEncoderMixin, MyPubSub):
        def encoder(msg, topic):
            json.dumps(msg).encode('utf-8')
        def decoder(msg, topic):
            data: json.loads(data.decode('utf-8'))
    """

    @abstractmethod
    def encode(self, msg: MsgT, topic: TopicT) -> EncodingT: ...

    @abstractmethod
    def decode(self, msg: EncodingT, topic: TopicT) -> MsgT: ...

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self._encode_callback_map: dict = {}  # type: ignore[type-arg]

    def publish(self, topic: TopicT, message: MsgT) -> None:
        """Encode the message and publish it."""
        if getattr(self, "_stop_event", None) is not None and self._stop_event.is_set():  # type: ignore[attr-defined]
            return
        encoded_message = self.encode(message, topic)
        if encoded_message is None:
            return
        super().publish(topic, encoded_message)  # type: ignore[misc]

    def subscribe(
        self, topic: TopicT, callback: Callable[[MsgT, TopicT], None]
    ) -> Callable[[], None]:
        """Subscribe with automatic decoding."""

        def wrapper_cb(encoded_data: EncodingT, topic: TopicT) -> None:
            decoded_message = self.decode(encoded_data, topic)
            callback(decoded_message, topic)

        return super().subscribe(topic, wrapper_cb)  # type: ignore[misc, no-any-return]


class PickleEncoderMixin(PubSubEncoderMixin[TopicT, MsgT, bytes]):
    """Encoder mixin that uses pickle for serialization."""

    def encode(self, msg: MsgT, *_: TopicT) -> bytes:  # type: ignore[return]
        try:
            return pickle.dumps(msg)
        except Exception as e:
            print("Pickle encoding error:", e)
            import traceback

            traceback.print_exc()
            print("Tried to pickle:", msg)

    def decode(self, msg: bytes, _: TopicT) -> MsgT:
        return pickle.loads(msg)  # type: ignore[no-any-return]


from typing import Any, Protocol


class TypedTopic(Protocol):
    """Protocol for topics that carry type information for decoding."""

    @property
    def lcm_type(self) -> type | None: ...


class LCMEncoderMixin(PubSubEncoderMixin[TopicT, Any, bytes]):
    """Encoder mixin for DimosMsg using LCM binary encoding."""

    def encode(self, msg: Any, _: TopicT) -> bytes:
        return msg.lcm_encode()  # type: ignore[no-any-return]

    def decode(self, msg: bytes, topic: TopicT) -> Any:
        lcm_type = getattr(topic, "lcm_type", None)
        if lcm_type is None:
            raise ValueError("Cannot decode: topic has no lcm_type")
        return lcm_type.lcm_decode(msg)


class JpegEncoderMixin(PubSubEncoderMixin[TopicT, Any, bytes]):
    """Encoder mixin for DimosMsg using JPEG encoding (for images)."""

    def encode(self, msg: Any, _: TopicT) -> bytes:
        return msg.lcm_jpeg_encode()  # type: ignore[no-any-return]

    def decode(self, msg: bytes, topic: TopicT) -> Any:
        lcm_type = getattr(topic, "lcm_type", None)
        if lcm_type is None:
            raise ValueError("Cannot decode: topic has no lcm_type")
        return lcm_type.lcm_jpeg_decode(msg)
