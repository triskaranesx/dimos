from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.Joy import Joy

# PointCloud2 depends on optional heavy dependencies (e.g. Open3D). Keep it optional so
# lightweight installs (like camera bring-up) don't require the full mapping stack.
try:
    from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
except Exception:  # pragma: no cover
    PointCloud2 = None  # type: ignore[assignment]

__all__ = ["CameraInfo", "Image", "ImageFormat", "Joy"]
if PointCloud2 is not None:
    __all__.append("PointCloud2")
from dimos.msgs.sensor_msgs.JointCommand import JointCommand
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.sensor_msgs.RobotState import RobotState
