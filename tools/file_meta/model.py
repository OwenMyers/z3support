from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.types import Integer, Text, String
import uuid

Base = declarative_base()

class ImageMeta(Base):
    def __init__(self, x, y, z, bucket, key, img_max, img_min, img_std):
        self.dimension_x = x
        self.dimension_y = y
        self.dimension_z = z
        self.bucket = bucket
        self.key = key
        self.image_max = img_max
        self.image_min = img_min
        self.image_std = img_std


    __tablename__ = "image_meta"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        unique=True,
        nullable=False
    )

    dimension_x = Column(Integer)
    dimension_y = Column(Integer)
    dimension_z = Column(Integer)
    bucket = Column(String)
    key = Column(String)
    image_max = Column(Integer)
    image_min = Column(Integer)
    image_std = Column(Integer)

    def __repr__(self):
        return f"Metta item uuid: {self.id}"