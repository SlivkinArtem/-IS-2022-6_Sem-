from db import engine
from models import Base, User, LikedText

Base.metadata.create_all(bind=engine)