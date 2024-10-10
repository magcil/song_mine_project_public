import os
from os.path import dirname, join
import os
from dotenv import dotenv_values
from motor.motor_asyncio import AsyncIOMotorClient

dotenv_path = join(os.getcwd(), ".env")
config = dotenv_values(dotenv_path)


mongodb_client = AsyncIOMotorClient(config["MONGODB_URI"])
db = mongodb_client[config["MONGO_DATABASE"]]
