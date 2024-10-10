import io
import json

from fastapi import APIRouter, File, UploadFile

database_router = r = APIRouter()


@r.put("/update_database")
async def update_database(file: UploadFile = File(...)):
    """Updates the database file of the fingerprints"""

    # Open the input file and read its contents
    contents = await file.read()

    # Parse the contents as JSON
    data = json.loads(contents)

    # Open the existing JSON file and write the new contents
    with open("mock_db.json", "w") as outfile:
        json.dump(data, outfile)

    return {"message": "Mock database updated"}


@r.put("/update_index")
async def update_index(file: UploadFile = File(...)):
    """Updates an index file with the contents of an input file"""

    # Open the input file and read its contents
    contents = await file.read()

    # Open the existing index file and write the new contents
    with open("index.txt", "w") as outfile:
        outfile.write(contents.decode())

    return {"message": "Index file updated"}


@r.put("/update_model")
async def update_model(file: UploadFile = File(...)):
    """Updates an model"""

    # Open the input file and read its contents
    contents = await file.read()

    # Open the existing index file and write the new contents
    with open("index.txt", "w") as outfile:
        outfile.write(contents.decode())

    return {"message": "Index file updated"}
