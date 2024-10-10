# Run the FastAPI app local

## Create a virtual enviroment and install all the necessary dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Start the FastAPI Server

```bash
cd backend
uvicorn main:app --reload
```

# Run the FastAPI app with docker

To execute the backend part of the app inside a container first navigate to `~/projects/song_mine_project/RnD/deep_fingerprinting/backend`. Build the image through the Dockerfile by running:

```bash
docker build -t museek_image .
```

Then start the container by running:

```bash
docker run -d --name museek_app -p 8000:15400 museek_image
```

Open a browser and interact with the API at `http://localhost:8000/docs`.
