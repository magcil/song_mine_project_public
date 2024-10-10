# Start from a Python image
FROM python:3.9

WORKDIR /usr/scr/app/backend

# Install required packages
COPY ./requirements.txt /usr/scr/app/backend/requirements.txt

RUN apt-get update && apt-get install -y ffmpeg

RUN pip install --no-cache-dir --upgrade -r ./requirements.txt && \
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy everything
COPY . /usr/scr/app/backend

CMD ["uvicorn","main:app", "--host","0.0.0.0",  "--port", "8080"]