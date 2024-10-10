# Museek DEMO

**Congrats** you made it to the museek demo.

This directory contains a demo version of the Museek project, showcasing its frontend and backend components. The frontend is built using ReactJS with typescript and more specifically vite, while the backend utilizes FastAPI.

<details><summary><strong>Install Docker and Docker-Compose on Ubuntu</strong></summary>

<p>
To install Docker and Docker Compose on Ubuntu, follow these steps:

1. Update your existing list of packages:
   ```bash
   sudo apt update
   ```
2. Install Docker:
   ```bash
   sudo apt install docker.io
   ```
3. Start and Enable Docker
   ```bash
   sudo systemctl start docker
   sudo systemctl enable docker
   ```
4. Install Docker Compose
   ```bash
   sudo apt install docker-compose
   ```
   That's it! Docker and Docker Compose should now be installed on your Ubuntu system. You can proceed to use Docker and Docker Compose to manage your containers and applications.
   > **_NOTE:_** You have to run docker using sudo.
   </p>
   </details>
   
 <details><summary><strong>Install Docker and Docker-Compose on MacOS</strong></summary>

<p>
To install Docker and Docker Compose on MacOS, follow these steps:

1. Install Docker:
   ```bash
   brew install --cask docker
   ```
> Then launch the Docker app. It will ask for privileged access, confirm.  

2. Install Docker Compose
   ```bash
   brew install docker-compose
   ```
   That's it! Docker and Docker Compose should now be installed on your MacOS. You can proceed to use Docker and Docker Compose to manage your containers and applications.
   </p>
   </details>

---
## Running the demo

### Local Development
To run your application locally, follow these steps:

#### Frontend Setup

1. Navigate to the frontend project directory.
```bash
cd frontend
```
2. Install the necessary packages:
```bash
npm install --legacy-peer-deps
```
3. Start the development server:
```bash
npm start
```
The frontend React.js app will now be running at http://localhost:3000.

#### Backend Setup
1. Navigate to the backend project directory.
```bash
cd backend
```
2. Install the necessary packages:
```bash
pip install -r requirements.txt
```
3. Start the development server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8080
```
The backend FastAPI server will now be running at http://localhost:8080.


### Using Docker Compose
To run your application using Docker Compose, follow these steps:

1. Open a terminal or command prompt and navigate to the project directory that contains the `docker-compose.yml` file.

2. Build and start the containers using Docker Compose:

   ```bash
   docker-compose up --build --detach/-d
   ```

   `-d`,`--detach`: Detached mode: Run containers in the background

   This command will build the Docker images for the frontend and backend, create the containers, and start them. The `--build` flag ensures that the images are rebuilt if there are any changes.

3. Wait for the containers to start up. You should see the logs from each container in the terminal.

4. Once the containers are up and running, you can access your application locally:
   - Nginx will proxy requests to the frontend at `http://localhost` and requests to the backend API at `http://localhost/api`. You can access the  backend endpoints at `http://localhost/api/docs`.

That's it! You can now access your React.js frontend and FastAPI backend running inside Docker containers, with Nginx as a reverse proxy.


**_NOTE:_**  Please ensure that the trained model, the songs index, and the JSON file from the FAISS are placed in the correct directories within the backend container, pre_trained_models and data respectively.  These files are essential for the full functionality of the app but are not included in the repository due to their size and sensitivity.