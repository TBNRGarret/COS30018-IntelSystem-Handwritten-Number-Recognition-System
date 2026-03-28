# How to Run the HNRS Application

Follow these steps carefully to get the website running on your machine.

---

## Step 1 — Make sure Python is installed

Open a terminal and run:

```bash
python --version
```

You should see something like `Python 3.x.x`. If you get an error or nothing appears, download and install Python from https://python.org. During installation, make sure to tick **"Add Python to PATH"**.

---

## Step 2 — Install the required packages

In your terminal, run this command:

```bash
pip install fastapi uvicorn opencv-python numpy python-multipart
```

Wait for it to finish. You only need to do this once.

---

## Step 3 — Navigate to the project folder

In your terminal, navigate to the folder where you cloned or downloaded this project. For example:

```bash
cd "C:\Users\YourName\Documents\GUI"
```

Replace the path with wherever you saved the project on your machine. You should be in the root folder — the one that contains both the `backend` and `frontend` folders.

---

## Step 4 — Start the server

Run this command:

```bash
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

If that does not work, find your full Python path and use it instead. For example:

```bash
C:\Users\YourName\AppData\Local\Python\pythoncore-3.14-64\python.exe -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

You will know it is working when you see this in the terminal:

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

---

## Step 5 — Open the website

Open your browser and go to:

```
http://localhost:8000
```

The HNRS application should now load with full styling and functionality.

---

## Important notes

- **Do not open `index.html` directly** by double-clicking it or through VS Code's Live Server. The application will not work correctly that way — the backend must be running first.
- **Keep the terminal open** the entire time you are using the app. Closing it will shut down the server and the website will stop working.
- **To stop the server**, go back to the terminal and press `Ctrl + C`.
- **If port 8000 is already in use**, change `--port 8000` to another number like `--port 8001`, then open `http://localhost:8001` in the browser instead.
- **When any necessary change is made to any of the components**, to the code or files, always press **Ctrl + Shift + R** in your browser to force a full refresh and clear the cache — otherwise the browser may still show the old version of the website.
