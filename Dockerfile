# Step 1: Start from an official Python 3.10 base image
FROM python:3.10-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy the requirements file in first and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 4: Copy the rest of your application code into the container
COPY . .

# Step 5: Tell Azure what port my app will run on.
EXPOSE 8000

# Step 6: Define the command to run my app using Gunicorn
# This tells Gunicorn to find the 'app' object inside your 'app.py' file.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]