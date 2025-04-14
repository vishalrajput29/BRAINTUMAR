FROM python:3.9-slim-buster

# Set working directory
WORKDIR /app

# Copy app code into the container
COPY . /app

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port App Runner expects
EXPOSE 5000

# Run the app using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
