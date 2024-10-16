FROM python:3.12-slim-bullseye

# Set working directory
WORKDIR /app

# copy source code
COPY . .

# install dependencies with poetry
RUN pip install poetry==1.8.3
RUN poetry install --no-dev

# Expose port
EXPOSE 8000

# Run the app
CMD ["poetry", "run", "python", "fastapi_app.py"]