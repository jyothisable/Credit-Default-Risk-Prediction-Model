FROM python:3.12-slim-bullseye

# Set working directory
WORKDIR /app

# copy source code
COPY . .

# install dependencies with poetry
RUN pip install poetry
RUN poetry install

# Expose port
EXPOSE 5000

# Run the app
CMD ["poetry", "run", "python", "flask_app.py"]