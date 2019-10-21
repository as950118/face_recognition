FROM python:3.6

# Installs the actual app
WORKDIR /app

COPY . .
RUN pip install cmake
RUN pip install --no-cache-dir -r requirements.txt

# These could be done at the yaml level but this allows for better tests
EXPOSE 5000
CMD ["python", "./api.py"]