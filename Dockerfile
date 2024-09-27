FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

EXPOSE 7200

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["streamlit", "run", "interface.py"]

# CMD ["interface.py"]