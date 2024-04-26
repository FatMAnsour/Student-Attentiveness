FROM python:3.10.11
WORKDIR / C:\Users\Fatima\Documents\Student-Attentiveness
COPY . .
CMD ["python", "./api.py"]

EXPOSE 5000


