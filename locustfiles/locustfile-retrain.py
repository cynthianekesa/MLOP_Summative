from locust import HttpUser, task, between
import time

@task
class Fastapi(HttpUser):
    wait_time = between(2, 5)
    host = "https://retrainapi.onrender.com"

    @task
    def retrain(self):
        self.client.post(url = "/upload/retrain")

    @task
    def evaluate(self):
        self.client.post(url = "/upload/evaluate")