from locust import HttpUser, task, between
import time

@task
class Fastapi(HttpUser):
    wait_time = between(2, 5)
    host = "https://prediction-eewt.onrender.com"

    @task
    def predict(self):
        self.client.post(url = "/predict")
