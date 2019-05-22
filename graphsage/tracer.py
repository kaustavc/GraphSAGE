from datetime import datetime


class Tracer:
    def __init__(self, description, **kwargs):
        self.description = description
        self.params = kwargs
        self.start_time = None
        self.last_update_time = None

    def __enter__(self):
        self.start_time = self.last_update_time = datetime.now()
        print("[{}] START [{}] params={}".format(self.start_time, self.description, self.params))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        self.elapsed_seconds = (self.end_time-self.start_time).seconds
        print("[{}] END [{}] duration={}s".format(self.end_time, self.description, self.elapsed_seconds))

    def update(self, **kwargs):
        now = datetime.now()
        total_elapsed = (now-self.start_time).seconds
        delta_elapsed = (now-self.last_update_time).seconds
        self.last_update_time = now
        print("[{}] UPDATE [{}] params={} delta={}s total={}s".format(now, self.description, kwargs, delta_elapsed, total_elapsed))

    def duration(self):
        return self.elapsed_seconds


# Testing code
if __name__ == '__main__':
    import time
    with Tracer("Addition", a=1, b=2) as tracer:
        time.sleep(2)
        tracer.update(status="1/3")
        time.sleep(2)
        tracer.update(status="2/3")
        time.sleep(2)


