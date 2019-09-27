import queue
import threading
import traceback


class NoTasksException(RuntimeError):
    pass


class Task:
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        return self.func(*self.args, **self.kwargs)


class StopTask(Task):
    def __init__(self):
        super().__init__(func=None)


class Worker(threading.Thread):
    def __init__(self):
        super().__init__()
        self.task_queue = queue.Queue()
        self.return_queue = queue.Queue()
        self.running_task = None

    def run(self):
        super().run()

        while True:
            self.running_task = self.task_queue.get()

            if isinstance(self.running_task, StopTask):
                self.running_task = None
                break

            try:
                ret = self.running_task.run()
                if self.task_queue.empty():
                    self.running_task = None

                self.return_queue.put(ret)
            except Exception as e:
                traceback.print_exc()
                print('Task failed with exception: {}'.format(e))

    def queue_task(self, func, *args, **kwargs):
        task = Task(func, *args, **kwargs)
        self.task_queue.put(task)

    def schedule_stop(self):
        self.task_queue.put(StopTask())

    def is_idle(self):
        return self.task_queue.empty() and self.running_task is None

    def wait_task(self, timeout=None):
        if not self.is_alive() or self.is_idle():
            raise NoTasksException()

        return self.return_queue.get(timeout=timeout)

    def wait_all_tasks(self, timeout=None):
        while True:
            try:
                yield self.wait_task(timeout=timeout)
            except NoTasksException:
                break


def workers_pool(n_workers):
    pool = []

    for i in range(n_workers):
        w = Worker()
        w.start()
        pool.append(w)

    return pool
