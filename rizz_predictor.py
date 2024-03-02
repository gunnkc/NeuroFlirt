import threading
from queue import Queue

from Muse_input.muse_input import recieve_inputs

class RizzPredictor:
    IP = "0.0.0.0"
    PORT = 5000
    ROW_BATCH_SIZE = 30

    def __init__(self):
        self.running = False

        self._raw_inputs = Queue()
        self._batched_inputs = Queue()

        serve_function = self._setup_server()

        self.muse_thread = threading.Thread(target=serve_function)
        self.batch_thread = threading.Thread(target=self._batch_inputs)
        self.prediction_thread = threading.Thread(target=self.make_predictions)


    def start_predictions(self):
        self.running = True
        self.muse_thread.start()
        self.batch_thread.start()
        self.prediction_thread.start()


    def stop_predictions(self):
        self.running = False
        self.muse_thread.join()
        self.batch_thread.join()
        self.prediction_thread.join()
    

    def make_predictions(self):
        while self.running:
            batched_input = self._batched_inputs.get()
            # Process input
            # Make prediction
            self._batched_inputs.task_done()
    

    def _batch_inputs(self, address: str, *args):
        while self.running:
            batch = []

            while len(batch) < self.ROW_BATCH_SIZE:
                item = self._raw_inputs.get()
                batch.append(item)
                self._raw_inputs.task_done()
            
            if batch:
                # process batch
                processed = batch
                self._batched_inputs.put(processed)
    
    
    def _setup_server(self) -> callable:
        return recieve_inputs(self.IP, self.PORT, self._raw_inputs)
