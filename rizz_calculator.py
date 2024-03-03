import threading
from queue import Queue

from Muse_input.muse_input import setup_server

"""
TODO:
    - Add preprocessing to _batch_inputs
        - Currently _batch_inputs adds [TP9, TP10, AF7, AF8] to queue
        - Should calculate summary statistics
    - Add prediction model(s)
"""

class RizzCalculator:
    IP = "0.0.0.0"
    PORT = 5000
    POLL_TIME = 0.03
    ROW_BATCH_SIZE = 990

    def __init__(self):
        self.running = False

        self._raw_inputs = Queue()      # Raw inputs by Muse (list of nums)
        self._batched_inputs = Queue()  # Preprocessed inputs
        self._metrics = Queue()          # Predicted metrics

        # Server - need access to server to shut it down later
        self._server = setup_server(self.IP, self.PORT, self._raw_inputs)

        # Threads -- need to run each process in "parallel"
        self._muse_thread = threading.Thread(target=self._server.serve_forever, args=(self.POLL_TIME))
        self._batch_thread = threading.Thread(target=self._batch_inputs)
        self._prediction_thread = threading.Thread(target=self._make_predictions)


    def start_predictions(self):
        # Starts all threads
        self.running = True
        self._muse_thread.start()
        self._batch_thread.start()
        self._prediction_thread.start()


    def stop_predictions(self):
        # Closes all threads
        self.running = False
        self._server.shutdown()
        self._muse_thread.join()
        self._batch_thread.join()
        self._prediction_thread.join()


    def get_metrics(self) -> list[int]:
        """
        Sends batch of most recent metrics.
        No duplicates should be sent.
        """
        # TODO: define metrics and return them
        raise NotImplementedError
    

    def _make_predictions(self):
        """
        Calculates Arousal and Engagement scores based on 
        batched EEG signals.
        """
        while self.running:
            batched_input = self._batched_inputs.get()
            # Make prediction
            self._batched_inputs.task_done()
            # Calculate metrics and push on to queue
            self._metrics.put()
    

    def _batch_inputs(self, address: str, *args):
        """
        Preprocesses raw inputs from Muse into groups
        and extracts features to predict on.
        """
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
