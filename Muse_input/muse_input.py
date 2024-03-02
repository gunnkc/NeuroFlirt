from datetime import datetime
from pythonosc import dispatcher, osc_server

from queue import Queue

def recieve_inputs(ip: str, port: int, q: Queue):
    def push_to_queue(address: str, *args):
        q.put(args)
    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("/muse/eeg", push_to_queue)

    server = osc_server.ThreadingOSCUDPServer((ip, port), dispatcher)
    print("Listening on UDP port "+str(port))
    return server.serve_forever