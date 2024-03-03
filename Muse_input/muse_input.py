from pythonosc import dispatcher, osc_server

from queue import Queue

def setup_server(ip: str, port: int, q: Queue):
    def push_to_queue(address: str, *args):
        q.put(args)
    dis = dispatcher.Dispatcher()
    dis.map("/muse/eeg", push_to_queue)

    server = osc_server.ThreadingOSCUDPServer((ip, port), dis)
    print("Listening on UDP port " + str(port))
    return server