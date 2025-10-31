import http.server, socketserver, webbrowser
from functools import partial
def launch_frontend(directory,port):
    Handler = partial(http.server.SimpleHTTPRequestHandler, directory=directory)
    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"Serving at http://localhost:{port}")
        webbrowser.open(f'http://localhost:{port}')
        httpd.serve_forever()
directory = './frontend'
port = 5173
launch_frontend(directory,port)
