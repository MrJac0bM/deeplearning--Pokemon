"""Local-only inference server. Uses the existing project and its preprocessing."""
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import argparse
import base64
import hashlib
import json
import sys
import tempfile
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, default=Path.home() / 'Downloads/pokemon-demo')
    p.add_argument('--port', type=int, default=8765)
    a = p.parse_args()
    project = a.project.resolve()
    sys.path.insert(0, str(project / 'python'))
    from procesamiento import preparar_imagen
    import tensorflow as tf
    modeldir = project / 'modelo_final'
    classes = (modeldir / 'clases.txt').read_text(encoding='utf-8').splitlines()
    print('Loading your trained model…', flush=True)
    model = tf.keras.models.load_model(modeldir / 'modelo.keras', compile=False)
    if model.output_shape[-1] != len(classes):
        raise ValueError('Model and class list do not match')
    web = Path(__file__).parent / 'web'

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, payload, mime='application/json'):
            data = json.dumps(payload, ensure_ascii=False).encode() if mime == 'application/json' else payload
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/api/status':
                return self.send(200, {'ready': True, 'classes': len(classes), 'model': 'EfficientNetB0'})
            routes = {'/': ('index.html','text/html; charset=utf-8'), '/app.js': ('app.js','text/javascript'), '/style.css': ('style.css','text/css')}
            if self.path not in routes:
                return self.send(404, {'error': 'Not found'})
            name, mime = routes[self.path]
            self.send(200, (web / name).read_bytes(), mime)

        def do_POST(self):
            if self.path != '/api/predict':
                return self.send(404, {'error': 'Not found'})
            # Prevent websites from submitting requests to this local server.
            origin = self.headers.get('Origin')
            if origin and origin != f'http://127.0.0.1:{a.port}':
                return self.send(403, {'error': 'Open the app using its 127.0.0.1 address.'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 12_000_000:
                    return self.send(413, {'error': 'Choose a JPEG or PNG under 8 MB.'})
                body = json.loads(self.rfile.read(size))
                raw = base64.b64decode(body['image'], validate=True)
                if len(raw) > 8_000_000:
                    raise ValueError('Image must be under 8 MB')
                if not (raw.startswith(b'\x89PNG\r\n\x1a\n') or raw.startswith(b'\xff\xd8\xff')):
                    raise ValueError('Only actual JPEG and PNG images are supported')
                start = time.perf_counter()
                with tempfile.TemporaryDirectory(prefix='pokemon-center-') as temp:
                    path = Path(temp) / 'image'
                    path.write_bytes(raw)
                    image = preparar_imagen(str(path))
                    scores = model(image[None], training=False).numpy()[0]
                top = scores.argsort()[::-1][:5]
                result = {'top5': [{'class': classes[i], 'score': float(scores[i])} for i in top],
                          'seconds': round(time.perf_counter()-start, 3),
                          'model': 'EfficientNetB0', 'image_sha256': hashlib.sha256(raw).hexdigest(),
                          'note': 'Softmax scores are not calibrated certainty. Unknown classes are not detected.'}
                self.send(200, result)
            except (ValueError, KeyError, TypeError) as error:
                self.send(400, {'error': str(error)})
            except Exception:
                import traceback
                traceback.print_exc()
                self.send(500, {'error': 'Could not process this image. Try a smaller, valid JPEG or PNG.'})

    server = HTTPServer(('127.0.0.1', a.port), Handler)
    print(f'Pokémon Center ready: http://127.0.0.1:{a.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nCenter closed.')
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
