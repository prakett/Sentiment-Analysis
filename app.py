import subprocess
import sys
import time
import socket
import webbrowser


HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}"


def wait_for_server(host, port, timeout=60):
    """Wait until the FastAPI server is accepting connections."""
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)

    return False


def main():
    print("Starting Sentiment Analysis API...")
    print()

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api:app",
            "--host",
            HOST,
            "--port",
            str(PORT)
        ]
    )

    try:
        print("Waiting for server to start...")

        if wait_for_server(HOST, PORT):

            print()
            print("=" * 50)
            print("Sentiment Analysis application is running!")
            print(f"Website : {URL}")
            print(f"API     : {URL}/docs")
            print("=" * 50)
            print()

            webbrowser.open(URL)

            print("Browser opened automatically.")
            print("Press CTRL+C to stop the server.")

            # Keep this script alive while Uvicorn is running
            process.wait()

        else:
            print("Server failed to start within the timeout.")
            process.terminate()

    except KeyboardInterrupt:
        print("\nStopping server...")
        process.terminate()

        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

        print("Server stopped.")


if __name__ == "__main__":
    main()