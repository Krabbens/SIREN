import pty
import os
import time

def read_until(fd, marker):
    buf = b""
    while True:
        try:
            chunk = os.read(fd, 1)
            if not chunk:
                break
            buf += chunk
            if marker.encode() in buf:
                return buf
        except OSError:
            break
    return buf

pid, fd = pty.fork()

if pid == 0:
    # Child
    os.execlp("ssh-copy-id", "ssh-copy-id", "-p", "2222", "-o", "StrictHostKeyChecking=no", "kubs@192.168.0.102")
else:
    # Parent
    try:
        # We might get a password prompt directly if strict host checking is off or host is known
        # Or we might get other prompts.
        # Since I added StrictHostKeyChecking=no, we should skip the yes/no part usually, 
        # but ssh-copy-id calls ssh which might still ask if not in known_hosts.
        # Actually StrictHostKeyChecking=no suppresses the yes/no.
        
        # Wait for password prompt
        print("Waiting for password prompt...")
        # Simple read loop
        buf = b""
        while True:
            try:
                chunk = os.read(fd, 1024)
                if not chunk:
                    break
                buf += chunk
                sys_stdout = os.write(1, chunk) # Echo to stdout for debugging
                
                if b"password:" in buf:
                    time.sleep(1) # Wait a bit to be safe
                    os.write(fd, b"zaqzaq\n")
                    print("\nPassword sent.")
                    break
            except OSError:
                break
        
        # Read the rest
        while True:
            try:
                chunk = os.read(fd, 1024)
                if not chunk:
                    break
                os.write(1, chunk)
            except OSError:
                break
                
        _, status = os.waitpid(pid, 0)
        print(f"Exited with status {status}")
        
    except Exception as e:
        print(f"Error: {e}")
