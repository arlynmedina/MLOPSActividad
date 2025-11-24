import time
import paramiko
from scp import SCPClient


# CONFIGURACION
MASTER_IP = "3.87.113.157"

SLAVES_PUBLIC_IPS = [
    "3.95.2.13",
    "3.94.89.187",
    "35.172.138.196",
    "98.87.7.59",
    "13.218.211.127",
    "98.81.190.107",
]

SLAVES_PRIVATE_IPS = [
    "10.0.1.72",
    "10.0.1.142",
    "10.0.1.77",
    "10.0.1.49",
    "10.0.1.200",
    "10.0.1.123",
]

KEY_PATH = "id_rsa"
USERNAME = "ec2-user"

MODEL_FILE = "cats_vs_dogs_cnn.pth"
BACKEND_FILE = "inference_server.py"
API_PORT = 8000


# UTILS
def ssh_connect(ip: str) -> paramiko.SSHClient:
    key = paramiko.RSAKey.from_private_key_file(KEY_PATH)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    print(f"\nConectando a {ip} ...")
    ssh.connect(ip, username=USERNAME, pkey=key)
    return ssh


def run_commands(ssh: paramiko.SSHClient, commands: list[str]) -> None:
    for cmd in commands:
        print(f"Ejecutando: {cmd}")
        _, stdout, stderr = ssh.exec_command(cmd)
        out = stdout.read().decode()
        err = stderr.read().decode()
        if out:
            print(out)
        if err:
            print(err)


def upload_files(ssh: paramiko.SSHClient, files: list[tuple[str, str]]) -> None:
    with SCPClient(ssh.get_transport()) as scp:
        for src, dest in files:
            print(f"Subiendo {src} -> {dest}")
            scp.put(src, dest)


# CONFIGURAR SLAVES
def setup_slaves() -> None:
    for ip in SLAVES_PUBLIC_IPS:
        ssh = ssh_connect(ip)

        # Subir backend + modelo
        upload_files(ssh, [
            (MODEL_FILE, f"/home/{USERNAME}/{MODEL_FILE}"),
            (BACKEND_FILE, f"/home/{USERNAME}/{BACKEND_FILE}"),
        ])

        # Instalar Docker
        run_commands(ssh, [
            "sudo yum update -y",
            "sudo yum install -y docker",
            "sudo systemctl start docker",
            "sudo systemctl enable docker",
            f"sudo usermod -aG docker {USERNAME}",
        ])

        time.sleep(2)

        # Crear Dockerfile
        dockerfile = f"""
FROM pytorch/pytorch:latest

WORKDIR /app

COPY {BACKEND_FILE} /app/{BACKEND_FILE}
COPY {MODEL_FILE} /app/{MODEL_FILE}

RUN pip install fastapi uvicorn pillow python-multipart

CMD ["uvicorn", "inference_server:app", "--host", "0.0.0.0", "--port", "{API_PORT}"]
""".lstrip()

        print("Subiendo Dockerfile...")
        sftp = ssh.open_sftp()
        with sftp.open(f"/home/{USERNAME}/Dockerfile", "w") as f:
            f.write(dockerfile)
        sftp.close()

        # Construir y ejecutar la app backend
        run_commands(ssh, [
            "cd /home/ec2-user && sudo docker build -t fastapi_app .",
            "sudo docker rm -f fastapi_app || true",
            f"sudo docker run -d -p {API_PORT}:{API_PORT} --name fastapi_app fastapi_app",
        ])

        print(f"Slave configurado correctamente: {ip}")
        ssh.close()


# CONFIGURAR MASTER
def setup_master() -> None:
    ssh = ssh_connect(MASTER_IP)

    # Usamos IPs PRIVADAS en el upstream
    upstream = "\n        ".join(
        [f"server {ip}:{API_PORT};" for ip in SLAVES_PRIVATE_IPS])

    nginx_conf = f"""
user nginx;
worker_processes auto;

error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {{
    worker_connections 1024;
}}

http {{
    upstream fastapi_app {{
        {upstream}
    }}

    server {{
        listen 80;

        location / {{
            proxy_pass http://fastapi_app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }}
    }}
}}
""".lstrip()

    print("Subiendo configuracion de NGINX...")
    sftp = ssh.open_sftp()
    with sftp.open(f"/home/{USERNAME}/nginx.conf", "w") as f:
        f.write(nginx_conf)
    sftp.close()

    run_commands(ssh, [
        "sudo yum update -y",
        "sudo yum install -y nginx",
        "sudo systemctl start nginx",
        "sudo systemctl enable nginx",
        f"sudo mv -f /home/{USERNAME}/nginx.conf /etc/nginx/nginx.conf",
        "sudo nginx -t",
        "sudo systemctl restart nginx",
    ])

    print("Master configurado correctamente.")
    ssh.close()


if __name__ == "__main__":
    print("\n=== CONFIGURANDO SLAVES ===")
    setup_slaves()

    print("\n=== CONFIGURANDO MASTER ===")
    setup_master()

    print("\n=== CONFIGURACION COMPLETA ===\n")
