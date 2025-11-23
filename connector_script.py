import boto3
import paramiko
from scp import SCPClient

# Constantes
master_ip = '52.2.245.12'
slaves_ip = ['34.207.201.212', '54.237.233.68', '3.81.94.198', '54.91.226.85', '52.55.90.184', '54.242.242.220']

# Crear una sesión de boto3
ec2_client = boto3.client('ec2', region_name='us-east-1')

# Configurar la conexión SSH
key_path = 'id_rsa'
key = paramiko.RSAKey.from_private_key_file(key_path)
ssh_client = paramiko.SSHClient()
ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())


### CONFIGURAR INSTANCIAS SLAVES CON LO NECESARIO
# Conectar a las instancias slaves via SSH
for slave_ip in slaves_ip:
    ssh_client.connect(slave_ip, username='ec2-user', pkey=key)
    # Subir el modelo
    scp_client = SCPClient(ssh_client.get_transport())
    scp_client.put('cats_vs_dogs_cnn.pth', '/home/ec2-user/cats_vs_dogs_cnn.pth')
    # TODO: Subir archivo backend
    scp_client.put('backend_demo.py', '/home/ec2-user/backend_demo.py')
    # Instalar docker en los slaves
    commands = [
        'sudo yum update -y',
        'sudo yum install docker -y',
        'sudo systemctl start docker',
        'sudo systemctl enable docker',
        'sudo usermod -aG docker ec2-user'
    ]

    for command in commands:
        stdin, stdout, stderr = ssh_client.exec_command(command)
        print(stdout.read().decode(), stderr.read().decode())

    # Correr la aplicacion backend en todos los slaves
    dockerfile_content = """
    FROM pytorch/pytorch:latest
    WORKDIR /app
    COPY backend_demo.py /app/backend_demo.py
    COPY cats_vs_dogs_cnn.pth /app/cats_vs_dogs_cnn.pth
    RUN pip install fastapi uvicorn pillow
    CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
    """

    sftp_client = ssh_client.open_sftp()
    with sftp_client.open('/home/ec2-user/Dockerfile', 'w') as dockerfile:
        dockerfile.write(dockerfile_content)
    
    commands = [
        'docker build -t fastapi_app .',
        'docker run -d -p 8000:8000 fastapi_app'
    ]

    for command in commands:
        stdin, stdout, stderr = ssh_client.exec_command(command)
        print(stdout.read().decode(), stderr.read().decode())

    # Cerrar la conexión SSH
    ssh_client.close()


### CONFIGURAR INSTANCIA MASTER CON NGINX
# Instalar NGINX en la instancia master
ssh_client.connect(master_ip, username='ec2-user', pkey=key)

# Archivo de configuración de NGINX (master)
upstream_servers = "\n        ".join([f"server {ip}:8000;" for ip in slaves_ip])

nginx_config = f"""
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {{
    worker_connections 1024;
}}

http {{
    upstream fastapi_app {{
        {upstream_servers}
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
"""

# Crear/modificar el archivo de configuración en /etc/nginx/nginx.conf
sftp_client = ssh_client.open_sftp()
with sftp_client.open('/home/ec2-user/nginx.conf', 'w') as conf_file:
    conf_file.write(nginx_config)
sftp_client.close()
# Mover el archivo de configuración a la ubicación correcta y reiniciar NGINX
commands = [
    'sudo yum update -y',
    'sudo yum install nginx -y',
    'sudo systemctl start nginx',
    'sudo systemctl enable nginx',
    "sudo mv /home/ec2-user/nginx.conf /etc/nginx/nginx.conf",
    "sudo systemctl restart nginx"
]
for cmd in commands:
    stdin, stdout, stderr = ssh_client.exec_command(cmd)
    print(stdout.read().decode(), stderr.read().decode())

print("Configuración en master y slaves completada.")
# Cerrar la conexión SSH
ssh_client.close()
