provider "aws" {
  region = "us-east-1"
}

# Generar clave SSH
resource "tls_private_key" "deployer" {
  algorithm = "RSA"
  rsa_bits  = 2048
}

resource "local_file" "private_key" {
  filename        = "id_rsa"
  content         = tls_private_key.deployer.private_key_pem
  file_permission = "0600"
}

resource "aws_key_pair" "deployer" {
  key_name   = "deployer-key"
  public_key = tls_private_key.deployer.public_key_openssh
}

# Security Group
resource "aws_security_group" "pods_security_group" {
  name_prefix = "pods-security-group"

  # HTTP para master (entrada externa)
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # SSH para debugging / acceso
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Tráfico interno entre master y slaves
  ingress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["10.0.0.0/16"] # o 0.0.0.0/0 si no tienes VPC personalizada
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Instancias slaves
resource "aws_instance" "slaves" {
  count         = 6
  ami           = "ami-025ca978d4c1d9825"
  instance_type = "t3.micro"
  key_name      = aws_key_pair.deployer.key_name

  security_groups = [
    aws_security_group.pods_security_group.name
  ]

  root_block_device {
    volume_size = 32
  }

  tags = {
    Name = "slave-${count.index + 1}"
  }
}

# Instacia master
resource "aws_instance" "master_gateway" {
  ami           = "ami-025ca978d4c1d9825"
  instance_type = "t3.micro"
  key_name      = aws_key_pair.deployer.key_name

  security_groups = [
    aws_security_group.pods_security_group.name
  ]

  tags = {
    Name = "master_gateway"
  }
}

# Outputs
output "master_ip" {
  value = aws_instance.master_gateway.public_ip
}

output "slaves_private_ips" {
  value = aws_instance.slaves[*].private_ip
}

output "slaves_public_ips" {
  value = aws_instance.slaves[*].public_ip
}
