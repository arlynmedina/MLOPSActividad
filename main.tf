provider "aws" {
  region = "us-east-1"
}

# VPC
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = "pods-vpc"
  }
}

resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "pods-igw"
  }
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  map_public_ip_on_launch = true

  tags = {
    Name = "pods-public-subnet"
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }

  tags = {
    Name = "pods-public-rt"
  }
}

resource "aws_route_table_association" "public_assoc" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

# SSH Key Pair
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
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = [aws_vpc.main.cidr_block]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "pods-sg"
  }
}

# Slaves
resource "aws_instance" "slaves" {
  count         = 6
  ami           = "ami-0fa3fe0fa7920f68e"
  instance_type = "t3.micro"
  key_name      = aws_key_pair.deployer.key_name

  subnet_id                   = aws_subnet.public.id
  vpc_security_group_ids      = [aws_security_group.pods_security_group.id]
  associate_public_ip_address = true

  root_block_device {
    volume_size = 32
  }

  tags = {
    Name = "slave-${count.index + 1}"
    Role = "api-node"
  }
}

# Master
resource "aws_instance" "master_gateway" {
  ami           = "ami-0fa3fe0fa7920f68e"
  instance_type = "t3.micro"
  key_name      = aws_key_pair.deployer.key_name

  subnet_id                   = aws_subnet.public.id
  vpc_security_group_ids      = [aws_security_group.pods_security_group.id]
  associate_public_ip_address = true

  tags = {
    Name = "master_gateway"
    Role = "load-balancer"
  }
}

output "master_public_ip" {
  value = aws_instance.master_gateway.public_ip
}

output "slaves_private_ips" {
  value = aws_instance.slaves[*].private_ip
}

output "slaves_public_ips" {
  value = aws_instance.slaves[*].public_ip
}
