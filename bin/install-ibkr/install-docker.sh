# 1) Remove old packages (safe if none installed)
sudo apt-get remove -y docker docker-engine docker.io containerd runc || true

# 2) Pre-reqs
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

# 3) Docker APT repo (ARM64-aware)
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
 | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release; echo $VERSION_CODENAME) stable" \
 | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 4) Install Docker Engine + Buildx + Compose v2
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 5) Add your user to the docker group (no sudo)
sudo usermod -aG docker $USER
newgrp docker

# 6) Sanity checks
docker --version
docker compose version
