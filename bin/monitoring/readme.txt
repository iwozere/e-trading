this vector.toml is used by systemd service 'vector.service'.

Vector forwards logs to a remote Loki instance.
Example remote Loki:
{
  "url": "https://loki.example.com/loki/api/v1/push",
  "username": "remote-user",
  "password": "[PASSWORD]"
} 