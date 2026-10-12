"""Access boundary for operator-managed LAN/public mappings."""
import base64
import ipaddress
import os
from pathlib import Path
import re
import secrets
from urllib.parse import urlsplit


def authority(value):
    if not isinstance(value, str) or not value or len(value) > 300 or re.search(r'[\s/@?#\\]', value):
        return None
    try:
        parsed = urlsplit('//' + value)
        host, port = parsed.hostname, parsed.port
        if not host or parsed.username is not None or parsed.path:
            return None
        if ':' in host:
            host = str(ipaddress.IPv6Address(host))
        elif not re.fullmatch(r'[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?', host):
            return None
        if port is not None and not 1 <= port <= 65535:
            return None
        return host.lower(), port
    except ValueError:
        return None


def load_token(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 256:
        raise ValueError('Access token must be a small regular file')
    if os.name == 'posix' and path.stat().st_mode & 0o077:
        raise ValueError('Access token file must have mode 0600')
    token = path.read_text(encoding='ascii').strip()
    if not re.fullmatch(r'[A-Za-z0-9_-]{32,128}', token):
        raise ValueError('Access token must contain 32–128 URL-safe characters')
    return token


class AccessPolicy:
    def __init__(self, remote=False, allowed_hosts=(), token=None, public_origins=()):
        self.remote, self.token = remote, token
        self.allowed_hosts = tuple(authority(value) for value in allowed_hosts)
        if any(value is None for value in self.allowed_hosts):
            raise ValueError('Invalid allowed host; use hostname[:port], not a URL')
        self.public_origins = tuple(self.origin_key(value) for value in public_origins)
        if any(value is None for value in self.public_origins):
            raise ValueError('Public origin must be an http/https URL without path, credentials or query')

    @staticmethod
    def origin_key(origin):
        try:
            parsed = urlsplit(origin)
            if parsed.scheme not in ('http', 'https') or parsed.username is not None or parsed.path or parsed.query or parsed.fragment:
                return None
            endpoint = authority(parsed.netloc)
            if endpoint is None:
                return None
            return parsed.scheme, endpoint[0], endpoint[1] or (443 if parsed.scheme == 'https' else 80)
        except (ValueError, TypeError):
            return None

    def host_ok(self, host, port):
        parsed = authority(host)
        if parsed is None:
            return False
        if self.allowed_hosts:
            return parsed in self.allowed_hosts
        return self.remote or parsed in (('127.0.0.1', port), ('localhost', port))

    def origin_ok(self, origin, host):
        if origin is None:
            return True  # Non-browser clients must still send the CSRF token.
        incoming, target = self.origin_key(origin), authority(host)
        if incoming is None or target is None:
            return False
        if incoming in self.public_origins:
            return True
        default = 443 if incoming[0] == 'https' else 80
        return incoming[1] == target[0] and incoming[2] == (target[1] or default)

    def authenticated(self, authorization):
        if self.token is None:
            return True
        try:
            kind, encoded = str(authorization or '').split(' ', 1)
            if kind.lower() != 'basic' or len(encoded) > 512:
                return False
            raw = base64.b64decode(encoded, validate=True).decode('utf-8')
            username, password = raw.split(':', 1)
            return username == 'operator' and secrets.compare_digest(password.encode(), self.token.encode())
        except (ValueError, UnicodeError):
            return False
