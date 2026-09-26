"""HTTPS-only public acquisition with credential-safe redirects."""

from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


def require_https(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("acquisition requires an HTTPS URL without embedded credentials")


class _HTTPSRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        require_https(newurl)
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and urlsplit(req.full_url).netloc != urlsplit(newurl).netloc:
            redirected.remove_header("Authorization")
        return redirected


def open_https(request: Request, *, timeout: float):
    require_https(request.full_url)
    return build_opener(_HTTPSRedirect()).open(request, timeout=timeout)
