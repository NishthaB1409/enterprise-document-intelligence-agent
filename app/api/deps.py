"""Request-scoped access to the things built once at startup.

Handlers depend on these rather than importing `build_services`, so a test can
swap the whole stack by constructing the app with different `Services`.
"""

from typing import Annotated

from fastapi import Depends, Request

from app.config import Settings
from app.services import Services


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_services(request: Request) -> Services:
    return request.app.state.services


SettingsDep = Annotated[Settings, Depends(get_settings)]
ServicesDep = Annotated[Services, Depends(get_services)]
