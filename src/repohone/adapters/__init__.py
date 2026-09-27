from . import claude

ADAPTERS = {claude.NAME: claude}


def get(name: str):
    return ADAPTERS.get(name)
