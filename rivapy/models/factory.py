_factory_entries = None


def _factory(depp=None):
    global _factory_entries
    if _factory_entries is None:
        _factory_entries = {}
    return _factory_entries


def create(data: dict) -> object:
    if not isinstance(data, dict):
        return data
    if "cls" not in data.keys():
        raise Exception("Given dictionary has not a cls key, unable create from factory.")
    cls = data["cls"]

    if cls not in _factory_entries.keys():
        raise Exception("No class registered for given cls key " + data["cls"])
    return _factory_entries[cls].from_dict(data)
