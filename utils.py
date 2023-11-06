def get_methods(_object):
    return [method_name for method_name in dir(_object) if callable(getattr(_object, method_name))]