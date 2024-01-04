class Config:

    def __init__(self, params):
        if params is not None:
            for key, value in params.items():
                if isinstance(value, dict):
                    setattr(self, key, Config(value))
                else:
                    setattr(self, key, value)