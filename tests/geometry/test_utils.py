def assert_attributes(class_, dict_, exclude=None) -> None:
    for key, value in dict_.items():
        if exclude is not None:
            if key in exclude:
                continue
        assert getattr(class_, key) == value
        assert type(getattr(class_, key)) == type(value)